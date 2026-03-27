"""
Reflection-Mode BPM Forward Model
===================================

Physical model (Beam Propagation Method in reflection geometry):

    The rMSBP model propagates a complex field through N layers of a
    3D sample placed on a reflective substrate.  For each illumination
    angle the pipeline is:

    Forward (downward):
        For each layer i = 0, ..., N-1:
            u = P(dz_layer) · u       (free-space propagation)
            u = Q(dn_i, dz_layer) · u (scattering by RI slice)
            u = P(dz_gap) · u         (gap propagation)

    Reflection:
        u *= -1                        (perfect mirror, π phase shift)

    Backward (upward):
        For each layer i = N-1, ..., 0:
            u = P(dz_gap) · u         (gap propagation)
            u = P(dz_layer) · u       (free-space propagation)
            u = Q(dn_i, dz_layer) · u (scattering by RI slice)

    Detection:
        u = P(-total_depth) · u       (back-propagate to focal plane)
        u = Pupil · u                 (NA filter in Fourier space)
        I = |u|²                      (intensity measurement)

    P operator (propagation):
        In Fourier space: â *= exp(i·kz·dz) · eva_mask

    Q operator (scattering):
        In real space: u *= exp(i · Δn · phase_factor · dz)
        where phase_factor = 2π·res_z / n0

    This is the paraxial BPM (single complex field), simpler than the
    non-paraxial SSNP (state vector with z-derivative).

Reference: Zhu et al., "rMS-FPT", arXiv:2503.12246 (2025)
Implementation uses PyTorch for GPU acceleration and autograd support.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ReflectionBPMConfig:
    """
    Physical parameters for the reflection-mode BPM model.

    Attributes
    ----------
    volume_shape : tuple of int
        (Nz, Ny, Nx) — number of layers and lateral pixels.
    res : tuple of float
        Normalised resolution (dx·n0/λ, dy·n0/λ, dz·n0/λ).
    n0 : float
        Background refractive index.
    NA_obj : float
        Objective numerical aperture.
    NA_illu : float
        Illumination numerical aperture.
    wavelength_um : float
        Illumination wavelength in micrometers.
    res_um : tuple of float
        Voxel size in micrometers (dx, dy, dz).
    n_angles : int
        Number of illumination angles.
    dz_layer : float
        BPM propagation distance for each layer scatter step (z-pixel units).
    dz_gap : float
        BPM propagation distance for gaps between layers (z-pixel units).
    """
    volume_shape: tuple
    res: tuple
    n0: float
    NA_obj: float
    NA_illu: float
    wavelength_um: float
    res_um: tuple
    n_angles: int
    dz_layer: float
    dz_gap: float

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ReflectionBPMConfig":
        """Construct config from a metadata dictionary.

        Note: ``res`` is computed as ``res_um * n0 / wavelength_um``
        (normalised, dimensionless), matching the convention where the
        ssnp package internally computes ``res = xyz * n0 / lambda0``.
        """
        n0 = metadata["n0"]
        wavelength = metadata["wavelength_um"]
        res_um = tuple(metadata["res_um"])
        res = tuple(r * n0 / wavelength for r in res_um)
        return cls(
            volume_shape=tuple(metadata["volume_shape"]),
            res=res,
            n0=n0,
            NA_obj=metadata["NA_obj"],
            NA_illu=metadata.get("NA_illu", metadata["NA_obj"]),
            wavelength_um=wavelength,
            res_um=res_um,
            n_angles=metadata["n_angles"],
            dz_layer=metadata.get("dz_layer", 0.5),
            dz_gap=metadata.get("dz_gap", 10.0),
        )


class ReflectionBPMForwardModel:
    """
    Reflection-mode BPM forward model for multi-slice Fourier
    ptychographic tomography.

    Parameters
    ----------
    config : ReflectionBPMConfig
        Physical parameters.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, config: ReflectionBPMConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.complex128
        self.real_dtype = torch.float64

        nz, ny, nx = config.volume_shape
        self.nz = nz
        self.ny = ny
        self.nx = nx

        # Precompute frequency-domain quantities
        self.c_gamma = self._compute_c_gamma()
        self.kz = self._compute_kz()
        self.eva_mask = self._compute_evanescent_mask()
        self.pupil = self._compute_pupil()

    # ──────────────────────────────────────────────────────────────────────
    # Precomputation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_c_gamma(self) -> torch.Tensor:
        """
        Compute cos(gamma) = sqrt(1 - sin²α - sin²β) on the 2D frequency grid.

        Returns shape (Ny, Nx) float64 tensor.
        """
        res = self.config.res
        fx = torch.fft.fftfreq(self.nx, dtype=self.real_dtype,
                               device=self.device) / res[0]
        fy = torch.fft.fftfreq(self.ny, dtype=self.real_dtype,
                               device=self.device) / res[1]
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        eps = 1e-8
        c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=eps))
        return c_gamma

    def _compute_kz(self) -> torch.Tensor:
        """
        Compute kz = c_gamma * (2π · res_z) on the 2D frequency grid.

        Returns shape (Ny, Nx) float64 tensor.
        """
        kz = self.c_gamma * (2.0 * np.pi * self.config.res[2])
        return kz.to(self.device)

    def _compute_evanescent_mask(self) -> torch.Tensor:
        """
        Damping mask for evanescent waves.

        exp(min((c_gamma - 0.2) * 5, 0))

        Returns shape (Ny, Nx) float64 tensor.
        """
        mask = torch.exp(torch.clamp((self.c_gamma - 0.2) * 5.0, max=0.0))
        return mask.to(self.device)

    def _compute_pupil(self) -> torch.Tensor:
        """
        Binary pupil function with smooth cutoff at NA_obj.

        Uses a sigmoid-like function centred at c_gamma = sqrt(1 - NA_obj²).
        Note: NA_obj is used directly (not NA_obj/n0) because the frequency
        grid is normalised by res = dx·n0/λ, so the n0 factor cancels.

        Returns shape (Ny, Nx) complex128 tensor.
        """
        cutoff = np.sqrt(1.0 - self.config.NA_obj**2)
        mask = torch.exp(torch.clamp(self.c_gamma - cutoff, max=0.01) * 10000.0)
        mask = mask / (1.0 + mask)
        return mask.to(dtype=self.dtype, device=self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Incident field
    # ──────────────────────────────────────────────────────────────────────

    def _make_incident_field(self, angle_idx: int) -> torch.Tensor:
        """
        Construct a tilted plane wave for the given illumination angle.

        The illumination direction is:
            kx_in = NA_illu · cos(2π·m/n_angles)
            ky_in = NA_illu · sin(2π·m/n_angles)

        The tilt is truncated to the nearest integer frequency for
        periodicity (matching the original code's ``trunc=True``).

        Parameters
        ----------
        angle_idx : int
            Index m of the illumination angle (0 to n_angles-1).

        Returns
        -------
        u : (Ny, Nx) complex128 tensor — tilted plane wave
        """
        na = self.config.NA_illu
        n_angles = self.config.n_angles
        res = self.config.res

        theta = 2.0 * np.pi * angle_idx / n_angles
        c_a = na * np.cos(theta)  # direction cosine along x
        c_b = na * np.sin(theta)  # direction cosine along y

        # Truncate to nearest integer frequency for periodicity
        norm_x = self.nx * res[0]
        norm_y = self.ny * res[1]
        c_a = int(c_a * norm_x) / norm_x
        c_b = int(c_b * norm_y) / norm_y

        # Pixel coordinates
        x = torch.arange(self.nx, dtype=self.real_dtype, device=self.device)
        y = torch.arange(self.ny, dtype=self.real_dtype, device=self.device)

        # Phase ramps
        phase_x = 2j * np.pi * c_a * res[0] * x
        phase_y = 2j * np.pi * c_b * res[1] * y

        tilt = (torch.exp(phase_y).unsqueeze(1) *
                torch.exp(phase_x).unsqueeze(0))
        tilt = tilt.to(dtype=self.dtype, device=self.device)

        return tilt

    # ──────────────────────────────────────────────────────────────────────
    # BPM operators
    # ──────────────────────────────────────────────────────────────────────

    def _bpm_propagate(self, u: torch.Tensor, dz: float) -> torch.Tensor:
        """
        BPM free-space propagation operator (P).

        In Fourier domain: â *= exp(i · kz · dz) · eva_mask

        Parameters
        ----------
        u  : (Ny, Nx) complex — field
        dz : float — propagation distance in z-pixel units

        Returns
        -------
        u_new : (Ny, Nx) complex — propagated field
        """
        a = torch.fft.fft2(u)
        prop = torch.exp(1j * self.kz * dz) * self.eva_mask
        a = a * prop.to(dtype=a.dtype)
        return torch.fft.ifft2(a)

    def _bpm_scatter(self, u: torch.Tensor, dn_slice: torch.Tensor,
                     dz: float) -> torch.Tensor:
        """
        BPM scattering operator (Q).

        In real space: u *= exp(i · Δn · phase_factor)
        where phase_factor = 2π · res_z / n0 · dz

        Parameters
        ----------
        u        : (Ny, Nx) complex — field
        dn_slice : (Ny, Nx) real — RI contrast Δn at this layer
        dz       : float — slice thickness in z-pixel units

        Returns
        -------
        u_scattered : (Ny, Nx) complex
        """
        phase_factor = 2.0 * np.pi * self.config.res[2] / self.config.n0 * dz
        phase = dn_slice.to(dtype=self.real_dtype) * phase_factor
        return u * torch.exp(1j * phase).to(dtype=u.dtype)

    def _reflect(self, u: torch.Tensor) -> torch.Tensor:
        """Reflection off perfect mirror: u *= -1 (π phase shift)."""
        return -u

    # ──────────────────────────────────────────────────────────────────────
    # Full forward model
    # ──────────────────────────────────────────────────────────────────────

    def forward_single(self, dn_volume: torch.Tensor,
                       angle_idx: int) -> torch.Tensor:
        """
        Simulate intensity image for a single illumination angle.

        Pipeline:
            1. Construct incident tilted plane wave
            2. BPM forward through N layers (propagate + scatter + gap)
            3. Reflect off mirror
            4. BPM backward through N layers in reverse
            5. Back-propagate to focal plane
            6. Apply pupil filter
            7. Take intensity |u|²

        Parameters
        ----------
        dn_volume : (Nz, Ny, Nx) real tensor — RI contrast volume
        angle_idx : int — illumination angle index

        Returns
        -------
        intensity : (Ny, Nx) real tensor
        """
        dz_layer = self.config.dz_layer
        dz_gap = self.config.dz_gap

        u = self._make_incident_field(angle_idx)

        # Forward propagation (downward through sample)
        for iz in range(self.nz):
            u = self._bpm_propagate(u, dz_layer)
            u = self._bpm_scatter(u, dn_volume[iz], dz_layer)
            u = self._bpm_propagate(u, dz_gap)

        # Reflect off substrate
        u = self._reflect(u)

        # Backward propagation (upward through sample, reversed order)
        for iz in range(self.nz - 1, -1, -1):
            u = self._bpm_propagate(u, dz_gap)
            u = self._bpm_propagate(u, dz_layer)
            u = self._bpm_scatter(u, dn_volume[iz], dz_layer)

        # Back-propagate to focal plane
        total_depth = self.nz * (dz_gap + dz_layer)
        u = self._bpm_propagate(u, -total_depth)

        # Apply pupil in Fourier domain
        a = torch.fft.fft2(u)
        a = a * self.pupil
        u = torch.fft.ifft2(a)

        # Intensity
        intensity = torch.abs(u) ** 2
        return intensity

    def forward(self, dn_volume: torch.Tensor,
                n_angles: int = None) -> torch.Tensor:
        """
        Simulate intensity images for all illumination angles.

        Parameters
        ----------
        dn_volume : (Nz, Ny, Nx) real tensor — RI contrast volume
        n_angles  : int or None — number of angles (defaults to config)

        Returns
        -------
        intensities : (n_angles, Ny, Nx) real tensor
        """
        if n_angles is None:
            n_angles = self.config.n_angles

        intensities = []
        for m in range(n_angles):
            I_m = self.forward_single(dn_volume, m)
            intensities.append(I_m)

        return torch.stack(intensities, dim=0)

    def simulate_measurements(self, dn_volume: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for forward()."""
        return self.forward(dn_volume)

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self):
        c = self.config
        return (
            f"ReflectionBPMForwardModel("
            f"volume={c.volume_shape}, "
            f"NA_obj={c.NA_obj}, "
            f"NA_illu={c.NA_illu}, "
            f"n0={c.n0}, "
            f"λ={c.wavelength_um}μm, "
            f"angles={c.n_angles}, "
            f"device={self.device})"
        )
