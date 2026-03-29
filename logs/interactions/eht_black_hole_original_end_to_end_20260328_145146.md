# Multi-Agent Pipeline Log


## Iteration 1 — Ticket: Planner

### Planner

**Plan:**
Here is a rigorous mathematical and algorithmic plan for the EHT black hole imaging task.

### [Problem Formulation]

The goal is to reconstruct a non-negative image, $\mathbf{x} \in \mathbb{R}^{N^2}$, from a set of interferometric observables. The image $\mathbf{x}$ represents the vectorized sky brightness distribution on an $N \times N$ grid.

**Forward Model:** The underlying physical model relates the image $\mathbf{x}$ to the ideal complex visibilities $\mathbf{y}_{\text{model}} \in \mathbb{C}^M$ via a linear transformation:
$$ \mathbf{y}_{\text{model}} = \mathbf{A}\mathbf{x} $$

- $\mathbf{x} \in \mathbb{R}^{4096}$: The vectorized image to be recovered ($N=64$). Must satisfy $\mathbf{x} \ge 0$ and $\sum_i x_i = F_{\text{total}}$.
- $\mathbf{A} \in \mathbb{C}^{M \times N^2}$: The measurement matrix, representing a Non-Uniform Discrete Fourier Transform (NUDFT). $M=421$ is the number of baselines. Each element is given by:
  $$ A_{k,j} = P(u_k, v_k) \exp\left[+2\pi i (u_k l_j + v_k m_j)\right] $$
  - $(u_k, v_k)$ are the baseline coordinates for the $k$-th measurement.
  - $(l_j, m_j)$ are the image-plane coordinates for the $j$-th pixel.
  - $P(u, v) = \text{sinc}^2(\pi u \Delta l) \cdot \text{sinc}^2(\pi v \Delta m)$ is the Fourier transform of the triangle-shaped pixel response function, where $\Delta l = \Delta m$ is the pixel size in radians.
- $\mathbf{n} \in \mathbb{C}^M$: Additive thermal noise, modeled as a complex Gaussian random variable with zero mean and variance $\sigma_k^2$ for each visibility.

**Optimization Problem:** We will solve a Regularized Maximum Likelihood (RML) problem. The general form of the objective function to be minimized is:
$$ \mathcal{L}(\mathbf{x}) = \mathcal{L}_{\text{data}}(\mathbf{x}) + \mathcal{L}_{\text{reg}}(\mathbf{x}) $$
$$ \mathbf{x}^* = \arg\min_{\mathbf{x} \ge 0, \sum \mathbf{x} = F_{\text{total}}} \mathcal{L}(\mathbf{x}) $$

We will implement three variants based on different data terms $\mathcal{L}_{\text{data}}$...

