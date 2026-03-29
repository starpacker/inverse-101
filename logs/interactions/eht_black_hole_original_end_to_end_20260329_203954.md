# Multi-Agent Pipeline Log


## Iteration 1 — Ticket: Planner

### Planner

**Plan:**
### [Problem Formulation]

The goal is to reconstruct a non-negative image **x** from a set of interferometric measurements. The problem is formulated as a Regularized Maximum Likelihood (RML) optimization problem. We seek the image **x** that minimizes a composite objective function `J(x)`:

$$ \mathbf{x}^* = \arg\min_{\mathbf{x} \ge 0, \sum \mathbf{x} = F_{\text{total}}} J(\mathbf{x}) $$

where `J(x)` is the sum of data fidelity terms (chi-squared, $\chi^2$) and regularization terms `R(x)`:

$$ J(\mathbf{x}) = J_{\text{data}}(\mathbf{x}) + J_{\text{reg}}(\mathbf{x}) $$

The image **x** is a vectorized representation of a 64x64 pixel grid, so **x** $\in \mathbb{R}^{4096}$. The constraints are non-negativity (**x** $\ge 0$) and conservation of total flux ($\sum \mathbf{x} = F_{\text{total}}$).

**Forward Model:**
The underlying physical model connecting the image **x** to the ideal complex visibilities **y** is a Non-Uniform Discrete Fourier Transform (NUDFT):

$$ \mathbf{y} = \mathbf{A}(\mathbf{x}) $$

where **A** is the forward operator. For a given visibility `m` at baseline coordinates `(u_m, v_m)` and a pixel `n` at sky coordinates `(l_n, m_n)`:

$$ A_{m,n}(\mathbf{x}) = P(u_m, v_m) \sum_{n=1}^{N^2} x_n \exp\left[+2\pi i (u_m l_n + v_m m_n)\right] $$

- **x**: Vectorized sky brightness image, shape `(4096,)`.
- **y**: Model complex visibilities, shape `(421,)`.
- `(u_m, v_m)`: Baseline coordinates for the m-th measurement.
- `(l_n, m_n)`: Sky coordinates for the n-th pixel.
- `P(u, v)`: Fourier transform of the pixel shape, modeled as a separable triangle pulse: $P(u,v) = \text{sinc}^2(u \cdot \Delta l) \cdot \text{sinc}^2(v \cdot \Delta m)$, where $\Delta l, \Delta m$ are the pixel sizes in radians.

**Data and Regularization Terms:**
We will implement three different data terms, corresponding to three different imaging methods, to demonstrate the effect of gain corruption.

1.  **Visibility RML (`vis`)**: Fits complex visibilities directly.
    $$ J_{\text{da...

