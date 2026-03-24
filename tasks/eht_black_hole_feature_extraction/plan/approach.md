# Approach: α-DPI for Black Hole Feature Extraction

## Algorithm Overview

α-DPI is a two-step probabilistic inference method that infers posterior distributions over geometric parameters of black hole images from interferometric closure quantities.

### Step 1: α-Divergence Variational Inference

Train a normalizing flow q_θ(x) to approximate the posterior p(x|y):

1. Sample z ~ N(0, I) from the base distribution
2. Transform via flow: x_unconstrained = flow.reverse(z), x = sigmoid(x_unconstrained)
3. Generate image: img = GeometricModel(x)
4. Compute closure quantities via NUFFT forward model
5. Compute data fidelity loss (closure phase + log closure amplitude chi-squared)
6. Compute log-probability: log q(x) = -logdet_flow - logdet_sigmoid - 0.5*||z||²
7. Apply α-divergence reweighting: w_n ∝ softmax(-(1-α)*loss_n)
8. Update: θ ← θ - lr * ∇_θ Σ w_n * loss_n

Data warmup gradually increases the data weight from 10^-4 to 1.0 over 2000 epochs.

### Step 2: Importance Sampling

After training, generate N samples and reweight by importance weights:
w(x) = p(y|x) * p(x) / q_θ(x) ∝ softmax(-loss(x))

The weighted samples provide a more accurate approximation of the true posterior.

### Model Selection via ELBO

Compare models with different numbers of Gaussian components (0, 1, 2, 3) using:
ELBO = E_q[log p(y|x)] - KL(q||p) ≈ -mean(loss_data + log q)

The model with the highest ELBO best balances fit quality and complexity.

## Key Design Choices

- **Geometric model in parameter space** (not pixel space): 4-16 parameters vs 4096 pixels
- **Sigmoid for bounded parameters**: Maps (-∞, ∞) flow output to [0, 1]
- **α-divergence (α < 1)**: More mass-covering than KL, avoids mode collapse
- **Small seqfrac (1/16)**: Large hidden layers relative to input dimension since parameter space is low-dimensional
- **Large batch size (2048)**: Needed for stable α-divergence gradient estimation
