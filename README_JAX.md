# Realized GARCH Model in JAX

**Full implementation of the Realized GARCH model from Hansen, Huang & Shek (2012)**

This is the complete nonlinear state-space formulation, **not** a GARCH-X approximation.

---

## Mathematical Model

The Realized GARCH model jointly models returns and realized variance through a latent conditional variance process:

### 1. Return Equation
$$r_t = \mu + \sqrt{h_t} \, z_t$$

where $z_t \sim \mathcal{N}(0, 1)$ is the standardized return shock.

### 2. Variance (State) Equation
$$\log h_t = \omega + \beta \log h_{t-1} + \gamma z_{t-1}^2$$

This is a log-linear GARCH specification for the latent conditional variance $h_t$.

### 3. Measurement Equation
$$\log \text{RV}_t = \xi + \phi \log h_t + \tau z_t + \eta_t$$

where $\eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$ is the measurement error.

The realized variance $\text{RV}_t$ (computed from high-frequency data) is a noisy proxy for the latent $h_t$.

---

## Model Parameters

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| $\mu$ | Mean return | Unrestricted |
| $\omega$ | Variance intercept | Unrestricted |
| $\beta$ | Variance persistence | $(0, 1)$ |
| $\gamma$ | Leverage effect | $\geq 0$ |
| $\xi$ | Measurement intercept | Unrestricted |
| $\phi$ | Measurement loading on $\log h_t$ | Typically $> 0$ |
| $\tau$ | Measurement loading on $z_t$ | Unrestricted |
| $\sigma_\eta$ | Measurement error std | $> 0$ |

---

## Likelihood Function

The joint log-likelihood is:

$$\ell(\theta) = \sum_{t=1}^T \left[ \log p(r_t \mid h_t) + \log p(\log \text{RV}_t \mid r_t, h_t) \right]$$

where:

1. **Return contribution**: $z_t = \frac{r_t - \mu}{\sqrt{h_t}} \sim \mathcal{N}(0, 1)$

   $$\log p(r_t \mid h_t) = -\frac{1}{2}\log(2\pi) - \frac{1}{2} z_t^2$$

2. **Measurement contribution**: $\eta_t = \log \text{RV}_t - (\xi + \phi \log h_t + \tau z_t) \sim \mathcal{N}(0, \sigma_\eta^2)$

   $$\log p(\log \text{RV}_t \mid r_t, h_t) = -\frac{1}{2}\log(2\pi\sigma_\eta^2) - \frac{1}{2\sigma_\eta^2} \eta_t^2$$

---

## Implementation Features

### Pure JAX Implementation
- **`jax.numpy`**: All array operations
- **`jax.lax.scan`**: Efficient recursive variance computation
- **`jax.jit`**: JIT-compiled likelihood function
- **`jax.grad`**: Automatic differentiation for gradients

### Parameter Constraints
Enforced via smooth reparameterizations:
- $\beta = \text{sigmoid}(\beta_{\text{raw}})$ maps to $(0, 1)$
- $\gamma = \text{softplus}(\gamma_{\text{raw}})$ maps to $(0, \infty)$
- $\sigma_\eta = \exp(\log \sigma_\eta)$ maps to $(0, \infty)$

### Optimization
- **Adam optimizer** (via `optax`) - default, robust
- **L-BFGS optimizer** (via `jaxopt`) - faster convergence for smooth problems

### Forecasting
Multi-step ahead forecasts using:
$$\mathbb{E}[\log h_{T+k}] = \omega + \beta \mathbb{E}[\log h_{T+k-1}] + \gamma$$

since $\mathbb{E}[z_t^2] = 1$ for future shocks.

---

## Installation

```bash
pip install -r requirements.txt
```

### Core Requirements
- `jax >= 0.4.0`
- `jaxlib >= 0.4.0`
- `optax >= 0.1.7`
- `numpy >= 1.24.0`

### Optional
- `jaxopt >= 0.8.0` (for L-BFGS optimization)
- `matplotlib >= 3.7.0` (for plotting)
- `pandas >= 2.0.0` (for data handling)

---

## Quick Start

### Example 1: Synthetic Data (Adam Optimizer)

```python
from realized_garch_jax import (
    simulate_realized_garch,
    fit_realized_garch,
    realized_garch_forecast,
    transform_params
)

# 1. Define true parameters
true_params = {
    'mu': 0.05,
    'omega': -0.3,
    'beta': 0.85,
    'gamma': 0.15,
    'xi': 0.1,
    'phi': 0.9,
    'tau': -0.05,
    'sigma_eta': 0.3
}

# 2. Simulate data
sim_data = simulate_realized_garch(
    params=true_params,
    T=1000,
    h0=1.0,
    seed=42
)

returns = sim_data['returns']
log_rv = sim_data['log_rv']

# 3. Fit model
params_opt, info = fit_realized_garch(
    returns=returns,
    log_rv=log_rv,
    learning_rate=0.01,
    num_iterations=1000,
    verbose=True
)

# 4. Get estimated parameters
params_estimated = transform_params(params_opt)
print(params_estimated)

# 5. Forecast
forecasts = realized_garch_forecast(
    params_raw=params_opt,
    returns=returns,
    log_rv=log_rv,
    horizon=10
)

print("Variance forecasts:", forecasts['h_forecast'])
print("RV forecasts:", forecasts['rv_forecast'])
```

### Example 2: L-BFGS Optimization

```python
from example_lbfgs_optimization import fit_realized_garch_lbfgs

# Fit using L-BFGS (faster convergence)
params_opt, info = fit_realized_garch_lbfgs(
    returns=returns,
    log_rv=log_rv,
    maxiter=500,
    verbose=True
)
```

---

## Running Examples

### Complete synthetic data example:
```bash
python example_synthetic_data.py
```

This will:
1. Simulate data from known parameters
2. Fit the model to recover parameters
3. Compare estimated vs. true values
4. Generate forecasts
5. Compute diagnostic statistics

### L-BFGS optimization example:
```bash
python example_lbfgs_optimization.py
```

This demonstrates faster quasi-Newton optimization.

---

## File Structure

```
.
├── realized_garch_jax.py           # Core model implementation
├── example_synthetic_data.py       # Complete working example (Adam)
├── example_lbfgs_optimization.py   # L-BFGS optimization example
├── requirements.txt                # Python dependencies
└── README_JAX.md                   # This file
```

---

## API Reference

### Core Functions

#### `simulate_realized_garch(params, T, h0=None, seed=0)`
Simulate data from the Realized GARCH model.

**Args:**
- `params`: Dictionary of model parameters (constrained space)
- `T`: Sample size
- `h0`: Initial variance (default: unconditional variance)
- `seed`: Random seed

**Returns:**
- Dictionary with keys: `'returns'`, `'log_rv'`, `'rv'`, `'h'`, `'z'`

---

#### `fit_realized_garch(returns, log_rv, params_init=None, learning_rate=0.01, num_iterations=1000, verbose=True)`
Fit the model using Adam optimizer.

**Args:**
- `returns`: Return series (T,)
- `log_rv`: Log realized variance (T,)
- `params_init`: Initial parameters (default: auto-initialize)
- `learning_rate`: Adam learning rate
- `num_iterations`: Number of iterations
- `verbose`: Print progress

**Returns:**
- `params_opt`: Optimized parameters (unconstrained space)
- `info`: Dictionary with optimization information

---

#### `realized_garch_forecast(params_raw, returns, log_rv, horizon=1, h0=None)`
Generate multi-step forecasts.

**Args:**
- `params_raw`: Model parameters (unconstrained space)
- `returns`: Historical returns
- `log_rv`: Historical log realized variance
- `horizon`: Forecast horizon
- `h0`: Initial variance

**Returns:**
- Dictionary with keys: `'h_forecast'`, `'log_rv_forecast'`, `'rv_forecast'`

---

#### `transform_params(params_raw)`
Transform parameters from unconstrained to constrained space.

**Args:**
- `params_raw`: `RealizedGARCHParams` object (unconstrained)

**Returns:**
- Dictionary of parameters (constrained space)

---

#### `realized_garch_loglik(params_raw, returns, log_rv, h0=None)`
Compute the negative log-likelihood (JIT-compiled).

**Args:**
- `params_raw`: Parameters (unconstrained space)
- `returns`: Return series
- `log_rv`: Log realized variance
- `h0`: Initial variance

**Returns:**
- Negative log-likelihood value (scalar)

---

## Model Properties

### Stationarity
The variance process is covariance-stationary if:
$$\beta + \gamma \mathbb{E}[z_{t-1}^2] < 1$$

Since $\mathbb{E}[z_{t-1}^2] = 1$, this requires $\beta + \gamma < 1$.

### Unconditional Variance
When $\gamma = 0$, the unconditional log-variance is:
$$\mathbb{E}[\log h_t] = \frac{\omega}{1 - \beta}$$

With $\gamma > 0$, the unconditional mean is:
$$\mathbb{E}[\log h_t] = \frac{\omega + \gamma}{1 - \beta}$$

### Leverage Effect
The parameter $\gamma$ captures the leverage effect: positive shocks ($z_{t-1}^2$ large) increase future variance.

### Measurement Quality
The ratio $\phi$ determines how informative realized variance is about conditional variance:
- $\phi \approx 1$: $\text{RV}_t$ is a precise proxy for $h_t$
- $\phi < 1$: Realized variance underestimates conditional variance
- Large $\sigma_\eta$: Noisy measurement

---

## Technical Details

### Numerical Stability
- Log-space variance prevents numerical overflow/underflow
- Smooth parameter transformations ensure constraints without barriers
- JAX's automatic differentiation provides numerically stable gradients

### Computational Efficiency
- `jax.jit` compiles the likelihood function to XLA
- `jax.lax.scan` efficiently handles recursive loops
- All operations are vectorized where possible
- Typical speed: ~10ms per likelihood evaluation on CPU for T=1000

### Differentiation
The entire likelihood is automatically differentiable:
```python
from jax import grad

# Gradient function
grad_fn = grad(realized_garch_loglik)

# Compute gradient
grads = grad_fn(params_raw, returns, log_rv)
```

---

## Comparison with GARCH-X

The Realized GARCH model is **not** equivalent to GARCH-X:

| Feature | Realized GARCH | GARCH-X |
|---------|----------------|---------|
| Latent variance | Yes ($h_t$) | Yes ($h_t$) |
| Measurement equation | Yes (nonlinear) | No |
| State-space form | Nonlinear | N/A |
| Uses RV in variance | Via measurement | Directly in $h_t$ |
| Likelihood | Joint (returns + RV) | Returns only |

GARCH-X uses realized variance directly in the variance equation:
$$h_t = \omega + \beta h_{t-1} + \gamma \text{RV}_{t-1}$$

Realized GARCH treats RV as a noisy measurement of the latent $h_t$.

---

## References

Hansen, P. R., Huang, Z., & Shek, H. H. (2012). **Realized GARCH: A joint model for returns and realized measures of volatility.** *Journal of Applied Econometrics*, 27(6), 877-906.

---

## Troubleshooting

### Common Issues

**1. Optimization not converging**
- Try L-BFGS instead of Adam
- Increase number of iterations
- Check data quality (NaNs, infinities)
- Try different initial parameters

**2. Numerical overflow**
- Ensure data is properly scaled
- Check for extreme outliers in returns or RV
- Verify log(RV) is computed correctly (RV must be positive)

**3. JAX/GPU issues**
- Set `jax.config.update('jax_platform_name', 'cpu')` to force CPU
- Ensure JAX and jaxlib versions match
- Check CUDA installation if using GPU

**4. Slow optimization**
- Use `jit=True` (enabled by default)
- Try L-BFGS for faster convergence
- Reduce sample size for testing
- Use smaller horizon for forecasts

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{hansen2012realized,
  title={Realized GARCH: A joint model for returns and realized measures of volatility},
  author={Hansen, Peter Reinhard and Huang, Zhuo and Shek, Howard Howan},
  journal={Journal of Applied Econometrics},
  volume={27},
  number={6},
  pages={877--906},
  year={2012},
  publisher={Wiley Online Library}
}
```
