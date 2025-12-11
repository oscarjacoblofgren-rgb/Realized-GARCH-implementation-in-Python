"""
Realized GARCH Model Implementation in JAX
Hansen, Huang & Shek (2012)

This is the full nonlinear state-space model, NOT a GARCH-X approximation.

Model Equations:
----------------
1. Return equation:
   r_t = μ + √h_t * z_t

2. Variance (state) equation:
   log(h_t) = ω + β*log(h_{t-1}) + γ*z_{t-1}^2

3. Measurement equation:
   log(RV_t) = ξ + φ*log(h_t) + τ*z_t + η_t

where:
   - h_t is the latent conditional variance
   - z_t ~ N(0, 1) is the standardized return shock
   - η_t ~ N(0, σ_η^2) is the measurement error shock
   - RV_t is the realized variance (observed)

Parameters:
-----------
θ = [μ, ω, β, γ, ξ, φ, τ, σ_η]

Constraints:
- β ∈ (0, 1) for stationarity
- γ ≥ 0
- σ_η > 0
- φ > 0 typically (but not strictly required)
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.scipy.stats import norm
import optax
from typing import Dict, Tuple, NamedTuple
from functools import partial


class RealizedGARCHParams(NamedTuple):
    """Realized GARCH parameters (unconstrained space for optimization)"""
    mu: float           # mean return
    omega: float        # variance intercept
    beta_raw: float     # transformed β (unconstrained)
    gamma_raw: float    # transformed γ (unconstrained)
    xi: float           # measurement intercept
    phi: float          # measurement loading on log(h_t)
    tau: float          # measurement loading on z_t
    log_sigma_eta: float # log of measurement error std


def transform_params(params_raw: RealizedGARCHParams) -> Dict[str, float]:
    """
    Transform unconstrained parameters to constrained space.

    Constraints enforced:
    - β ∈ (0, 1) via sigmoid
    - γ ≥ 0 via softplus
    - σ_η > 0 via exp

    Args:
        params_raw: Parameters in unconstrained space

    Returns:
        Dictionary of parameters in constrained space
    """
    beta = jax.nn.sigmoid(params_raw.beta_raw)  # maps to (0, 1)
    gamma = jax.nn.softplus(params_raw.gamma_raw)  # maps to (0, ∞)
    sigma_eta = jnp.exp(params_raw.log_sigma_eta)  # maps to (0, ∞)

    return {
        'mu': params_raw.mu,
        'omega': params_raw.omega,
        'beta': beta,
        'gamma': gamma,
        'xi': params_raw.xi,
        'phi': params_raw.phi,
        'tau': params_raw.tau,
        'sigma_eta': sigma_eta
    }


def compute_variance_path(params: Dict[str, float],
                         returns: jnp.ndarray,
                         h0: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recursively compute the latent variance path h_t and standardized shocks z_t.

    Uses jax.lax.scan for efficient, differentiable recursion.

    Equations:
    - z_t = (r_t - μ) / √h_t
    - log(h_t) = ω + β*log(h_{t-1}) + γ*z_{t-1}^2

    Args:
        params: Model parameters in constrained space
        returns: Return series (T,)
        h0: Initial variance

    Returns:
        h_t: Conditional variance path (T,)
        z_t: Standardized shocks (T,)
    """
    mu = params['mu']
    omega = params['omega']
    beta = params['beta']
    gamma = params['gamma']

    T = len(returns)

    def scan_fn(carry, t):
        """
        Scan function for variance recursion.

        carry: (log_h_prev, z_prev)
        t: time index
        """
        log_h_prev, z_prev = carry

        # Current variance from state equation
        log_h_t = omega + beta * log_h_prev + gamma * z_prev**2
        h_t = jnp.exp(log_h_t)

        # Current standardized shock
        r_t = returns[t]
        z_t = (r_t - mu) / jnp.sqrt(h_t)

        return (log_h_t, z_t), (h_t, z_t)

    # Initial values
    log_h0 = jnp.log(h0)
    z0 = 0.0  # Initial shock (not used for t=0)

    # Run scan
    _, (h_path, z_path) = jax.lax.scan(scan_fn, (log_h0, z0), jnp.arange(T))

    return h_path, z_path


def compute_measurement_residuals(params: Dict[str, float],
                                  log_rv: jnp.ndarray,
                                  h_t: jnp.ndarray,
                                  z_t: jnp.ndarray) -> jnp.ndarray:
    """
    Compute measurement equation residuals.

    Equation:
    m_t = log(RV_t) - (ξ + φ*log(h_t) + τ*z_t)

    where m_t = η_t ~ N(0, σ_η^2)

    Args:
        params: Model parameters
        log_rv: Log realized variance (T,)
        h_t: Conditional variance path (T,)
        z_t: Standardized shocks (T,)

    Returns:
        Measurement residuals (T,)
    """
    xi = params['xi']
    phi = params['phi']
    tau = params['tau']

    log_h_t = jnp.log(h_t)
    residuals = log_rv - (xi + phi * log_h_t + tau * z_t)

    return residuals


@jit
def realized_garch_loglik(params_raw: RealizedGARCHParams,
                         returns: jnp.ndarray,
                         log_rv: jnp.ndarray,
                         h0: float = None) -> float:
    """
    Compute the joint log-likelihood of returns and log(RV).

    Joint density:
    p(r_t, log(RV_t) | h_{t-1}, z_{t-1}; θ) = p(r_t | h_t; θ) * p(log(RV_t) | r_t, h_t; θ)

    where:
    - r_t | h_t ~ N(μ, h_t), equivalently z_t ~ N(0, 1)
    - η_t = log(RV_t) - (ξ + φ*log(h_t) + τ*z_t) ~ N(0, σ_η^2)

    Args:
        params_raw: Parameters in unconstrained space
        returns: Return series (T,)
        log_rv: Log realized variance (T,)
        h0: Initial variance (if None, use sample variance)

    Returns:
        Negative log-likelihood (for minimization)
    """
    # Transform parameters
    params = transform_params(params_raw)

    # Initial variance
    if h0 is None:
        h0 = jnp.var(returns)

    # Compute variance path and standardized shocks
    h_t, z_t = compute_variance_path(params, returns, h0)

    # Compute measurement residuals
    eta_t = compute_measurement_residuals(params, log_rv, h_t, z_t)

    # Log-likelihood components
    sigma_eta = params['sigma_eta']

    # 1. Return shock contribution: z_t ~ N(0, 1)
    loglik_returns = jnp.sum(norm.logpdf(z_t))

    # 2. Measurement shock contribution: η_t ~ N(0, σ_η^2)
    loglik_measurement = jnp.sum(norm.logpdf(eta_t, loc=0.0, scale=sigma_eta))

    # Total log-likelihood
    total_loglik = loglik_returns + loglik_measurement

    # Return negative for minimization
    return -total_loglik


@jit
def realized_garch_loglik_with_grad(params_raw: RealizedGARCHParams,
                                   returns: jnp.ndarray,
                                   log_rv: jnp.ndarray,
                                   h0: float = None) -> Tuple[float, RealizedGARCHParams]:
    """
    Compute log-likelihood and its gradient simultaneously.

    This is more efficient than computing them separately.

    Returns:
        neg_loglik: Negative log-likelihood
        grad: Gradient with respect to params_raw
    """
    return value_and_grad(realized_garch_loglik)(params_raw, returns, log_rv, h0)


def initialize_params(returns: jnp.ndarray,
                     log_rv: jnp.ndarray,
                     seed: int = 42) -> RealizedGARCHParams:
    """
    Initialize parameters with reasonable starting values.

    Strategy:
    - μ: sample mean of returns
    - ω: scaled to give E[log(h)] ≈ mean(log(RV))
    - β: 0.8 (typical persistence)
    - γ: 0.1 (modest leverage effect)
    - ξ: adjusted intercept for measurement equation
    - φ: 1.0 (typical loading)
    - τ: 0.0 (often close to zero)
    - σ_η: sample std of log(RV) residuals

    Args:
        returns: Return series
        log_rv: Log realized variance
        seed: Random seed for any stochastic initialization

    Returns:
        Initial parameters in unconstrained space
    """
    # Basic statistics
    mu_init = jnp.mean(returns)
    mean_log_rv = jnp.mean(log_rv)
    std_log_rv = jnp.std(log_rv)

    # Variance equation parameters
    beta_target = 0.8
    gamma_target = 0.1

    # Compute omega to match unconditional mean of log(h)
    # E[log(h)] = ω / (1 - β) if γ = 0 approximately
    # Set to match mean of log(RV) / φ
    phi_init = 1.0
    omega_init = mean_log_rv / phi_init * (1 - beta_target) * 0.5

    # Measurement equation parameters
    xi_init = mean_log_rv - phi_init * mean_log_rv  # Residual
    tau_init = 0.0

    # Measurement error (set to fraction of log(RV) variance)
    sigma_eta_init = std_log_rv * 0.3

    # Transform to unconstrained space
    # β: sigmoid^{-1}(β) = log(β / (1 - β))
    beta_raw = jnp.log(beta_target / (1 - beta_target))

    # γ: softplus^{-1}(γ) ≈ log(exp(γ) - 1) for γ > 0
    # For small γ, softplus^{-1}(γ) ≈ γ
    gamma_raw = jnp.log(jnp.exp(gamma_target) - 1)

    # σ_η: log transform
    log_sigma_eta = jnp.log(sigma_eta_init)

    return RealizedGARCHParams(
        mu=mu_init,
        omega=omega_init,
        beta_raw=beta_raw,
        gamma_raw=gamma_raw,
        xi=xi_init,
        phi=phi_init,
        tau=tau_init,
        log_sigma_eta=log_sigma_eta
    )


def fit_realized_garch(returns: jnp.ndarray,
                      log_rv: jnp.ndarray,
                      params_init: RealizedGARCHParams = None,
                      learning_rate: float = 0.01,
                      num_iterations: int = 1000,
                      verbose: bool = True) -> Tuple[RealizedGARCHParams, Dict]:
    """
    Fit Realized GARCH model using gradient-based optimization (Adam).

    Args:
        returns: Return series (T,)
        log_rv: Log realized variance (T,)
        params_init: Initial parameters (if None, auto-initialize)
        learning_rate: Learning rate for Adam optimizer
        num_iterations: Number of optimization iterations
        verbose: Print progress

    Returns:
        params_opt: Optimized parameters (unconstrained space)
        info: Dictionary with optimization information
    """
    # Initialize parameters
    if params_init is None:
        params_init = initialize_params(returns, log_rv)

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_init)

    # Optimization loop
    params = params_init
    h0 = jnp.var(returns)

    losses = []

    for i in range(num_iterations):
        # Compute loss and gradient
        loss, grads = realized_garch_loglik_with_grad(params, returns, log_rv, h0)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        losses.append(float(loss))

        # Print progress
        if verbose and (i % 100 == 0 or i == num_iterations - 1):
            params_constrained = transform_params(params)
            print(f"Iteration {i:4d}: Loss = {loss:.4f}, "
                  f"β = {params_constrained['beta']:.4f}, "
                  f"γ = {params_constrained['gamma']:.4f}")

    info = {
        'losses': losses,
        'num_iterations': num_iterations,
        'final_loss': losses[-1]
    }

    return params, info


def realized_garch_forecast(params_raw: RealizedGARCHParams,
                           returns: jnp.ndarray,
                           log_rv: jnp.ndarray,
                           horizon: int = 1,
                           h0: float = None) -> Dict[str, jnp.ndarray]:
    """
    Forecast future variance and realized variance.

    Multi-step ahead forecast:
    - E[h_{T+k}] for k = 1, ..., horizon
    - E[log(RV_{T+k})] for k = 1, ..., horizon

    Recursive forecast equations:
    - E[z_t^2] = 1 for future shocks
    - E[log(h_{T+k})] = ω + β*E[log(h_{T+k-1})] + γ
    - E[log(RV_{T+k})] = ξ + φ*E[log(h_{T+k})]

    Args:
        params_raw: Model parameters (unconstrained)
        returns: Historical returns (T,)
        log_rv: Historical log realized variance (T,)
        horizon: Forecast horizon
        h0: Initial variance

    Returns:
        Dictionary containing:
        - 'h_forecast': E[h_{T+1}], ..., E[h_{T+horizon}]
        - 'log_rv_forecast': E[log(RV_{T+1})], ..., E[log(RV_{T+horizon})]
    """
    # Transform parameters
    params = transform_params(params_raw)

    mu = params['mu']
    omega = params['omega']
    beta = params['beta']
    gamma = params['gamma']
    xi = params['xi']
    phi = params['phi']

    # Get current variance
    if h0 is None:
        h0 = jnp.var(returns)

    h_t, z_t = compute_variance_path(params, returns, h0)
    log_h_T = jnp.log(h_t[-1])

    # Multi-step forecast
    h_forecasts = []
    log_rv_forecasts = []

    log_h_current = log_h_T

    for k in range(horizon):
        # Forecast log(h_{T+k+1})
        # E[z_T^2] = 1 for future shocks
        log_h_next = omega + beta * log_h_current + gamma * 1.0
        h_next = jnp.exp(log_h_next)

        # Forecast log(RV_{T+k+1})
        # E[τ*z_{T+k+1}] = 0 since E[z] = 0
        log_rv_next = xi + phi * log_h_next

        h_forecasts.append(h_next)
        log_rv_forecasts.append(log_rv_next)

        log_h_current = log_h_next

    return {
        'h_forecast': jnp.array(h_forecasts),
        'log_rv_forecast': jnp.array(log_rv_forecasts),
        'rv_forecast': jnp.exp(jnp.array(log_rv_forecasts))
    }


def simulate_realized_garch(params: Dict[str, float],
                           T: int,
                           h0: float = None,
                           seed: int = 0) -> Dict[str, jnp.ndarray]:
    """
    Simulate data from the Realized GARCH model.

    Simulation steps:
    1. Initialize h_0
    2. For t = 1, ..., T:
       a. Draw z_t ~ N(0, 1)
       b. Compute r_t = μ + √h_t * z_t
       c. Update log(h_{t+1}) = ω + β*log(h_t) + γ*z_t^2
       d. Draw η_t ~ N(0, σ_η^2)
       e. Compute log(RV_t) = ξ + φ*log(h_t) + τ*z_t + η_t

    Args:
        params: Model parameters (constrained space)
        T: Sample size
        h0: Initial variance
        seed: Random seed

    Returns:
        Dictionary containing:
        - 'returns': Simulated returns (T,)
        - 'log_rv': Simulated log realized variance (T,)
        - 'rv': Simulated realized variance (T,)
        - 'h': True latent variance path (T,)
        - 'z': True standardized shocks (T,)
    """
    # Set random seed
    key = jax.random.PRNGKey(seed)
    key_z, key_eta = jax.random.split(key)

    # Extract parameters
    mu = params['mu']
    omega = params['omega']
    beta = params['beta']
    gamma = params['gamma']
    xi = params['xi']
    phi = params['phi']
    tau = params['tau']
    sigma_eta = params['sigma_eta']

    # Initial variance
    if h0 is None:
        # Unconditional variance (approximately)
        h0 = jnp.exp(omega / (1 - beta))

    # Draw shocks
    z_shocks = jax.random.normal(key_z, shape=(T,))
    eta_shocks = jax.random.normal(key_eta, shape=(T,)) * sigma_eta

    # Simulate recursively
    def scan_fn(carry, t):
        log_h_prev, z_prev = carry

        # Current variance
        log_h_t = omega + beta * log_h_prev + gamma * z_prev**2
        h_t = jnp.exp(log_h_t)

        # Current shock and return
        z_t = z_shocks[t]
        r_t = mu + jnp.sqrt(h_t) * z_t

        # Realized variance (measurement)
        log_rv_t = xi + phi * log_h_t + tau * z_t + eta_shocks[t]
        rv_t = jnp.exp(log_rv_t)

        return (log_h_t, z_t), (r_t, log_rv_t, rv_t, h_t, z_t)

    # Initial state
    log_h0 = jnp.log(h0)
    z0 = 0.0

    # Run simulation
    _, (returns, log_rv, rv, h_path, z_path) = jax.lax.scan(
        scan_fn, (log_h0, z0), jnp.arange(T)
    )

    return {
        'returns': returns,
        'log_rv': log_rv,
        'rv': rv,
        'h': h_path,
        'z': z_path
    }
