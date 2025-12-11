"""
Complete Working Example: Realized GARCH Model on Synthetic Data

This script demonstrates:
1. Simulating data from the Realized GARCH model
2. Fitting the model to recover parameters
3. Comparing estimated vs. true parameters
4. Forecasting future variance and realized variance
5. Diagnostic plots (optional)
"""

import jax
import jax.numpy as jnp
import numpy as np
from realized_garch_jax import (
    RealizedGARCHParams,
    simulate_realized_garch,
    fit_realized_garch,
    realized_garch_forecast,
    transform_params,
    initialize_params
)

# Set JAX to use CPU (avoid GPU memory issues on small examples)
jax.config.update('jax_platform_name', 'cpu')


def main():
    print("=" * 80)
    print("Realized GARCH Model - Synthetic Data Example")
    print("Hansen, Huang & Shek (2012)")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Define True Parameters
    # =========================================================================
    print("STEP 1: Setting True Parameters")
    print("-" * 80)

    true_params = {
        'mu': 0.05,           # mean return (annualized ~5%)
        'omega': -0.3,        # variance intercept
        'beta': 0.85,         # variance persistence
        'gamma': 0.15,        # leverage effect
        'xi': 0.1,            # measurement intercept
        'phi': 0.9,           # measurement loading on log(h_t)
        'tau': -0.05,         # measurement loading on z_t
        'sigma_eta': 0.3      # measurement error std
    }

    print("True parameters:")
    for key, value in true_params.items():
        print(f"  {key:12s} = {value:7.4f}")
    print()

    # =========================================================================
    # STEP 2: Simulate Data
    # =========================================================================
    print("STEP 2: Simulating Data")
    print("-" * 80)

    T = 1000  # Sample size
    h0 = 1.0  # Initial variance
    seed = 12345

    print(f"Sample size: T = {T}")
    print(f"Initial variance: h_0 = {h0}")
    print(f"Random seed: {seed}")
    print()

    # Simulate
    sim_data = simulate_realized_garch(
        params=true_params,
        T=T,
        h0=h0,
        seed=seed
    )

    returns = sim_data['returns']
    log_rv = sim_data['log_rv']
    rv = sim_data['rv']
    h_true = sim_data['h']

    print("Simulated data statistics:")
    print(f"  Returns:    mean = {np.mean(returns):7.4f}, std = {np.std(returns):7.4f}")
    print(f"  log(RV):    mean = {np.mean(log_rv):7.4f}, std = {np.std(log_rv):7.4f}")
    print(f"  Variance h: mean = {np.mean(h_true):7.4f}, std = {np.std(h_true):7.4f}")
    print()

    # =========================================================================
    # STEP 3: Initialize Parameters
    # =========================================================================
    print("STEP 3: Initializing Parameters for Estimation")
    print("-" * 80)

    params_init = initialize_params(returns, log_rv, seed=42)
    params_init_constrained = transform_params(params_init)

    print("Initial parameter guesses (constrained space):")
    for key, value in params_init_constrained.items():
        print(f"  {key:12s} = {value:7.4f}")
    print()

    # =========================================================================
    # STEP 4: Fit Model
    # =========================================================================
    print("STEP 4: Fitting Model")
    print("-" * 80)
    print()

    params_opt, info = fit_realized_garch(
        returns=returns,
        log_rv=log_rv,
        params_init=params_init,
        learning_rate=0.01,
        num_iterations=1000,
        verbose=True
    )

    print()
    print(f"Optimization completed in {info['num_iterations']} iterations")
    print(f"Final loss: {info['final_loss']:.4f}")
    print()

    # =========================================================================
    # STEP 5: Compare Estimated vs. True Parameters
    # =========================================================================
    print("STEP 5: Parameter Recovery")
    print("-" * 80)

    params_estimated = transform_params(params_opt)

    print(f"{'Parameter':<12s} {'True':>10s} {'Estimated':>10s} {'Error':>10s} {'% Error':>10s}")
    print("-" * 80)

    for key in true_params.keys():
        true_val = true_params[key]
        est_val = params_estimated[key]
        error = est_val - true_val
        pct_error = (error / true_val * 100) if true_val != 0 else np.nan

        print(f"{key:<12s} {true_val:10.4f} {est_val:10.4f} {error:10.4f} {pct_error:9.2f}%")

    print()

    # =========================================================================
    # STEP 6: Forecasting
    # =========================================================================
    print("STEP 6: Forecasting")
    print("-" * 80)

    forecast_horizon = 10

    forecasts = realized_garch_forecast(
        params_raw=params_opt,
        returns=returns,
        log_rv=log_rv,
        horizon=forecast_horizon,
        h0=h0
    )

    h_forecast = forecasts['h_forecast']
    rv_forecast = forecasts['rv_forecast']

    print(f"Forecasting {forecast_horizon} steps ahead:")
    print()
    print(f"{'Step':>6s} {'E[h_t]':>12s} {'E[RV_t]':>12s}")
    print("-" * 35)

    for k in range(forecast_horizon):
        print(f"{k+1:6d} {h_forecast[k]:12.4f} {rv_forecast[k]:12.4f}")

    print()

    # =========================================================================
    # STEP 7: Model Diagnostics
    # =========================================================================
    print("STEP 7: Model Diagnostics")
    print("-" * 80)

    # Compute fitted values
    from realized_garch_jax import compute_variance_path, compute_measurement_residuals

    h_fitted, z_fitted = compute_variance_path(params_estimated, returns, h0)
    eta_fitted = compute_measurement_residuals(params_estimated, log_rv, h_fitted, z_fitted)

    # Standardized residuals should be N(0,1)
    print("Standardized return shocks (z_t) - should be N(0,1):")
    print(f"  Mean:     {np.mean(z_fitted):7.4f} (target: 0.0000)")
    print(f"  Std Dev:  {np.std(z_fitted):7.4f} (target: 1.0000)")
    print(f"  Skewness: {float(jnp.mean((z_fitted - jnp.mean(z_fitted))**3) / jnp.std(z_fitted)**3):7.4f} (target: 0.0000)")
    print(f"  Kurtosis: {float(jnp.mean((z_fitted - jnp.mean(z_fitted))**4) / jnp.std(z_fitted)**4):7.4f} (target: 3.0000)")
    print()

    # Measurement residuals should be N(0, σ_η^2)
    print(f"Measurement residuals (η_t) - should be N(0, {params_estimated['sigma_eta']**2:.4f}):")
    print(f"  Mean:     {np.mean(eta_fitted):7.4f} (target: 0.0000)")
    print(f"  Std Dev:  {np.std(eta_fitted):7.4f} (target: {params_estimated['sigma_eta']:7.4f})")
    print()

    # Correlation between returns and RV
    corr_r_rv = np.corrcoef(returns, rv)[0, 1]
    print(f"Correlation between returns and RV: {corr_r_rv:7.4f}")
    print()

    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
