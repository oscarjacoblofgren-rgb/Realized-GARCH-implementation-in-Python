"""
Alternative Optimization Example: Using L-BFGS via jaxopt

L-BFGS is often more efficient than Adam for smooth optimization problems
like maximum likelihood estimation. This example shows how to use jaxopt's
L-BFGS optimizer with the Realized GARCH model.
"""

import jax
import jax.numpy as jnp
import numpy as np

try:
    import jaxopt
    JAXOPT_AVAILABLE = True
except ImportError:
    JAXOPT_AVAILABLE = False
    print("Warning: jaxopt not installed. Install with: pip install jaxopt")

from realized_garch_jax import (
    RealizedGARCHParams,
    simulate_realized_garch,
    realized_garch_loglik,
    transform_params,
    initialize_params
)


def fit_realized_garch_lbfgs(returns: jnp.ndarray,
                             log_rv: jnp.ndarray,
                             params_init: RealizedGARCHParams = None,
                             maxiter: int = 500,
                             verbose: bool = True):
    """
    Fit Realized GARCH model using L-BFGS optimization.

    L-BFGS is a quasi-Newton method that often converges faster than
    first-order methods like Adam, especially for smooth problems.

    Args:
        returns: Return series (T,)
        log_rv: Log realized variance (T,)
        params_init: Initial parameters (if None, auto-initialize)
        maxiter: Maximum number of iterations
        verbose: Print progress

    Returns:
        params_opt: Optimized parameters
        info: Optimization information
    """
    if not JAXOPT_AVAILABLE:
        raise ImportError("jaxopt is required for L-BFGS optimization")

    # Initialize parameters
    if params_init is None:
        params_init = initialize_params(returns, log_rv)

    h0 = jnp.var(returns)

    # Define objective function (takes flattened parameters)
    def objective(params_raw):
        return realized_garch_loglik(params_raw, returns, log_rv, h0)

    # Setup L-BFGS solver
    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=maxiter,
        tol=1e-6,
        verbose=verbose
    )

    # Run optimization
    result = solver.run(params_init)

    params_opt = result.params
    info = {
        'final_loss': float(result.state.value),
        'num_iterations': int(result.state.iter_num),
        'converged': result.state.error < 1e-6,
        'error': float(result.state.error)
    }

    return params_opt, info


def main():
    if not JAXOPT_AVAILABLE:
        print("This example requires jaxopt. Install with: pip install jaxopt")
        return

    print("=" * 80)
    print("Realized GARCH Model - L-BFGS Optimization Example")
    print("=" * 80)
    print()

    # True parameters
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

    print("True parameters:")
    for key, value in true_params.items():
        print(f"  {key:12s} = {value:7.4f}")
    print()

    # Simulate data
    T = 1000
    h0 = 1.0
    seed = 12345

    print(f"Simulating {T} observations...")
    sim_data = simulate_realized_garch(
        params=true_params,
        T=T,
        h0=h0,
        seed=seed
    )

    returns = sim_data['returns']
    log_rv = sim_data['log_rv']
    print("Done.\n")

    # Initialize parameters
    print("Initializing parameters...")
    params_init = initialize_params(returns, log_rv)
    print("Done.\n")

    # Fit with L-BFGS
    print("Fitting model with L-BFGS...")
    print("-" * 80)
    params_opt, info = fit_realized_garch_lbfgs(
        returns=returns,
        log_rv=log_rv,
        params_init=params_init,
        maxiter=500,
        verbose=True
    )

    print()
    print(f"Optimization {'converged' if info['converged'] else 'did not converge'}")
    print(f"Iterations: {info['num_iterations']}")
    print(f"Final loss: {info['final_loss']:.4f}")
    print(f"Final error: {info['error']:.6e}")
    print()

    # Compare results
    params_estimated = transform_params(params_opt)

    print("Parameter Recovery:")
    print("-" * 80)
    print(f"{'Parameter':<12s} {'True':>10s} {'Estimated':>10s} {'Error':>10s} {'% Error':>10s}")
    print("-" * 80)

    for key in true_params.keys():
        true_val = true_params[key]
        est_val = params_estimated[key]
        error = est_val - true_val
        pct_error = (error / true_val * 100) if true_val != 0 else np.nan

        print(f"{key:<12s} {true_val:10.4f} {est_val:10.4f} {error:10.4f} {pct_error:9.2f}%")

    print()
    print("=" * 80)
    print("L-BFGS optimization completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
