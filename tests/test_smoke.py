import os
import numpy as np

from albert_em import (
    generalized_expectation_maximization,
    two_state_simulation_with_noise,
    USING_NUMBA,
)


def generate_paradigm(num_trials: int = 200):
    # Simple step perturbation paradigm with final error clamp block.
    # 0-49: baseline (0), 50-129: perturbation (1), 130-169: washout (0), 170-199: error clamp (0)
    r = np.zeros(num_trials)
    r[50:130] = 1.0
    EC = np.zeros(num_trials)
    EC[170:] = 1  # mark error clamp trials
    EC_value = np.zeros(num_trials)  # clamp error to zero
    return r, EC, EC_value


def test_em_smoke():
    # True parameters used for simulation
    true_params = np.array(
        [
            0.985,  # aS retention slow
            0.60,  # aF retention fast
            0.05,  # bS learning slow
            0.40,  # bF learning fast
            0.0,  # xS1 initial slow
            0.0,  # xF1 initial fast
            1e-4,  # sigmax2 process variance
            1e-2,  # sigmau2 measurement variance
            1e-4,  # sigma12 initial variance component
        ]
    )

    r, EC, EC_value = generate_paradigm(200)
    c = np.array([1.0, 1.0])

    # Simulate noisy behavior
    y, xS, xF = two_state_simulation_with_noise(true_params, r, EC, EC_value, c)

    # Initial guess purposely different
    parameters0 = np.array(
        [
            0.90,  # aS
            0.30,  # aF
            0.02,  # bS
            0.20,  # bF
            0.0,  # xS1
            0.0,  # xF1
            5e-4,  # sigmax2
            2e-2,  # sigmau2
            5e-4,  # sigma12
        ]
    )

    # Bounds for each parameter (lower, upper)
    search_space = np.array(
        [
            [0.5, 1.05],  # aS
            [0.2, 0.95],  # aF
            [0.0, 0.2],  # bS
            [0.05, 0.8],  # bF
            [-0.5, 0.5],  # xS1
            [-0.5, 0.5],  # xF1
            [1e-6, 1e-2],  # sigmax2
            [1e-5, 5e-2],  # sigmau2
            [1e-6, 1e-2],  # sigma12
        ]
    )

    # Inequality constraints deltaA (slow faster retention) and deltaB (fast higher learning)
    constraints = np.array([0.05, 0.0])

    # Run a few EM iterations
    fitted_params, likelihoods = generalized_expectation_maximization(
        parameters0,
        y,
        r,
        EC,
        EC_value,
        c,
        search_space,
        constraints,
        num_iterations=10,
    )

    # Basic assertions
    assert not np.isnan(fitted_params).any(), "NaN in fitted parameters"
    assert not np.isnan(likelihoods).any(), "NaN in likelihood values"
    assert likelihoods[-1] > likelihoods[0], "Likelihood did not improve"
    assert np.all(fitted_params >= search_space[:, 0] - 1e-9)
    assert np.all(fitted_params <= search_space[:, 1] + 1e-9)

    # Monotonic (allow tiny numerical dip tolerance)
    diffs = np.diff(likelihoods)
    assert np.sum(diffs < -1e-6) == 0, "Non-monotonic likelihood decreases detected"

    # Check constraint satisfaction
    aS, aF, bS, bF = fitted_params[:4]
    assert aS >= aF + constraints[0] - 1e-6
    assert bF >= bS + constraints[1] - 1e-6

    # Acceleration flag exposed
    assert isinstance(USING_NUMBA, (bool, np.bool_))


def test_fallback_mode():
    # Force fallback by setting env var and reloading module in a subprocess-like manner.
    # Here we mimic by temporarily setting env var and importing importlib.reload.
    import importlib
    import sys

    # Get the module (not the function)
    ecl_module = sys.modules.get("albert_em.expected_complete_log_likelihood")
    if ecl_module is None:
        import albert_em.expected_complete_log_likelihood

        ecl_module = sys.modules["albert_em.expected_complete_log_likelihood"]

    os.environ["ALBERT_EM_DISABLE_NUMBA"] = "1"
    importlib.reload(ecl_module)
    assert ecl_module.USING_NUMBA is False

    # Minimal call to ensure function executes in fallback
    params = np.array([0.9, 0.3, 0.02, 0.2, 0.0, 0.0, 5e-4, 2e-2, 5e-4])
    y = np.zeros(5)
    r = np.zeros(5)
    c = np.array([1.0, 1.0])
    e = r - y
    from albert_em.kalman_smoother import kalman_smoother

    xnN, VnN, Vnp1nN = kalman_smoother(params, y, e, c)
    val = ecl_module.expected_complete_log_likelihood(params, y, e, c, xnN, VnN, Vnp1nN)
    assert np.isfinite(val)

    # Clean up environment variable for other tests
    os.environ.pop("ALBERT_EM_DISABLE_NUMBA", None)
