"""
Benchmark tests for the EM algorithm implementation.

This module tests the performance of the Numba-optimized likelihood function
compared to a pure Python/NumPy implementation.
"""

import numpy as np
import pytest
from albert_em import (
    generalized_expectation_maximization,
    two_state_simulation_with_noise,
    kalman_smoother,
)


def generate_test_data(N=120, seed=42):
    """Generate synthetic test data for benchmarking."""
    np.random.seed(seed)

    # Paradigm
    r = np.concatenate(
        [np.zeros(20), 30 * np.ones(50), np.full(20, np.nan), np.zeros(30)]
    )
    EC = np.concatenate([np.zeros(70), np.ones(20), np.zeros(30)])
    EC_value = np.concatenate([np.full(70, np.nan), np.zeros(20), np.full(30, np.nan)])
    c = np.array([1.0, 1.0])

    # True parameters
    params = np.array([0.985, 0.556, 0.097, 0.213, 0.0, 0.0, 1.694, 1.037, 0.0])

    # Simulate data
    y, _, _ = two_state_simulation_with_noise(params, r, EC, EC_value, c)

    return y, r, EC, EC_value, c


def test_em_convergence():
    """Test that EM algorithm converges (likelihood increases)."""
    y, r, EC, EC_value, c = generate_test_data()

    # Initial guess
    params_init = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])

    # Search space
    search_space = np.array(
        [
            [0, 1.1],
            [0, 1.1],
            [0, 1],
            [0, 1],
            [-30, 30],
            [-30, 30],
            [0.0000001, 10],
            [0.0000001, 10],
            [0.0000001, 10],
        ]
    )

    # Constraints
    constraints = np.array([0.001, 0.001])

    # Run EM for 10 iterations
    params_final, likelihoods = generalized_expectation_maximization(
        params_init, y, r, EC, EC_value, c, search_space, constraints, 10
    )

    # Check that likelihood generally increases (allowing small numerical errors)
    assert np.all(np.diff(likelihoods) >= -1e-6), "Likelihood should not decrease"
    assert likelihoods[-1] > likelihoods[0], "Likelihood should improve"


def test_em_single_iteration(benchmark):
    """Benchmark a single EM iteration."""
    y, r, EC, EC_value, c = generate_test_data()

    params_init = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])
    search_space = np.array(
        [
            [0, 1.1],
            [0, 1.1],
            [0, 1],
            [0, 1],
            [-30, 30],
            [-30, 30],
            [0.0000001, 10],
            [0.0000001, 10],
            [0.0000001, 10],
        ]
    )
    constraints = np.array([0.001, 0.001])

    # Benchmark single iteration
    result = benchmark(
        generalized_expectation_maximization,
        params_init,
        y,
        r,
        EC,
        EC_value,
        c,
        search_space,
        constraints,
        1,
    )

    params_final, likelihoods = result
    assert len(likelihoods) == 1


def test_kalman_smoother_performance(benchmark):
    """Benchmark the Kalman smoother (E-step)."""
    y, r, EC, EC_value, c = generate_test_data()

    params = np.array([0.985, 0.556, 0.097, 0.213, 0.0, 0.0, 1.694, 1.037, 0.0])

    # Compute errors
    N = len(y)
    e = np.zeros(N)
    for n in range(N):
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    # Benchmark Kalman smoother
    result = benchmark(kalman_smoother, params, y, e, c)
    xnN, VnN, Vnp1nN = result
    assert len(xnN) == N
    assert len(VnN) == N
    assert len(Vnp1nN) == N


if __name__ == "__main__":
    # Run basic convergence test
    print("Testing EM convergence...")
    test_em_convergence()
    print("✓ EM convergence test passed\n")

    # Run performance tests manually
    print("Running performance tests...")
    import time

    y, r, EC, EC_value, c = generate_test_data()
    params_init = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])
    search_space = np.array(
        [
            [0, 1.1],
            [0, 1.1],
            [0, 1],
            [0, 1],
            [-30, 30],
            [-30, 30],
            [0.0000001, 10],
            [0.0000001, 10],
            [0.0000001, 10],
        ]
    )
    constraints = np.array([0.001, 0.001])

    # Time 10 iterations
    start = time.time()
    params_final, likelihoods = generalized_expectation_maximization(
        params_init, y, r, EC, EC_value, c, search_space, constraints, 10
    )
    elapsed = time.time() - start

    print(f"10 EM iterations: {elapsed:.3f} seconds ({elapsed/10:.3f} sec/iteration)")
    print(f"Final likelihood: {likelihoods[-1]:.2f}")
    print(f"Likelihood improvement: {likelihoods[-1] - likelihoods[0]:.2f}")
