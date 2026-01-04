"""
Tests for One-State EM Model

This module contains comprehensive tests for the one-state model implementation:
1. Parameter recovery test
2. EM monotonicity test
3. Integration tests
"""

import numpy as np
import pytest
from albert_em import (
    fit_one_state,
    pack_one_state_params,
    one_state_simulation_with_noise,
)


class TestOneStateParameterRecovery:
    """Test that the one-state model can recover simulated parameters."""

    def test_parameter_recovery_simple(self):
        """Test parameter recovery with a simple scenario."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # True parameters for simulation
        A_true = 0.95
        B_true = 0.15
        Q_true = 0.01
        R_true = 0.05
        x0_true = 0.0
        P0_true = 0.1

        true_params = pack_one_state_params(
            A_true, B_true, Q_true, R_true, x0_true, P0_true
        )

        # Generate simulation data
        N = 300  # Number of trials
        r = np.zeros(N)
        r[50:150] = 10.0  # Perturbation block

        EC = np.zeros(N)
        EC_value = np.zeros(N)

        # Simulate data
        y, x = one_state_simulation_with_noise(true_params, r, EC, EC_value)

        # Initial parameter guess (somewhat off from true values)
        A_init = 0.9
        B_init = 0.2
        Q_init = 0.02
        R_init = 0.1
        x0_init = 0.0
        P0_init = 0.2

        init_params = pack_one_state_params(
            A_init, B_init, Q_init, R_init, x0_init, P0_init
        )

        # Search space bounds
        search_space = np.array(
            [
                [0.5, 1.0],  # A
                [0.0, 0.5],  # B
                [0.001, 0.5],  # Q
                [0.001, 0.5],  # R
                [-1.0, 1.0],  # x0
                [0.001, 1.0],  # P0
            ]
        )

        # Run EM
        num_iterations = 20
        fitted_params, likelihoods = fit_one_state(
            init_params, y, r, EC, EC_value, search_space, num_iterations
        )

        # Extract fitted parameters
        A_fit = fitted_params[0]
        B_fit = fitted_params[1]
        Q_fit = fitted_params[2]
        R_fit = fitted_params[3]
        x0_fit = fitted_params[4]
        P0_fit = fitted_params[5]

        # Check that fitted parameters are close to true parameters
        # Use relatively lenient tolerances since we're working with noisy data
        assert np.abs(A_fit - A_true) < 0.05, f"A recovery failed: {A_fit} vs {A_true}"
        assert np.abs(B_fit - B_true) < 0.05, f"B recovery failed: {B_fit} vs {B_true}"
        assert np.abs(Q_fit - Q_true) < 0.02, f"Q recovery failed: {Q_fit} vs {Q_true}"
        assert np.abs(R_fit - R_true) < 0.02, f"R recovery failed: {R_fit} vs {R_true}"

        print(f"Parameter Recovery Results:")
        print(f"  A: true={A_true:.4f}, fitted={A_fit:.4f}, diff={A_fit-A_true:.4f}")
        print(f"  B: true={B_true:.4f}, fitted={B_fit:.4f}, diff={B_fit-B_true:.4f}")
        print(f"  Q: true={Q_true:.4f}, fitted={Q_fit:.4f}, diff={Q_fit-Q_true:.4f}")
        print(f"  R: true={R_true:.4f}, fitted={R_fit:.4f}, diff={R_fit-R_true:.4f}")


class TestOneStateMonotonicity:
    """Test that incomplete log-likelihood increases across EM iterations."""

    def test_em_monotonicity(self):
        """Verify that likelihood is non-decreasing across iterations."""
        # Set random seed for reproducibility
        np.random.seed(123)

        # Parameters for simulation
        A_true = 0.92
        B_true = 0.18
        Q_true = 0.015
        R_true = 0.06
        x0_true = 0.0
        P0_true = 0.12

        true_params = pack_one_state_params(
            A_true, B_true, Q_true, R_true, x0_true, P0_true
        )

        # Generate simulation data
        N = 200
        r = np.zeros(N)
        r[40:120] = 8.0

        EC = np.zeros(N)
        EC_value = np.zeros(N)

        y, x = one_state_simulation_with_noise(true_params, r, EC, EC_value)

        # Initial parameter guess
        init_params = pack_one_state_params(0.85, 0.25, 0.03, 0.12, 0.0, 0.25)

        # Search space bounds
        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-1.0, 1.0],
                [0.001, 1.0],
            ]
        )

        # Run EM
        num_iterations = 15
        fitted_params, likelihoods = fit_one_state(
            init_params, y, r, EC, EC_value, search_space, num_iterations
        )

        # Check monotonicity
        for i in range(1, num_iterations):
            assert likelihoods[i] >= likelihoods[i - 1] - 1e-6, (
                f"Likelihood decreased at iteration {i}: "
                f"{likelihoods[i-1]:.4f} -> {likelihoods[i]:.4f}"
            )

        print(f"Monotonicity test passed!")
        print(f"  Initial likelihood: {likelihoods[0]:.4f}")
        print(f"  Final likelihood: {likelihoods[-1]:.4f}")
        print(f"  Improvement: {likelihoods[-1] - likelihoods[0]:.4f}")


class TestOneStateIntegration:
    """Integration tests for one-state model components."""

    def test_basic_fitting_workflow(self):
        """Test that the entire fitting workflow executes without errors."""
        np.random.seed(999)

        # Simple test case
        params = pack_one_state_params(0.9, 0.2, 0.02, 0.08, 0.0, 0.15)

        N = 150
        r = np.zeros(N)
        r[30:80] = 5.0

        EC = np.zeros(N)
        EC_value = np.zeros(N)

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        # Fit with same initial parameters
        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-1.0, 1.0],
                [0.001, 1.0],
            ]
        )

        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 10
        )

        # Basic sanity checks
        assert fitted_params.shape == (6,), "Fitted params should have 6 elements"
        assert likelihoods.shape == (10,), "Should have 10 likelihood values"
        assert np.all(np.isfinite(fitted_params)), "Fitted params should be finite"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods should be finite"

        print("Integration test passed!")

    def test_error_clamp_trials(self):
        """Test that error-clamp trials are handled correctly."""
        np.random.seed(777)

        params = pack_one_state_params(0.88, 0.22, 0.025, 0.07, 0.0, 0.18)

        N = 100
        r = np.zeros(N)
        r[20:60] = 6.0

        # Add error-clamp trials
        EC = np.zeros(N)
        EC[70:90] = 1.0
        EC_value = np.zeros(N)
        EC_value[70:90] = 2.0  # Clamp error to 2.0

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-1.0, 1.0],
                [0.001, 1.0],
            ]
        )

        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 8
        )

        # Should complete without errors
        assert np.all(np.isfinite(fitted_params))
        assert np.all(np.isfinite(likelihoods))

        print("Error-clamp test passed!")


if __name__ == "__main__":
    # Run tests manually
    print("=" * 70)
    print("Testing One-State Parameter Recovery")
    print("=" * 70)
    test_recovery = TestOneStateParameterRecovery()
    test_recovery.test_parameter_recovery_simple()

    print("\n" + "=" * 70)
    print("Testing EM Monotonicity")
    print("=" * 70)
    test_monotonicity = TestOneStateMonotonicity()
    test_monotonicity.test_em_monotonicity()

    print("\n" + "=" * 70)
    print("Testing Integration")
    print("=" * 70)
    test_integration = TestOneStateIntegration()
    test_integration.test_basic_fitting_workflow()
    test_integration.test_error_clamp_trials()

    print("\n" + "=" * 70)
    print("All one-state tests passed!")
    print("=" * 70)
