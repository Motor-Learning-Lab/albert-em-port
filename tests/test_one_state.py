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


class TestOneStateEdgeCases:
    """Test edge cases and boundary conditions for robustness."""

    def test_minimal_data_n1(self):
        """Test with N=1 (single trial) - boundary condition."""
        np.random.seed(99)

        params = pack_one_state_params(0.90, 0.20, 0.05, 0.1, 0.0, 0.2)

        # Single trial
        N = 1
        r = np.array([5.0])
        EC = np.array([0.0])
        EC_value = np.array([0.0])

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-5.0, 5.0],
                [0.001, 1.0],
            ]
        )

        # Should not crash with N=1
        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 3
        )

        assert np.all(np.isfinite(fitted_params)), "Params should be finite for N=1"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods should be finite for N=1"

        print("N=1 edge case test passed!")

    def test_minimal_data_n2(self):
        """Test with N=2 (two trials) - minimal for lag covariance."""
        np.random.seed(100)

        params = pack_one_state_params(0.85, 0.25, 0.03, 0.08, 0.0, 0.15)

        # Two trials
        N = 2
        r = np.array([0.0, 8.0])
        EC = np.array([0.0, 0.0])
        EC_value = np.array([0.0, 0.0])

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-5.0, 5.0],
                [0.001, 1.0],
            ]
        )

        # Should handle N=2 (minimum for RTS smoothing lag covariance)
        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 3
        )

        assert np.all(np.isfinite(fitted_params)), "Params should be finite for N=2"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods should be finite for N=2"

        print("N=2 edge case test passed!")

    def test_all_error_clamp_trials(self):
        """Test when all trials are error-clamp trials."""
        np.random.seed(101)

        params = pack_one_state_params(0.92, 0.18, 0.02, 0.06, 0.0, 0.12)

        N = 50
        r = np.full(N, np.nan)  # All trials clamped, no perturbations observed
        EC = np.ones(N)  # All error-clamp
        EC_value = np.full(N, 1.5)  # Clamp error to constant value

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-2.0, 2.0],
                [0.001, 1.0],
            ]
        )

        # Should handle all-clamp scenario
        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 5
        )

        assert np.all(np.isfinite(fitted_params)), "Params finite with all error-clamp"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods finite with all clamp"

        print("All error-clamp trials test passed!")

    def test_no_error_clamp_trials(self):
        """Test with no error-clamp trials (explicit test)."""
        np.random.seed(102)

        params = pack_one_state_params(0.93, 0.17, 0.04, 0.09, 0.0, 0.11)

        N = 60
        r = np.zeros(N)
        r[15:45] = 12.0

        EC = np.zeros(N)  # Explicit: no error-clamp trials
        EC_value = np.zeros(N)

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [0.001, 0.5],
                [0.001, 0.5],
                [-3.0, 3.0],
                [0.001, 1.0],
            ]
        )

        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 8
        )

        assert np.all(np.isfinite(fitted_params)), "Params finite with no clamp"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods finite with no clamp"

        # Should show improvement
        assert likelihoods[-1] > likelihoods[0], "Likelihood should improve"

        print("No error-clamp trials test passed!")

    def test_degenerate_noise_parameters(self):
        """Test with very small Q and R (near-degenerate case)."""
        np.random.seed(103)

        # Use very small noise values
        params = pack_one_state_params(A=0.90, B=0.20, Q=1e-5, R=1e-5, x0=0.0, P0=0.1)

        N = 80
        r = np.zeros(N)
        r[20:50] = 7.0

        EC = np.zeros(N)
        EC_value = np.zeros(N)

        y, x = one_state_simulation_with_noise(params, r, EC, EC_value)

        # Search space allows small values
        search_space = np.array(
            [
                [0.5, 1.0],
                [0.0, 0.5],
                [1e-6, 0.1],  # Very small Q allowed
                [1e-6, 0.1],  # Very small R allowed
                [-2.0, 2.0],
                [0.001, 1.0],
            ]
        )

        # Should handle near-degenerate noise
        fitted_params, likelihoods = fit_one_state(
            params, y, r, EC, EC_value, search_space, 6
        )

        assert np.all(np.isfinite(fitted_params)), "Params finite with small Q/R"
        assert np.all(np.isfinite(likelihoods)), "Likelihoods finite with small Q/R"

        # Check that we don't have numerical collapse
        assert fitted_params[2] > 0, "Q should remain positive"
        assert fitted_params[3] > 0, "R should remain positive"

        print("Degenerate noise parameters test passed!")


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
    print("Testing Edge Cases")
    print("=" * 70)
    test_edge_cases = TestOneStateEdgeCases()
    test_edge_cases.test_minimal_data_n1()
    test_edge_cases.test_minimal_data_n2()
    test_edge_cases.test_all_error_clamp_trials()
    test_edge_cases.test_no_error_clamp_trials()
    test_edge_cases.test_degenerate_noise_parameters()

    print("\n" + "=" * 70)
    print("All one-state tests passed!")
    print("=" * 70)
