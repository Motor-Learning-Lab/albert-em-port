"""
Simple test script to verify the EM algorithm implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generalized_expectation_maximization import generalized_expectation_maximization
from two_state_simulation import two_state_simulation_with_noise


def test_em_algorithm():
    """
    Run a simple test of the EM algorithm.
    """
    print("Testing EM Algorithm Implementation")
    print("=" * 60)

    # Define simple paradigm
    r = np.concatenate(
        [np.zeros(20), 30 * np.ones(50), np.full(20, np.nan), np.zeros(30)]
    )
    EC = np.concatenate([np.zeros(70), np.ones(20), np.zeros(30)])
    EC_value = np.concatenate([np.full(70, np.nan), np.zeros(20), np.full(30, np.nan)])
    c = np.array([1.0, 1.0])

    # True parameters
    true_params = np.array([0.985, 0.556, 0.097, 0.213, 0.0, 0.0, 1.694, 1.037, 0.0])

    # Simulate data
    np.random.seed(5)
    print("\n1. Simulating behavioral data...")
    y, xS, xF = two_state_simulation_with_noise(true_params, r, EC, EC_value, c)
    print(f"   Generated {len(y)} trials of data")

    # Initial guess
    initial_params = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])

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

    # Run EM with fewer iterations for quick test
    print("\n2. Running EM algorithm (10 iterations for quick test)...")
    fitted_params, likelihoods = generalized_expectation_maximization(
        initial_params,
        y,
        r,
        EC,
        EC_value,
        c,
        search_space,
        constraints,
        num_iterations=10,
    )

    # Display results
    print("\n3. Results:")
    print("-" * 60)
    print(f"{'Parameter':<25} {'True':>10} {'Initial':>10} {'Fitted':>10}")
    print("-" * 60)

    param_names = [
        "aS",
        "aF",
        "bS",
        "bF",
        "xS1",
        "xF1",
        "sigmax2",
        "sigmau2",
        "sigma12",
    ]

    for i, name in enumerate(param_names):
        print(
            f"{name:<25} {true_params[i]:>10.4f} "
            f"{initial_params[i]:>10.4f} {fitted_params[i]:>10.4f}"
        )

    print("-" * 60)
    print(f"\nInitial log-likelihood: {likelihoods[0]:.4f}")
    print(f"Final log-likelihood:   {likelihoods[-1]:.4f}")
    print(f"Likelihood increase:    {likelihoods[-1] - likelihoods[0]:.4f}")

    # Check if likelihood is increasing
    if np.all(np.diff(likelihoods) >= -1e-6):  # Allow small numerical errors
        print("\n✓ Test PASSED: Likelihood is non-decreasing")
        return True
    else:
        print("\n✗ Test FAILED: Likelihood decreased")
        return False


if __name__ == "__main__":
    try:
        success = test_em_algorithm()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
