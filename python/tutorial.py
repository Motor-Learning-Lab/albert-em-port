"""
Tutorial: EM Algorithm for Two-State Model Fitting

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This script demonstrates how to use EM to fit a two state model
   of adaptation to a sensorimotor task that includes error-clamp trials.
   First, behavior is simulated according to a two state model of learning.
   Second, EM is used to fit the behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from generalized_expectation_maximization import generalized_expectation_maximization
from two_state_simulation import (
    two_state_simulation_with_noise,
    two_state_simulation_without_noise,
)


def main():
    """
    Main tutorial function demonstrating the EM algorithm.
    """

    ###########################################################################
    # PART 1: The Paradigm
    ###########################################################################
    # Here we specify the experimental paradigm. The paradigm is defined by
    # the sequence of error-clamp trials and perturbations.

    # The perturbation schedule
    # A value of zero indicates no perturbation
    # A NaN value indicates an error-clamp trial
    # A non-zero value indicates a non-zero perturbation
    r = np.concatenate(
        [np.zeros(20), 30 * np.ones(50), np.full(20, np.nan), np.zeros(30)]
    )

    # The error-clamp sequence
    # A value of zero indicates this trial is not an error-clamp trial
    # A non-zero entry indicates that this trial is an error-clamp trial
    EC = np.concatenate([np.zeros(20), np.zeros(50), np.ones(20), np.zeros(30)])

    # The value of the error during error-clamp trials
    # A NaN value indicates that this trial was not an error-clamp trial
    # A numeric value indicates the clamped error on this trial
    EC_value = np.concatenate(
        [np.full(20, np.nan), np.full(50, np.nan), np.zeros(20), np.full(30, np.nan)]
    )

    # Get the number of trials
    N = len(r)

    # Specify the c parameter
    # Should be equal to [1, 1] if it is assumed that behavior is
    # determined by the unweighted sum of the fast and slow states
    c = np.array([1.0, 1.0])

    ###########################################################################
    # PART 2: Simulation of behavior
    ###########################################################################
    # Specify a two-state model parameter set for simulation
    aS = 0.985  # the slow state retention factor
    aF = 0.556  # the fast state retention factor
    bS = 0.097  # the slow state error sensitivity
    bF = 0.213  # the fast state error sensitivity
    xS1 = 0.0  # the initial slow state
    xF1 = 0.0  # the initial fast state
    sigmax2 = 1.694  # variance of the update of the slow state
    sigmau2 = 1.037  # variance in the execution of a movement
    sigma12 = 0.0  # variance in the initial state

    # Construct a parameter set from the two state model parameters
    simulation_parameters = np.array(
        [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
    )

    # Seed the random number generator to get the same simulation results
    # across different runs of this script
    np.random.seed(5)

    # Simulate behavior corresponding to the two state model parameter set
    # and the paradigm
    y, xS, xF = two_state_simulation_with_noise(
        simulation_parameters, r, EC, EC_value, c
    )

    ###########################################################################
    # PART 3: Fit two state model with EM
    ###########################################################################

    # Specify the number of EM iterations
    num_iterations = 100

    # Set the boundaries for the parameter search
    # Each row specifies the min,max pair for a two state model parameter
    search_space = np.array(
        [
            [0, 1.1],  # bounds for the slow state retention factor
            [0, 1.1],  # bounds for the fast state retention factor
            [0, 1],  # bounds for the slow state error sensitivity
            [0, 1],  # bounds for the fast state error sensitivity
            [-30, 30],  # bounds for the initial slow state
            [-30, 30],  # bounds for the initial fast state
            [0.0000001, 10],  # bounds for state noise variance
            [0.0000001, 10],  # bounds for motor variance
            [0.0000001, 10],  # bounds for initial state variance
        ]
    )

    # Specify inequalities relating the parameter values to enforce two state
    # model dynamics
    constraints = np.array(
        [
            0.001,  # deltaA for the inequality: aSlow >= deltaA + aFast
            0.001,  # deltaB for the inequality: bFast >= deltaB + bSlow
        ]
    )

    # Specify an initial parameter guess for the EM algorithm
    IC_EM = np.array(
        [
            0.95,  # initial guess of slow state retention factor
            0.7,  # initial guess of fast state retention factor
            0.05,  # initial guess for the slow state error sensitivity
            0.30,  # initial guess for the fast state error sensitivity
            0,  # initial guess for the initial slow state
            0,  # initial guess for the initial fast state
            2,  # initial guess for the state update variance
            2,  # initial guess for the motor variance
            5,  # initial guess for the initial state variance
        ]
    )

    # Use the EM algorithm to fit the two state model
    print("Running EM algorithm...")
    parameters_EM, likelihoods_EM = generalized_expectation_maximization(
        IC_EM, y, r, EC, EC_value, c, search_space, constraints, num_iterations
    )

    print("\nEM Algorithm Results:")
    print("=" * 60)
    print(f"{'Parameter':<25} {'True Value':>12} {'EM Estimate':>12}")
    print("-" * 60)
    param_names = [
        "aS (slow retention)",
        "aF (fast retention)",
        "bS (slow sensitivity)",
        "bF (fast sensitivity)",
        "xS1 (initial slow)",
        "xF1 (initial fast)",
        "sigmax2 (state noise)",
        "sigmau2 (motor noise)",
        "sigma12 (initial var)",
    ]
    for i, name in enumerate(param_names):
        print(f"{name:<25} {simulation_parameters[i]:>12.6f} {parameters_EM[i]:>12.6f}")
    print("=" * 60)

    ###########################################################################
    # PART 4: Visualize results
    ###########################################################################

    # Use the EM parameters to simulate the behavior, fast, and slow states
    y_EM, xS_EM, xF_EM = two_state_simulation_without_noise(
        parameters_EM, r, EC, EC_value, c
    )

    # Plot the paradigm, the simulated behavior, and the EM fit
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Behavior
    axes[0].plot(range(1, N + 1), EC, "b", label="error-clamp", linewidth=1.5)
    axes[0].plot(range(1, N + 1), r, "r", label="perturbation", linewidth=1.5)
    axes[0].plot(range(1, N + 1), y, "-k", label="motor output", linewidth=1.5)
    axes[0].plot(range(1, N + 1), y_EM, "--k", label="EM", linewidth=2)
    axes[0].set_xlim([1, N])
    axes[0].legend()
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Motor output")
    axes[0].set_title("Behavior and Model Fit")
    axes[0].grid(True, alpha=0.3)

    # Slow state
    axes[1].plot(range(1, N + 1), xS, "-k", label="Truth", linewidth=1.5)
    axes[1].plot(range(1, N + 1), xS_EM, "--k", label="EM", linewidth=2)
    axes[1].set_xlim([1, N])
    axes[1].legend()
    axes[1].set_xlabel("Trial number")
    axes[1].set_ylabel("Slow state of learning")
    axes[1].set_title("Slow State")
    axes[1].grid(True, alpha=0.3)

    # Fast state
    axes[2].plot(range(1, N + 1), xF, "-k", label="Truth", linewidth=1.5)
    axes[2].plot(range(1, N + 1), xF_EM, "--k", label="EM", linewidth=2)
    axes[2].set_xlim([1, N])
    axes[2].legend()
    axes[2].set_xlabel("Trial number")
    axes[2].set_ylabel("Fast state of learning")
    axes[2].set_title("Fast State")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tutorial_behavior_fit.png", dpi=150)
    print("\nSaved behavior fit plot to 'tutorial_behavior_fit.png'")

    # Plot the incomplete log-likelihood as a function of EM iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, num_iterations + 1), likelihoods_EM, "-k", linewidth=2)
    ax.set_xlim([1, num_iterations])
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log[L(y|θ)]")
    ax.set_title("Incomplete Log-Likelihood vs EM Iteration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("tutorial_likelihood.png", dpi=150)
    print("Saved likelihood plot to 'tutorial_likelihood.png'")

    plt.show()


if __name__ == "__main__":
    main()
