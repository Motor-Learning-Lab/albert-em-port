"""
Generalized Expectation-Maximization Algorithm

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module coordinates the EM algorithm. It successively
   calls the Kalman Smoother to specify the E-Step, and then performs
   the M-step to update the parameter set.
"""

import numpy as np
import warnings
from typing import Tuple
from albert_em.kalman_smoother import kalman_smoother
from albert_em.m_step import m_step
from albert_em.incomplete_log_likelihood import incomplete_log_likelihood


def generalized_expectation_maximization(
    parameters: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    c: np.ndarray,
    search_space: np.ndarray,
    constraints: np.ndarray,
    num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinates the EM algorithm for fitting a two-state model.

    Args:
        parameters: Initial guess of the two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        y: Motor output on each trial (N,)
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
            Non-zero indicates error-clamp trial
        EC_value: Array indicating the value of the clamped error (N,)
        c: Model parameter that is assumed invariant (2,)
        search_space: Matrix containing upper and lower bounds (n_params x 2)
        constraints: Array specifying inequality constraints [deltaA, deltaB]
        num_iterations: Number of EM iterations

    Returns:
        parameters: Final parameters obtained at the conclusion of EM
        likelihoods: Array containing the value of the incomplete
                    log-likelihood function on each iteration
    """

    # Compute the error on each trial
    N = len(y)
    e = np.zeros(N)

    for n in range(N):
        if EC[n] == 0:
            # This is not an error-clamp trial
            e[n] = r[n] - y[n]
        else:
            # This is an error-clamp trial
            e[n] = EC_value[n]

    # Create an array that stores the value of the incomplete log-likelihood
    # function on each iteration of the EM algorithm
    likelihoods = np.zeros(num_iterations)

    # The EM algorithm
    for n in range(num_iterations):
        # E-step:
        # Get the smoothed Kalman estimates of states, variances, and
        # covariances using a Kalman smoother
        xnN, VnN, Vnp1nN = kalman_smoother(parameters, y, e, c)

        # M-step:
        # Perform maximum likelihood estimation in a constrained parameter
        # space
        parameters = m_step(
            parameters, xnN, VnN, Vnp1nN, y, e, c, search_space, constraints
        )

        # Compute the incomplete log-likelihood for this parameter set
        likelihoods[n] = incomplete_log_likelihood(y, e, c, parameters)

        # Check to make sure that the likelihood function has increased
        if (n > 0) and (likelihoods[n] < likelihoods[n - 1]):
            # The likelihood has not increased, warn the user
            warnings.warn(
                "The expected complete log-likelihood function has stopped increasing"
            )

    return parameters, likelihoods
