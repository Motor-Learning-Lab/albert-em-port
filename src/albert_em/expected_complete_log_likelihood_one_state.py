"""
Expected Complete Log-Likelihood for One-State Model

Author: Python Port 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control

Summary: This module computes the expected complete log-likelihood for
   the one-state model. This is the objective function maximized in the
   M-step of the EM algorithm.

Model:
    x_{n+1} ~ N(A x_n + B e_n, Q)
    y_n ~ N(x_n, R)
"""

import numpy as np


def expected_complete_log_likelihood_one_state(
    parameters: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    xnN: np.ndarray,
    VnN: np.ndarray,
    Vnp1nN: np.ndarray,
) -> float:
    """
    Computes the expected complete log-likelihood for the one-state model.

    Args:
        parameters: Current estimate of the one-state model parameters
                   [A, B, Q, R, x0, P0]
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        xnN: Smoothed Kalman state expectation (N,)
        VnN: Smoothed Kalman state variance (N,)
        Vnp1nN: Smoothed Kalman covariance of consecutive states (N-1,)

    Returns:
        likelihood: The expected complete log-likelihood
    """

    # Extract parameter values
    A = parameters[0]
    B = parameters[1]
    Q = parameters[2]
    R = parameters[3]
    x0 = parameters[4]
    P0 = parameters[5]

    # Determine the number of trials
    N = len(y)

    ###########################################################################
    # Compute term 1: observation error term
    ###########################################################################
    term1 = 0.0
    for n in range(N):
        # Expected value of (y_n - x_n)^2
        term1 += y[n] ** 2
        term1 += VnN[n] + xnN[n] ** 2
        term1 -= 2.0 * y[n] * xnN[n]

    term1 = -term1 / (2.0 * R)

    ###########################################################################
    # Compute term 2: dynamics error term
    ###########################################################################
    term2 = 0.0
    for n in range(N - 1):
        # Expected value of (x_{n+1} - A*x_n - B*e_n)^2
        term2 += VnN[n + 1] + xnN[n + 1] ** 2
        term2 -= 2.0 * xnN[n + 1] * A * xnN[n]
        term2 -= 2.0 * A * Vnp1nN[n]
        term2 -= 2.0 * xnN[n + 1] * B * e[n]
        term2 += A * A * (VnN[n] + xnN[n] ** 2)
        term2 += 2.0 * A * B * xnN[n] * e[n]
        term2 += B * B * e[n] * e[n]

    term2 = -term2 / (2.0 * Q)

    ###########################################################################
    # Compute term 3: initial state error term
    ###########################################################################
    term3 = 0.0
    # Expected value of (x_0 - x0)^2
    term3 += VnN[0] + xnN[0] ** 2
    term3 -= 2.0 * xnN[0] * x0
    term3 += x0 * x0

    term3 = -term3 / (2.0 * P0)

    ###########################################################################
    # Compute term 4: log-determinant terms (constants omitted)
    ###########################################################################
    term4 = -0.5 * np.log(P0)
    term4 -= (N / 2.0) * np.log(R)
    # Note: Constant term -(3/2)*N*log(2π) omitted (does not affect optimization)

    ###########################################################################
    # Compute term 5: process noise log-determinant
    ###########################################################################
    term5 = -(N - 1) * np.log(Q) / 2.0

    # Sum all terms
    likelihood = term1 + term2 + term3 + term4 + term5

    return likelihood
