"""
Incomplete Log-Likelihood Function for One-State Model

Author: Python Port 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control

Summary: This module computes the incomplete log-likelihood function for
   the one-state model. The EM algorithm attempts to increase the value
   of this function each iteration.

Model:
    x_{n+1} ~ N(A x_n + B e_n, Q)
    y_n ~ N(x_n, R)
"""

import numpy as np


def incomplete_log_likelihood_one_state(
    y: np.ndarray, e: np.ndarray, parameters: np.ndarray
) -> float:
    """
    Computes the incomplete log-likelihood function for the one-state model.

    Args:
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        parameters: One-state model parameter set [A, B, Q, R, x0, P0]

    Returns:
        likelihood: The incomplete log-likelihood for this parameter set
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
    # Forward Kalman filter
    ###########################################################################

    # Allocate space for arrays for the prior and posterior expectations and
    # variances of the hidden states
    xnnm1 = [None] * N
    Vnnm1 = [None] * N
    xnn = [None] * N
    Vnn = [None] * N

    # Specify the initial prior, x(1|0) = x0
    xnnm1[0] = x0

    # Specify the initial prior variance, V(1|0) = P0
    Vnnm1[0] = P0

    # Standard forward Kalman filter
    for n in range(N):
        # Compute the Kalman gain
        k_denom = Vnnm1[n] + R
        k = Vnnm1[n] / k_denom

        # Compute the error between our actual and predicted y values
        y_error = y[n] - xnnm1[n]

        # Compute the posterior state expectation
        xnn[n] = xnnm1[n] + k * y_error

        # Compute the posterior state variance
        Vnn[n] = (1 - k) * Vnnm1[n]

        # Forward project, unless the last trial has been reached
        if n < N - 1:
            # Compute the next prior state
            xnnm1[n + 1] = A * xnn[n] + B * e[n]

            # Compute the next prior variance
            Vnnm1[n + 1] = A * Vnn[n] * A + Q

    ###########################################################################
    # Compute the incomplete log-likelihood
    ###########################################################################

    # Compute the log-likelihood, log[L(y(1),y(2),...y(N)|parameters)]
    likelihood = -(N / 2) * np.log(2 * np.pi)
    for n in range(N):
        # The variance and mean of the normal random variable
        SIGMA = Vnnm1[n] + R
        MU = xnnm1[n]

        # Update the likelihood
        likelihood += -0.5 * np.log(SIGMA) - 0.5 * ((y[n] - MU) ** 2) / SIGMA

    return likelihood
