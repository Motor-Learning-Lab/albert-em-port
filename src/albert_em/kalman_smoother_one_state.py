"""
Kalman Smoother for One-State Model

Author: Python Port 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control

Summary: This module implements the Kalman smoother for the one-state model.
   It computes the expected value of the state and covariances, given a set
   of behavioral observations and the one-state model parameters.

Model:
    x_{n+1} ~ N(A x_n + B e_n, Q)
    y_n ~ N(x_n, R)
"""

import numpy as np
from typing import Tuple


def kalman_smoother_one_state(
    parameters: np.ndarray, y: np.ndarray, e: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements the Kalman smoother for a one-state model.

    Args:
        parameters: Current estimate of the one-state model parameters
                   [A, B, Q, R, x0, P0]
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)

    Returns:
        xnN: Smoothed Kalman state expectation E[x(n)|y(1),y(2),...,y(N)]
             (N,)
        VnN: Smoothed Kalman state variance var(x(n)|y(1),y(2),...,y(N))
             (N,)
        Vnp1nN: Smoothed Kalman covariance of consecutive states
                cov(x(n+1),x(n)|y(1),y(2),...,y(N))
                (N-1,)
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
    xnnm1 = np.zeros(N)  # Prior expectations
    Vnnm1 = np.zeros(N)  # Prior variances
    xnn = np.zeros(N)  # Posterior expectations
    Vnn = np.zeros(N)  # Posterior variances

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
    # Kalman smoother
    ###########################################################################

    # Allocate space for the smoothed expectations and variances
    xnN = np.zeros(N)
    VnN = np.zeros(N)

    # Instantiate the expectation and variance of the final trial as the
    # posteriors obtained at the end of the forward Kalman filter
    xnN[-1] = xnn[-1]
    VnN[-1] = Vnn[-1]

    # Allocate space for the J parameter
    Jn = np.zeros(N)

    # Backwards recursions for Kalman smoothing
    for n in range(N - 2, -1, -1):
        # Compute J
        Jn[n] = Vnn[n] * A / Vnnm1[n + 1]

        # Compute the smoothed variance
        VnN[n] = Vnn[n] + Jn[n] * (VnN[n + 1] - Vnnm1[n + 1]) * Jn[n]

        # Compute the smoothed expectation
        xnN[n] = xnn[n] + Jn[n] * (xnN[n + 1] - xnnm1[n + 1])

    # Compute the smoothed covariances
    Vnp1nN = np.zeros(N - 1)
    for n in range(N - 1):
        Vnp1nN[n] = VnN[n + 1] * Jn[n]

    return xnN, VnN, Vnp1nN
