"""
Kalman Smoother for Two-State Model

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module implements the Kalman smoother. It computes the
   expected value of the state and covariances, given a set of
   behavioral observations and the two state model parameters.
"""

import numpy as np
from typing import Tuple, List


def kalman_smoother(
    parameters: np.ndarray, y: np.ndarray, e: np.ndarray, c: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Implements the Kalman smoother for a two-state model.

    Args:
        parameters: Current estimate of the two-state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        c: Model parameter that is assumed invariant (2,)

    Returns:
        xnN: Smoothed Kalman state expectation E[x(n)|y(1),y(2),...,y(N)]
        VnN: Smoothed Kalman state variance var(x(n)|y(1),y(2),...,y(N))
        Vnp1nN: Smoothed Kalman covariance of consecutive states
               cov(x(n+1),x(n)|y(1),y(2),...,y(N))
    """

    # Extract parameter values
    aS = parameters[0]
    aF = parameters[1]
    bS = parameters[2]
    bF = parameters[3]
    xS1 = parameters[4]
    xF1 = parameters[5]
    sigmax2 = parameters[6]
    sigmau2 = parameters[7]
    sigma12 = parameters[8]

    # Set the means and variances for the initial states
    x1 = np.array([xS1, xF1])
    V1 = np.array([[sigma12, 0], [0, sigma12]])

    # Set matrices and vectors for the update of the fast and slow states
    b = np.array([bS, bF])
    A = np.array([[aS, 0], [0, aF]])
    Q = np.array([[sigmax2, 0], [0, sigmax2]])

    # Determine the number of trials
    N = len(y)

    ###########################################################################
    # Forward Kalman filter
    ###########################################################################

    # Allocate space for arrays for the prior and posterior expectations and
    # variances of the hidden states
    xnnm1 = [None] * N  # Prior expectations
    Vnnm1 = [None] * N  # Prior variances
    xnn = [None] * N  # Posterior expectations
    Vnn = [None] * N  # Posterior variances

    # Specify the initial prior, x(1|0) = x1
    xnnm1[0] = x1.copy()

    # Specify the initial prior variance, V(1|0) = V1
    Vnnm1[0] = V1.copy()

    # Reshape c for matrix operations
    c = c.reshape(-1, 1)

    # Standard forward Kalman filter
    for n in range(N):
        # Compute the Kalman gain
        k_denom = c.T @ Vnnm1[n] @ c + sigmau2
        k = (Vnnm1[n] @ c) / k_denom

        # Compute the error between our actual and predicted y values
        y_error = y[n] - (c.T @ xnnm1[n].reshape(-1, 1))[0, 0]

        # Compute the posterior state expectation
        xnn[n] = xnnm1[n] + (k * y_error).flatten()

        # Compute the posterior state variance
        Vnn[n] = (np.eye(2) - k @ c.T) @ Vnnm1[n]

        # Forward project, unless the last trial has been reached
        if n < N - 1:
            # Compute the next prior state
            xnnm1[n + 1] = A @ xnn[n] + b * e[n]

            # Compute the next prior variance
            Vnnm1[n + 1] = A @ Vnn[n] @ A.T + Q

    ###########################################################################
    # Kalman smoother
    ###########################################################################

    # Allocate space for the smoothed expectations and variances
    xnN = [None] * N
    VnN = [None] * N

    # Instantiate the expectation and variance of the final trial as the
    # posteriors obtained at the end of the forward Kalman filter
    xnN[-1] = xnn[-1].copy()
    VnN[-1] = Vnn[-1].copy()

    # Allocate space for the J parameter
    Jn = [None] * N

    # Backwards recursions for Kalman smoothing
    for n in range(N - 2, -1, -1):
        # Compute J
        Jn[n] = Vnn[n] @ A.T @ np.linalg.inv(Vnnm1[n + 1])

        # Compute the smoothed variance
        VnN[n] = Vnn[n] + Jn[n] @ (VnN[n + 1] - Vnnm1[n + 1]) @ Jn[n].T

        # Compute the smoothed expectation
        xnN[n] = xnn[n] + Jn[n] @ (xnN[n + 1] - xnnm1[n + 1])

    # Compute the smoothed covariances
    Vnp1nN = [None] * N
    for n in range(N - 1):
        Vnp1nN[n] = VnN[n + 1] @ Jn[n].T

    return xnN, VnN, Vnp1nN
