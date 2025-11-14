"""
Incomplete Log-Likelihood Function

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module computes the incomplete log-likelihood function.
   The EM algorithm attempts to increase the value of this function each
   iteration. It is the function maximized in standard MLE.
"""

import numpy as np


def incomplete_log_likelihood(
    y: np.ndarray, e: np.ndarray, c: np.ndarray, parameters: np.ndarray
) -> float:
    """
    Computes the incomplete log-likelihood function.

    Args:
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        c: Model parameter that is assumed invariant (2,)
        parameters: Two state model parameter set
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]

    Returns:
        likelihood: The incomplete log-likelihood for this parameter set
    """

    # Store input variables using descriptive names
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

    # Reshape c for matrix operations
    c = c.reshape(-1, 1)

    ###########################################################################
    # Forward Kalman filter
    ###########################################################################

    # Allocate space for arrays for the prior and posterior expectations and
    # variances of the hidden states
    xnnm1 = [None] * N
    Vnnm1 = [None] * N
    xnn = [None] * N
    Vnn = [None] * N

    # Specify the initial prior, x(1|0) = x1
    xnnm1[0] = x1.copy()

    # Specify the initial prior variance, V(1|0) = V1
    Vnnm1[0] = V1.copy()

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

    # Compute the log-likelihood, log[L(y(1),y(2),...y(N)|parameters)]
    likelihood = -(N / 2) * np.log(2 * np.pi)
    for n in range(N):
        # The variance and mean of the normal random variable
        SIGMA = (c.T @ Vnnm1[n] @ c + sigmau2)[0, 0]
        MU = (c.T @ xnnm1[n].reshape(-1, 1))[0, 0]

        # Update the likelihood
        likelihood += -0.5 * np.log(SIGMA) - 0.5 * ((y[n] - MU) ** 2) / SIGMA

    return likelihood
