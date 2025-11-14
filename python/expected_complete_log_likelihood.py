"""
Expected Complete Log-Likelihood Function

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module computes the expected complete log-likelihood.
"""

import numpy as np
from typing import List


def expected_complete_log_likelihood(
    parameters: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    c: np.ndarray,
    xnN: List[np.ndarray],
    VnN: List[np.ndarray],
    Vnp1nN: List[np.ndarray],
) -> float:
    """
    Computes the expected complete log-likelihood.

    Args:
        parameters: Current estimate of the two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        c: Model parameter that is assumed invariant (2,)
        xnN: Smoothed Kalman state expectation
        VnN: Smoothed Kalman state variance
        Vnp1nN: Smoothed Kalman covariance of consecutive states

    Returns:
        likelihood: The expected complete log-likelihood
    """

    # Set the values of two-state parameters
    aS = parameters[0]
    aF = parameters[1]
    bS = parameters[2]
    bF = parameters[3]
    xS1 = parameters[4]
    xF1 = parameters[5]
    sigmax2 = parameters[6]
    sigmau2 = parameters[7]
    sigma12 = parameters[8]

    # Specify the mean and variance of the initial states
    x1 = np.array([xS1, xF1])
    V1 = np.array([[sigma12, 0], [0, sigma12]])

    # Store the number of trials
    N = len(y)

    # Matrices and vectors for the update of the fast and slow states
    A = np.array([[aS, 0], [0, aF]])
    b = np.array([bS, bF])
    Q = np.array([[sigmax2, 0], [0, sigmax2]])

    # Reshape c for matrix operations
    c = c.reshape(-1, 1)

    # Precompute important quantities referenced in the expected complete
    # log-likelihood function
    Qinv = np.linalg.inv(Q)
    QinvA = Qinv @ A
    Qinvb = Qinv @ b
    AtQinv = A.T @ Qinv
    AtQinvA = AtQinv @ A
    AtQinvb = A.T @ Qinvb
    btQinv = b.T @ Qinv
    btQinvA = btQinv @ A
    btQinvb = b.T @ Qinvb

    ###########################################################################
    # Compute the likelihood
    ###########################################################################

    # TERM 1: Derived from the likelihood of observing the motor output
    # given the states
    term1 = 0
    for n in range(N):
        xnN_n = xnN[n].reshape(-1, 1)
        term1 += (
            y[n] ** 2
            + (c.T @ (VnN[n] + xnN_n @ xnN_n.T) @ c)[0, 0]
            - 2 * y[n] * (c.T @ xnN_n)[0, 0]
        )
    term1 = -term1 / (2 * sigmau2)

    # TERM 2: Derived from the likelihood of observing the state on trial n+1
    # given the state on trial n
    term2 = 0
    for n in range(N - 1):
        xnN_n = xnN[n].reshape(-1, 1)
        xnN_np1 = xnN[n + 1].reshape(-1, 1)

        term2 += (
            (xnN_np1.T @ Qinv @ xnN_np1)[0, 0]
            + np.trace(Qinv @ VnN[n + 1])
            - (xnN_np1.T @ QinvA @ xnN_n)[0, 0]
            - np.trace(QinvA @ Vnp1nN[n].T)
            - (xnN_np1.T @ Qinvb)[0, 0] * e[n]
            - (xnN_n.T @ AtQinv @ xnN_np1)[0, 0]
            - np.trace(AtQinv @ Vnp1nN[n])
            + (xnN_n.T @ AtQinvA @ xnN_n)[0, 0]
            + np.trace(AtQinvA @ VnN[n])
            + (xnN_n.T @ AtQinvb)[0, 0] * e[n]
            - e[n] * (btQinv @ xnN_np1)[0, 0]
            + e[n] * (btQinvA @ xnN_n)[0, 0]
            + e[n] * btQinvb * e[n]
        )
    term2 = -term2 / 2

    # TERM 3: Derived from the likelihood of observing the initial state
    x1 = x1.reshape(-1, 1)
    xnN_0 = xnN[0].reshape(-1, 1)
    V1inv = np.linalg.inv(V1)
    term3 = (
        (xnN_0.T @ V1inv @ xnN_0)[0, 0]
        + np.trace(V1inv @ VnN[0])
        - (xnN_0.T @ V1inv @ x1)[0, 0]
        - (x1.T @ V1inv @ xnN_0)[0, 0]
        + (x1.T @ V1inv @ x1)[0, 0]
    )
    term3 = -term3 / 2

    # TERM 4: Derived from the pre-exponential factors
    term4 = (
        -(1 / 2) * np.log(np.linalg.det(V1))
        - (N / 2) * np.log(sigmau2)
        - (3 / 2) * N * np.log(2 * np.pi)
    )

    # TERM 5: Derived from the pre-exponential factors of x(n+1) given x(n)
    term5 = -(N - 1) * np.log(np.linalg.det(Q)) / 2

    # Compute the likelihood from the sum of all terms
    likelihood = term1 + term2 + term3 + term4 + term5

    return likelihood
