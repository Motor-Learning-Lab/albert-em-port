"""
Two-State Model Simulation Functions

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: Functions to simulate behavior according to a two-state model
   of learning with and without noise.
"""

import numpy as np
from typing import Tuple


def two_state_simulation_with_noise(
    parameters: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates behavior according to a two-state model with noise.

    Args:
        parameters: Two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
        EC_value: Array indicating the value of the clamped error (N,)
        c: Model parameter that is assumed invariant (2,)

    Returns:
        y: Motor output on each trial
        xS: Slow state on each trial
        xF: Fast state on each trial
    """

    # Extract parameters
    aS = parameters[0]
    aF = parameters[1]
    bS = parameters[2]
    bF = parameters[3]
    xS1 = parameters[4]
    xF1 = parameters[5]
    sigmax2 = parameters[6]
    sigmau2 = parameters[7]
    sigma12 = parameters[8]

    # Number of trials
    N = len(r)

    # Initialize state arrays
    xS = np.zeros(N)
    xF = np.zeros(N)
    y = np.zeros(N)
    e = np.zeros(N)

    # Set initial states with noise
    xS[0] = xS1 + np.sqrt(sigma12) * np.random.randn()
    xF[0] = xF1 + np.sqrt(sigma12) * np.random.randn()

    # Generate motor output for first trial
    y[0] = c[0] * xS[0] + c[1] * xF[0] + np.sqrt(sigmau2) * np.random.randn()

    # Compute error for first trial
    if EC[0] == 0:
        e[0] = r[0] - y[0]
    else:
        e[0] = EC_value[0]

    # Simulate remaining trials
    for n in range(1, N):
        # Update states with process noise
        xS[n] = aS * xS[n - 1] + bS * e[n - 1] + np.sqrt(sigmax2) * np.random.randn()
        xF[n] = aF * xF[n - 1] + bF * e[n - 1] + np.sqrt(sigmax2) * np.random.randn()

        # Generate motor output with measurement noise
        y[n] = c[0] * xS[n] + c[1] * xF[n] + np.sqrt(sigmau2) * np.random.randn()

        # Compute error
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    return y, xS, xF


def two_state_simulation_without_noise(
    parameters: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates behavior according to a two-state model without noise.

    Args:
        parameters: Two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
        EC_value: Array indicating the value of the clamped error (N,)
        c: Model parameter that is assumed invariant (2,)

    Returns:
        y: Motor output on each trial
        xS: Slow state on each trial
        xF: Fast state on each trial
    """

    # Extract parameters
    aS = parameters[0]
    aF = parameters[1]
    bS = parameters[2]
    bF = parameters[3]
    xS1 = parameters[4]
    xF1 = parameters[5]

    # Number of trials
    N = len(r)

    # Initialize state arrays
    xS = np.zeros(N)
    xF = np.zeros(N)
    y = np.zeros(N)
    e = np.zeros(N)

    # Set initial states
    xS[0] = xS1
    xF[0] = xF1

    # Generate motor output for first trial
    y[0] = c[0] * xS[0] + c[1] * xF[0]

    # Compute error for first trial
    if EC[0] == 0:
        e[0] = r[0] - y[0]
    else:
        e[0] = EC_value[0]

    # Simulate remaining trials
    for n in range(1, N):
        # Update states without noise
        xS[n] = aS * xS[n - 1] + bS * e[n - 1]
        xF[n] = aF * xF[n - 1] + bF * e[n - 1]

        # Generate motor output without noise
        y[n] = c[0] * xS[n] + c[1] * xF[n]

        # Compute error
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    return y, xS, xF


def one_state_simulation_with_noise(
    parameters: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates behavior according to a one-state model with noise.

    Model:
        x_{n+1} ~ N(A x_n + B e_n, Q)
        y_n ~ N(x_n, R)

    Args:
        parameters: One-state model parameters [A, B, Q, R, x0, P0]
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
        EC_value: Array indicating the value of the clamped error (N,)

    Returns:
        y: Motor output on each trial
        x: State on each trial
    """

    # Extract parameters
    A = parameters[0]
    B = parameters[1]
    Q = parameters[2]
    R = parameters[3]
    x0 = parameters[4]
    P0 = parameters[5]

    # Number of trials
    N = len(r)

    # Initialize state arrays
    x = np.zeros(N)
    y = np.zeros(N)
    e = np.zeros(N)

    # Set initial state with noise
    x[0] = x0 + np.sqrt(P0) * np.random.randn()

    # Generate motor output for first trial
    y[0] = x[0] + np.sqrt(R) * np.random.randn()

    # Compute error for first trial
    if EC[0] == 0:
        e[0] = r[0] - y[0]
    else:
        e[0] = EC_value[0]

    # Simulate remaining trials
    for n in range(1, N):
        # Update state with process noise
        x[n] = A * x[n - 1] + B * e[n - 1] + np.sqrt(Q) * np.random.randn()

        # Generate motor output with measurement noise
        y[n] = x[n] + np.sqrt(R) * np.random.randn()

        # Compute error
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    return y, x


def one_state_simulation_without_noise(
    parameters: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates behavior according to a one-state model without noise.

    Model:
        x_{n+1} = A x_n + B e_n
        y_n = x_n

    Args:
        parameters: One-state model parameters [A, B, Q, R, x0, P0]
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
        EC_value: Array indicating the value of the clamped error (N,)

    Returns:
        y: Motor output on each trial
        x: State on each trial
    """

    # Extract parameters
    A = parameters[0]
    B = parameters[1]
    x0 = parameters[4]

    # Number of trials
    N = len(r)

    # Initialize state arrays
    x = np.zeros(N)
    y = np.zeros(N)
    e = np.zeros(N)

    # Set initial state
    x[0] = x0

    # Generate motor output for first trial
    y[0] = x[0]

    # Compute error for first trial
    if EC[0] == 0:
        e[0] = r[0] - y[0]
    else:
        e[0] = EC_value[0]

    # Simulate remaining trials
    for n in range(1, N):
        # Update state without noise
        x[n] = A * x[n - 1] + B * e[n - 1]

        # Generate motor output without noise
        y[n] = x[n]

        # Compute error
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    return y, x
