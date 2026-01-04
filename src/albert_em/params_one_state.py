"""
Parameter Packing/Unpacking for One-State Model

Author: Python Port 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control

Summary: This module provides explicit parameter packing and unpacking
   utilities for the one-state EM model.

Model:
    x_{n+1} ~ N(A x_n + B y_n, Q)
    y_n ~ N(x_n, R)

Parameters:
    A: State transition coefficient (scalar)
    B: Input coefficient (scalar)
    Q: Process noise variance (scalar)
    R: Observation noise variance (scalar)
    x0: Initial state mean (scalar)
    P0: Initial state variance (scalar)
"""

import numpy as np
from typing import Tuple


def pack_one_state_params(
    A: float, B: float, Q: float, R: float, x0: float, P0: float
) -> np.ndarray:
    """
    Pack one-state model parameters into a single array.

    Args:
        A: State transition coefficient
        B: Input coefficient
        Q: Process noise variance
        R: Observation noise variance
        x0: Initial state mean
        P0: Initial state variance

    Returns:
        theta: Packed parameter array [A, B, Q, R, x0, P0]
    """
    return np.array([A, B, Q, R, x0, P0])


def unpack_one_state_params(
    theta: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """
    Unpack one-state model parameters from a single array.

    Args:
        theta: Packed parameter array [A, B, Q, R, x0, P0]

    Returns:
        A: State transition coefficient
        B: Input coefficient
        Q: Process noise variance
        R: Observation noise variance
        x0: Initial state mean
        P0: Initial state variance
    """
    A = theta[0]
    B = theta[1]
    Q = theta[2]
    R = theta[3]
    x0 = theta[4]
    P0 = theta[5]

    return A, B, Q, R, x0, P0
