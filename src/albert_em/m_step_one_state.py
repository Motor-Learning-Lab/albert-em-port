"""
M-Step for One-State EM Algorithm

Author: Python Port 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control

Summary: This module uses optimization to perform the M-step of the EM
   algorithm for the one-state model. The parameter set that maximizes
   the expected complete log-likelihood function in a constrained
   parameter space is identified.

Model:
    x_{n+1} ~ N(A x_n + B e_n, Q)
    y_n ~ N(x_n, R)
"""

import numpy as np
from scipy.optimize import minimize
from typing import List
from albert_em.expected_complete_log_likelihood_one_state import (
    expected_complete_log_likelihood_one_state,
)


def m_step_one_state(
    parameters_0: np.ndarray,
    xnN: List[float],
    VnN: List[float],
    Vnp1nN: List[float],
    y: np.ndarray,
    e: np.ndarray,
    search_space: np.ndarray,
) -> np.ndarray:
    """
    Performs the M-step of the EM algorithm for one-state model using
    constrained optimization.

    Args:
        parameters_0: Current estimate of the one-state model parameters
                     [A, B, Q, R, x0, P0]
        xnN: Smoothed Kalman state expectation (list of scalars)
        VnN: Smoothed Kalman state variance (list of scalars)
        Vnp1nN: Smoothed Kalman covariance of consecutive states (list of scalars)
        y: Motor output on each trial (N,)
        e: Error experienced by the subject on each trial (N,)
        search_space: Matrix containing upper and lower bounds (n_params x 2)

    Returns:
        parameters_final: Parameter set that maximizes the expected complete
                         log-likelihood function
    """

    # Create the objective function (negated likelihood)
    def likelihood_function(x):
        likelihood = expected_complete_log_likelihood_one_state(
            x, y, e, xnN, VnN, Vnp1nN
        )
        return -likelihood

    # Specify the lower and upper bounds for the search
    bounds = [
        (search_space[i, 0], search_space[i, 1]) for i in range(search_space.shape[0])
    ]

    # Use scipy.optimize.minimize to maximize the expected complete
    # log-likelihood function in a constrained parameter space
    result = minimize(
        likelihood_function,
        parameters_0,
        method="SLSQP",
        bounds=bounds,
        options={"disp": False},
    )

    parameters_final = result.x

    return parameters_final
