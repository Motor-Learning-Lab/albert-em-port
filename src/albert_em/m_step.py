"""
M-Step for EM Algorithm

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module uses optimization to perform the M-step of the EM
   algorithm. The parameter set that maximizes the expected complete
   log-likelihood function in a constrained parameter space is identified.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List
from albert_em.expected_complete_log_likelihood import expected_complete_log_likelihood


def m_step(
    parameters_0: np.ndarray,
    xnN: List[np.ndarray],
    VnN: List[np.ndarray],
    Vnp1nN: List[np.ndarray],
    y: np.ndarray,
    e: np.ndarray,
    c: np.ndarray,
    search_space: np.ndarray,
    constraints: np.ndarray,
) -> np.ndarray:
    """
    Performs the M-step of the EM algorithm using constrained optimization.

    Args:
        parameters_0: Current estimate of the two state model parameters
        xnN: Smoothed Kalman state expectation
        VnN: Smoothed Kalman state variance
        Vnp1nN: Smoothed Kalman covariance of consecutive states
        y: Motor output on each trial
        e: Error experienced by the subject on each trial
        c: Model parameter that is assumed invariant
        search_space: Matrix containing upper and lower bounds (n_params x 2)
        constraints: Array specifying inequality constraints [deltaA, deltaB]

    Returns:
        parameters_final: Parameter set that maximizes the expected complete
                         log-likelihood function
    """

    # Create the objective function (negated likelihood)
    def likelihood_function(x):
        likelihood = expected_complete_log_likelihood(x, y, e, c, xnN, VnN, Vnp1nN)
        return -likelihood

    # Specify the lower and upper bounds for the search
    bounds = [
        (search_space[i, 0], search_space[i, 1]) for i in range(search_space.shape[0])
    ]

    # Specify linear inequality constraints
    # Constraints are of the form: A_con @ x <= b_con
    # Constraint 1: aS >= aF + deltaA  =>  -aS + aF <= -deltaA
    # Constraint 2: bF >= bS + deltaB  =>  bS - bF <= -deltaB
    #                aS    aF   bS   bF   xS1   xF1   sigmax2    sigmau2  sigma12
    A_con = np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0, 0, 0]])
    b_con = np.array(
        [-constraints[0], -constraints[1]]  # -aS + aF <= -deltaA
    )  # bS - bF <= -deltaB

    # Define linear inequality constraints for scipy.optimize.minimize
    linear_constraints = {
        "type": "ineq",
        "fun": lambda x: b_con - A_con @ x,
        "jac": lambda x: -A_con,
    }

    # Use scipy.optimize.minimize to maximize the expected complete
    # log-likelihood function in a constrained parameter space
    result = minimize(
        likelihood_function,
        parameters_0,
        method="SLSQP",
        bounds=bounds,
        constraints=linear_constraints,
        options={"disp": False},
    )

    parameters_final = result.x

    return parameters_final
