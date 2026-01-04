"""
Generalized Expectation-Maximization Algorithm

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
Advisor: Reza Shadmehr

Summary: This module coordinates the EM algorithm. It successively
   calls the Kalman Smoother to specify the E-Step, and then performs
   the M-step to update the parameter set.
"""

import numpy as np
import warnings
from typing import Tuple, Callable, List
from albert_em.kalman_smoother import kalman_smoother
from albert_em.m_step import m_step
from albert_em.incomplete_log_likelihood import incomplete_log_likelihood
from albert_em.kalman_smoother_one_state import kalman_smoother_one_state
from albert_em.m_step_one_state import m_step_one_state
from albert_em.incomplete_log_likelihood_one_state import (
    incomplete_log_likelihood_one_state,
)


def _run_em(
    parameters: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    search_space: np.ndarray,
    num_iterations: int,
    smoother: Callable,
    m_step_optimizer: Callable,
    ill_function: Callable,
    m_step_extra_args: tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared EM algorithm driver for both one-state and two-state models.

    This is the internal EM loop. Both fit_one_state and fit_two_state
    call this function with model-specific callbacks.

    Args:
        parameters: Initial parameter guess
        y: Motor output on each trial (N,)
        e: Error signal on each trial (N,)
        search_space: Parameter bounds (n_params x 2)
        num_iterations: Number of EM iterations
        smoother: Callable that returns (xnN, VnN, Vnp1nN)
        m_step_optimizer: Callable that returns updated parameters
        ill_function: Callable that returns incomplete log-likelihood value
        m_step_extra_args: Extra arguments to pass to m_step_optimizer

    Returns:
        parameters: Final parameters
        likelihoods: Incomplete log-likelihood at each iteration
    """
    likelihoods = np.zeros(num_iterations)

    for n in range(num_iterations):
        # E-step: Kalman smoothing
        xnN, VnN, Vnp1nN = smoother(parameters, y, e)

        # M-step: Constrained optimization
        parameters = m_step_optimizer(
            parameters, xnN, VnN, Vnp1nN, y, e, search_space, *m_step_extra_args
        )

        # Compute incomplete log-likelihood
        likelihoods[n] = ill_function(y, e, parameters)

        # Check monotonicity
        if (n > 0) and (likelihoods[n] < likelihoods[n - 1]):
            warnings.warn(
                "The expected complete log-likelihood function has stopped increasing"
            )

    return parameters, likelihoods


def generalized_expectation_maximization(
    parameters: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    c: np.ndarray,
    search_space: np.ndarray,
    constraints: np.ndarray,
    num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinates the EM algorithm for fitting a two-state model.

    Args:
        parameters: Initial guess of the two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        y: Motor output on each trial (N,)
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
            Non-zero indicates error-clamp trial
        EC_value: Array indicating the value of the clamped error (N,)
        c: Model parameter that is assumed invariant (2,)
        search_space: Matrix containing upper and lower bounds (n_params x 2)
        constraints: Array specifying inequality constraints [deltaA, deltaB]
        num_iterations: Number of EM iterations

    Returns:
        parameters: Final parameters obtained at the conclusion of EM
        likelihoods: Array containing the value of the incomplete
                    log-likelihood function on each iteration
    """

    # Compute the error on each trial
    N = len(y)
    e = np.zeros(N)

    for n in range(N):
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    # Wrapper for two-state m_step that includes constraints
    def two_state_m_step(param, xnN, VnN, Vnp1nN, y, e, search_space, constraints):
        return m_step(param, xnN, VnN, Vnp1nN, y, e, c, search_space, constraints)

    # Call shared EM driver with two-state functions
    return _run_em(
        parameters,
        y,
        e,
        search_space,
        num_iterations,
        smoother=lambda param, y, e: kalman_smoother(param, y, e, c),
        m_step_optimizer=two_state_m_step,
        ill_function=lambda y, e, param: incomplete_log_likelihood(y, e, c, param),
        m_step_extra_args=(constraints,),
    )


def fit_two_state(
    parameters: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    c: np.ndarray,
    search_space: np.ndarray,
    constraints: np.ndarray,
    num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper for fitting a two-state model.

    This function selects the two-state model-specific functions and
    calls the shared EM driver.

    Args:
        parameters: Initial guess of the two state model parameters
                   [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
        y: Motor output on each trial (N,)
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
            Non-zero indicates error-clamp trial
        EC_value: Array indicating the value of the clamped error (N,)
        c: Model parameter that is assumed invariant (2,)
        search_space: Matrix containing upper and lower bounds (n_params x 2)
        constraints: Array specifying inequality constraints [deltaA, deltaB]
        num_iterations: Number of EM iterations

    Returns:
        parameters: Final parameters obtained at the conclusion of EM
        likelihoods: Array containing the value of the incomplete
                    log-likelihood function on each iteration
    """
    return generalized_expectation_maximization(
        parameters, y, r, EC, EC_value, c, search_space, constraints, num_iterations
    )


def fit_one_state(
    parameters: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    EC: np.ndarray,
    EC_value: np.ndarray,
    search_space: np.ndarray,
    num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper for fitting a one-state model.

    This function selects the one-state model-specific functions and
    calls the shared EM driver.

    Args:
        parameters: Initial guess of the one-state model parameters
                   [A, B, Q, R, x0, P0]
        y: Motor output on each trial (N,)
        r: Perturbation on each trial (N,)
        EC: Array indicating if a trial is an error-clamp trial (N,)
            Non-zero indicates error-clamp trial
        EC_value: Array indicating the value of the clamped error (N,)
        search_space: Matrix containing upper and lower bounds (n_params x 2)
        num_iterations: Number of EM iterations

    Returns:
        parameters: Final parameters obtained at the conclusion of EM
        likelihoods: Array containing the value of the incomplete
                    log-likelihood function on each iteration
    """
    # Compute the error on each trial
    N = len(y)
    e = np.zeros(N)

    for n in range(N):
        if EC[n] == 0:
            e[n] = r[n] - y[n]
        else:
            e[n] = EC_value[n]

    # Call shared EM driver with one-state functions
    return _run_em(
        parameters,
        y,
        e,
        search_space,
        num_iterations,
        smoother=kalman_smoother_one_state,
        m_step_optimizer=m_step_one_state,
        ill_function=incomplete_log_likelihood_one_state,
    )
