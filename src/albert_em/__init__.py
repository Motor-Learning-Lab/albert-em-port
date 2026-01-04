"""
Python port of Albert and Shadmehr's EM algorithm for two-state model fitting.

This package provides a complete implementation of the Expectation-Maximization
algorithm for fitting a two-state model of sensorimotor adaptation.

Main components:
- generalized_expectation_maximization: Main EM algorithm coordinator
- kalman_smoother: E-step implementation using Kalman smoothing
- m_step: M-step implementation using constrained optimization
- expected_complete_log_likelihood: Likelihood function for M-step (Numba-optimized)
- incomplete_log_likelihood: Likelihood function for monitoring convergence
- two_state_simulation: Functions for simulating behavior

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
"""

from albert_em.em import (
    generalized_expectation_maximization,
    fit_two_state,
    fit_one_state,
)
from albert_em.kalman_smoother import kalman_smoother
from albert_em.m_step import m_step
from albert_em.expected_complete_log_likelihood import (
    expected_complete_log_likelihood,
    USING_NUMBA,
)
from albert_em.incomplete_log_likelihood import incomplete_log_likelihood
from albert_em.simulation import (
    two_state_simulation_with_noise,
    two_state_simulation_without_noise,
    one_state_simulation_with_noise,
    one_state_simulation_without_noise,
)
from albert_em.kalman_smoother_one_state import kalman_smoother_one_state
from albert_em.m_step_one_state import m_step_one_state
from albert_em.expected_complete_log_likelihood_one_state import (
    expected_complete_log_likelihood_one_state,
)
from albert_em.incomplete_log_likelihood_one_state import (
    incomplete_log_likelihood_one_state,
)
from albert_em.params_one_state import pack_one_state_params, unpack_one_state_params

__version__ = "1.0.0"
__author__ = "Scott Albert (Original), Python Port 2025"

__all__ = [
    "generalized_expectation_maximization",
    "fit_two_state",
    "fit_one_state",
    "kalman_smoother",
    "m_step",
    "expected_complete_log_likelihood",
    "USING_NUMBA",
    "incomplete_log_likelihood",
    "two_state_simulation_with_noise",
    "two_state_simulation_without_noise",
    "one_state_simulation_with_noise",
    "one_state_simulation_without_noise",
    "kalman_smoother_one_state",
    "m_step_one_state",
    "expected_complete_log_likelihood_one_state",
    "incomplete_log_likelihood_one_state",
    "pack_one_state_params",
    "unpack_one_state_params",
]
