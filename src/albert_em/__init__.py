"""
Python port of Albert and Shadmehr's EM algorithm for sensorimotor adaptation.

This package provides complete implementations of the Expectation-Maximization
algorithm for fitting both one-state and two-state models of sensorimotor
adaptation.

Supported Models:
- One-State Model: Single timescale learning (x_{n+1} ~ N(A*x_n + B*e_n, Q))
- Two-State Model: Fast/slow dual timescale learning (independent dynamics)

Main Components (One-State):
- fit_one_state: Main entry point for one-state model fitting
- kalman_smoother_one_state: E-step for one-state model
- m_step_one_state: M-step optimizer for one-state model
- expected_complete_log_likelihood_one_state: Objective function for M-step
- incomplete_log_likelihood_one_state: Convergence monitoring
- pack_one_state_params / unpack_one_state_params: Parameter utilities

Main Components (Two-State):
- fit_two_state: Entry point for two-state model fitting
- generalized_expectation_maximization: Two-state EM coordinator
- kalman_smoother: E-step for two-state model
- m_step: M-step optimizer for two-state model
- expected_complete_log_likelihood: Objective function (Numba-optimized)
- incomplete_log_likelihood: Convergence monitoring

Simulation & Utilities:
- one_state_simulation_with_noise / without_noise: Generate synthetic data
- two_state_simulation_with_noise / without_noise: Generate synthetic data

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
