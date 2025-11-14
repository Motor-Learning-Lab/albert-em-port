"""
Python port of Albert and Shadmehr's EM algorithm for two-state model fitting.

This package provides a complete implementation of the Expectation-Maximization
algorithm for fitting a two-state model of sensorimotor adaptation.

Main components:
- generalized_expectation_maximization: Main EM algorithm coordinator
- kalman_smoother: E-step implementation using Kalman smoothing
- m_step: M-step implementation using constrained optimization
- expected_complete_log_likelihood: Likelihood function for M-step
- incomplete_log_likelihood: Likelihood function for monitoring convergence
- two_state_simulation: Functions for simulating behavior

Author: Scott Albert (Original MATLAB)
Python Port: 2025
Institution: Johns Hopkins University
Lab: Laboratory for Computational Motor Control
"""

from .generalized_expectation_maximization import generalized_expectation_maximization
from .kalman_smoother import kalman_smoother
from .m_step import m_step
from .expected_complete_log_likelihood import expected_complete_log_likelihood
from .incomplete_log_likelihood import incomplete_log_likelihood
from .two_state_simulation import (
    two_state_simulation_with_noise,
    two_state_simulation_without_noise,
)

__version__ = "1.0.0"
__author__ = "Scott Albert (Original), Python Port 2025"

__all__ = [
    "generalized_expectation_maximization",
    "kalman_smoother",
    "m_step",
    "expected_complete_log_likelihood",
    "incomplete_log_likelihood",
    "two_state_simulation_with_noise",
    "two_state_simulation_without_noise",
]
