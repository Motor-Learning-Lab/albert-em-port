"""
Quick Reference Guide for Python EM Algorithm
==============================================

INSTALLATION
------------
pip install -r requirements.txt

BASIC USAGE
-----------
from generalized_expectation_maximization import generalized_expectation_maximization
import numpy as np

# Parameters: [aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12]
initial_params = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])

# Run EM
fitted_params, likelihoods = generalized_expectation_maximization(
    initial_params, y, r, EC, EC_value, c,
    search_space, constraints, num_iterations=100
)

PARAMETER DEFINITIONS
---------------------
Model Parameters (9 total):
1. aS       - Slow state retention factor (0-1)
2. aF       - Fast state retention factor (0-1)
3. bS       - Slow state error sensitivity (0-1)
4. bF       - Fast state error sensitivity (0-1)
5. xS1      - Initial slow state value
6. xF1      - Initial fast state value
7. sigmax2  - State update variance (>0)
8. sigmau2  - Motor output variance (>0)
9. sigma12  - Initial state variance (>0)

Input Data:
- y         - Motor output on each trial (N,)
- r         - Perturbation schedule (N,)
              0 = no perturbation
              NaN = error-clamp trial
              non-zero = perturbation value
- EC        - Error-clamp indicator (N,)
              0 = normal trial
              1 = error-clamp trial
- EC_value  - Clamped error value (N,)
              NaN = not error-clamp
              numeric = clamped error
- c         - State weighting (2,)
              Usually [1, 1] for equal weighting

Constraints:
- search_space  - Parameter bounds (9x2 array)
                  Each row: [min, max] for each parameter
- constraints   - Two-element array [deltaA, deltaB]
                  deltaA: aS >= aF + deltaA
                  deltaB: bF >= bS + deltaB

TYPICAL PARAMETER RANGES
-------------------------
aS:       0.95 - 0.995   (slow retention)
aF:       0.5 - 0.8      (fast retention)
bS:       0.01 - 0.1     (slow learning rate)
bF:       0.1 - 0.4      (fast learning rate)
xS1/xF1:  0              (initial states)
sigmax2:  0.1 - 5        (state noise)
sigmau2:  0.5 - 3        (motor noise)
sigma12:  0.1 - 10       (initial variance)

STANDARD SEARCH SPACE
---------------------
search_space = np.array([
    [0, 1.1],           # aS bounds
    [0, 1.1],           # aF bounds
    [0, 1],             # bS bounds
    [0, 1],             # bF bounds
    [-30, 30],          # xS1 bounds
    [-30, 30],          # xF1 bounds
    [0.0000001, 10],    # sigmax2 bounds
    [0.0000001, 10],    # sigmau2 bounds
    [0.0000001, 10]     # sigma12 bounds
])

constraints = np.array([0.001, 0.001])

EXAMPLE: SIMPLE ADAPTATION PARADIGM
------------------------------------
# 20 baseline, 50 adaptation, 20 error-clamp, 30 washout
r = np.concatenate([
    np.zeros(20),           # baseline
    30 * np.ones(50),       # adaptation with 30° perturbation
    np.full(20, np.nan),    # error-clamp trials
    np.zeros(30)            # washout
])

EC = np.concatenate([
    np.zeros(70),           # normal trials
    np.ones(20),            # error-clamp trials
    np.zeros(30)            # normal trials
])

EC_value = np.concatenate([
    np.full(70, np.nan),    # not clamped
    np.zeros(20),           # clamped at 0°
    np.full(30, np.nan)     # not clamped
])

SIMULATION
----------
from two_state_simulation import two_state_simulation_with_noise

# Simulate behavior
y, xS, xF = two_state_simulation_with_noise(
    parameters, r, EC, EC_value, c
)

CHECKING CONVERGENCE
--------------------
# Plot likelihood over iterations
import matplotlib.pyplot as plt

plt.figure()
plt.plot(likelihoods)
plt.xlabel('EM Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Convergence')
plt.show()

# Check parameter changes
# If parameters stabilize, EM has converged

COMMON ISSUES
-------------
1. Likelihood decreases
   → Bug in implementation or numerical instability
   → Check parameter bounds

2. Parameters hit bounds
   → Widen search space
   → Check initial conditions

3. Slow convergence
   → Increase num_iterations
   → Try different initial conditions
   → Check data quality

4. Poor fit
   → Model may not describe data
   → Try different initial parameters
   → Check for outliers in data

TESTING
-------
# Quick test (10 iterations)
python test_em.py

# Full tutorial (100 iterations with plots)
python tutorial.py

VISUALIZATION
-------------
from two_state_simulation import two_state_simulation_without_noise
import matplotlib.pyplot as plt

# Get model predictions
y_pred, xS_pred, xF_pred = two_state_simulation_without_noise(
    fitted_params, r, EC, EC_value, c
)

# Plot
fig, axes = plt.subplots(3, 1)
axes[0].plot(y, 'k', label='Data')
axes[0].plot(y_pred, 'r--', label='Model')
axes[0].legend()
axes[1].plot(xS_pred, label='Slow')
axes[2].plot(xF_pred, label='Fast')
plt.show()

MODULE OVERVIEW
---------------
generalized_expectation_maximization.py  - Main EM algorithm
kalman_smoother.py                       - E-step (state estimation)
m_step.py                                - M-step (parameter update)
expected_complete_log_likelihood.py      - Objective for M-step
incomplete_log_likelihood.py             - Convergence monitoring
two_state_simulation.py                  - Behavior simulation

GETTING HELP
------------
See python/README.md for detailed documentation
See tutorial.py for complete working example
See PYTHON_PORT_SUMMARY.md for technical details
"""

if __name__ == "__main__":
    print(__doc__)
