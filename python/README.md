# Python Port of Albert and Shadmehr's EM Algorithm

This is a Python port of the MATLAB implementation of the Expectation-Maximization (EM) algorithm for fitting a two-state model of sensorimotor adaptation.

## Original Work

**Author:** Scott Albert  
**Email:** salbert8@jhu.edu  
**Institution:** Johns Hopkins University  
**Lab:** Laboratory for Computational Motor Control  
**Advisor:** Reza Shadmehr  
**Date:** July 25, 2017  

## Overview

This package implements an EM algorithm for parameter estimation in a two-state model of motor learning. The model assumes that motor adaptation is governed by two hidden states (fast and slow) that evolve according to state-space dynamics. The algorithm uses:

- **E-step:** Kalman smoothing to estimate hidden states
- **M-step:** Constrained optimization to update model parameters

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the tutorial script to see a complete example:

```bash
python tutorial.py
```

This will:
1. Simulate behavior according to a two-state model
2. Fit the model using the EM algorithm
3. Display the results and create visualization plots

### Basic Example

```python
import numpy as np
from generalized_expectation_maximization import generalized_expectation_maximization

# Define experimental paradigm
r = np.concatenate([np.zeros(20), 30 * np.ones(50), 
                   np.full(20, np.nan), np.zeros(30)])
EC = np.concatenate([np.zeros(70), np.ones(20), np.zeros(30)])
EC_value = np.concatenate([np.full(70, np.nan), np.zeros(20), 
                          np.full(30, np.nan)])
c = np.array([1.0, 1.0])

# Simulate or load behavioral data
y = # your motor output data

# Set up EM parameters
initial_params = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])
search_space = np.array([
    [0, 1.1], [0, 1.1], [0, 1], [0, 1],
    [-30, 30], [-30, 30],
    [0.0000001, 10], [0.0000001, 10], [0.0000001, 10]
])
constraints = np.array([0.001, 0.001])

# Run EM algorithm
parameters, likelihoods = generalized_expectation_maximization(
    initial_params, y, r, EC, EC_value, c, 
    search_space, constraints, num_iterations=100
)
```

## Model Parameters

The two-state model has 9 parameters:

1. **aS** - Slow state retention factor (0 to 1)
2. **aF** - Fast state retention factor (0 to 1)
3. **bS** - Slow state error sensitivity (0 to 1)
4. **bF** - Fast state error sensitivity (0 to 1)
5. **xS1** - Initial slow state
6. **xF1** - Initial fast state
7. **sigmax2** - State update variance
8. **sigmau2** - Motor output variance
9. **sigma12** - Initial state variance

## Module Structure

- `generalized_expectation_maximization.py` - Main EM algorithm coordinator
- `kalman_smoother.py` - E-step: Kalman smoothing implementation
- `m_step.py` - M-step: Constrained optimization
- `expected_complete_log_likelihood.py` - Likelihood for M-step
- `incomplete_log_likelihood.py` - Likelihood for convergence monitoring
- `two_state_simulation.py` - Behavior simulation functions
- `tutorial.py` - Complete usage example

## Key Differences from MATLAB Version

1. **Lists instead of cell arrays:** Python uses lists to store variable-length sequences
2. **NumPy arrays:** All matrices and vectors use NumPy arrays
3. **SciPy optimization:** Uses `scipy.optimize.minimize` instead of MATLAB's `fmincon`
4. **Zero-based indexing:** Python uses 0-based indexing (MATLAB uses 1-based)
5. **No MEX dependency:** Pure Python implementation (MEX optimization not ported)

## Performance Notes

The Python implementation provides similar results to the MATLAB version but may have different performance characteristics:

- The optimization in M-step uses SciPy's SLSQP method
- For very large datasets, consider vectorizing operations further
- The MEX-optimized likelihood function from MATLAB is not ported (pure Python only)

## References

For more information about the algorithm and its applications, please refer to:

Albert, S. T., & Shadmehr, R. (2016). The neural feedback response to error as a teaching signal for the motor learning system. *Journal of Neuroscience*, 36(17), 4832-4845.

## License

Please refer to the LICENSE file in the repository root.

## Citation

If you use this code, please cite the original work by Scott Albert and Reza Shadmehr.
