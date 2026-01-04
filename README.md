# albert-em

Python package for fitting sensorimotor adaptation models using the Expectation-Maximization (EM) algorithm.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This package provides high-performance Python implementations of the Expectation-Maximization algorithm for fitting both **one-state** and **two-state** models of sensorimotor motor learning. The implementation uses **Numba JIT compilation** to achieve C-like performance without requiring manual compilation.

**Supported Models:**
- **One-State Model**: Single timescale learning dynamics
- **Two-State Model**: Dual fast/slow timescale learning (original Albert & Shadmehr)

**Original Work:**

- Author: Scott Albert
- Institution: Johns Hopkins University
- Lab: Laboratory for Computational Motor Control
- Advisor: Reza Shadmehr
- Date: July 25, 2017

## Features

✨ **High Performance**: Numba-optimized likelihood functions with near-C performance  
📦 **Easy Installation**: Install via pip or pixi  
🎯 **Two Model Variants**: One-state and two-state models supported  
🔬 **Scientific Computing**: Built on NumPy, SciPy, and Numba  
📊 **Visualization**: Optional matplotlib integration  
🧪 **Well-Tested**: Comprehensive test suite including parameter recovery and monotonicity tests  
📖 **Comprehensive Docs**: Type hints, docstrings, and tutorials  

## Installation

### Using pixi (recommended)

```bash
# Clone the repository
git clone https://github.com/Motor-Learning-Lab/albert-em-port.git
cd albert-em-port

# Install with pixi
pixi install

# Run tests
pixi run test
```

### Using pip (from GitHub)

```bash
pip install "albert-em @ git+https://github.com/Motor-Learning-Lab/albert-em-port@main"
```

### Using pip (editable clone)

```bash
git clone https://github.com/Motor-Learning-Lab/albert-em-port.git
cd albert-em-port
pip install -e .
```

## Quick Start: Two-State Model

```python
import numpy as np
from albert_em import fit_two_state

# Define experimental paradigm
r = np.concatenate([np.zeros(20), 30*np.ones(50), 
                   np.full(20, np.nan), np.zeros(30)])
EC = np.concatenate([np.zeros(70), np.ones(20), np.zeros(30)])
EC_value = np.concatenate([np.full(70, np.nan), np.zeros(20), 
                          np.full(30, np.nan)])
c = np.array([1.0, 1.0])

# Your behavioral data
y = # ... motor output data

# Set up EM
initial_params = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])
search_space = np.array([
    [0, 1.1], [0, 1.1], [0, 1], [0, 1],
    [-30, 30], [-30, 30],
    [0.0000001, 10], [0.0000001, 10], [0.0000001, 10]
])
constraints = np.array([0.001, 0.001])

# Run EM algorithm
parameters, likelihoods = fit_two_state(
    initial_params, y, r, EC, EC_value, c, 
    search_space, constraints, num_iterations=100
)
```

## Quick Start: One-State Model

```python
import numpy as np
from albert_em import fit_one_state, pack_one_state_params

# Define experimental paradigm
r = np.concatenate([np.zeros(50), 15*np.ones(100), np.zeros(50)])
EC = np.zeros(200)
EC_value = np.zeros(200)

# Your behavioral data
y = # ... motor output data

# Set up one-state model parameters
params = pack_one_state_params(
    A=0.95,      # State transition (decay/retention)
    B=0.15,      # Error input gain
    Q=0.01,      # Process noise variance
    R=0.05,      # Observation noise variance
    x0=0.0,      # Initial state mean
    P0=0.1       # Initial state variance
)

# Define search space
search_space = np.array([
    [0.5, 1.0],      # A bounds
    [0.0, 0.5],      # B bounds
    [0.001, 0.5],    # Q bounds
    [0.001, 0.5],    # R bounds
    [-1.0, 1.0],     # x0 bounds
    [0.001, 1.0],    # P0 bounds
])

# Run EM algorithm
fitted_params, likelihoods = fit_one_state(
    params, y, r, EC, EC_value, search_space, num_iterations=50
)
```


## Repository Structure

```text
albert-em-port/
├── src/albert_em/              # Main package
│   ├── __init__.py
│   ├── em.py                   # Main EM coordinator
│   ├── kalman_smoother.py      # E-step
│   ├── m_step.py               # M-step
│   ├── expected_complete_log_likelihood.py  # Numba-optimized with fallback
│   ├── incomplete_log_likelihood.py
│   └── simulation.py
├── ipynb/
│   └── tutorial.ipynb          # End-to-end demo (keep this)
├── tests/
│   └── test_smoke.py           # Fast, clear smoke test
├── Matlab/                     # Original MATLAB code
├── pixi.toml                   # Pixi environment (dev + notebook)
├── pyproject.toml              # Python package metadata
└── README.md
```

## Performance

The Python implementation uses **Numba JIT compilation** to achieve near-C performance:

- ✅ **No manual compilation required** (unlike MEX)
- ✅ **Automatic optimization** on first run
- ✅ **Comparable performance** to C++/MEX
- ✅ **Pure Python** with scientific libraries

### Why Numba instead of C++/Rust?

1. **Ease of use**: No build toolchain required
2. **Performance**: JIT compilation provides 50-100x speedup over pure Python
3. **Maintainability**: Keep everything in Python
4. **Cross-platform**: Works on Windows, Linux, macOS without recompilation
5. **Scientific ecosystem**: Seamless NumPy integration

## Model Parameters

The two-state model has 9 parameters:

1. **aS** - Slow state retention factor (0-1)
2. **aF** - Fast state retention factor (0-1)
3. **bS** - Slow state error sensitivity (0-1)
4. **bF** - Fast state error sensitivity (0-1)
5. **xS1** - Initial slow state
6. **xF1** - Initial fast state
7. **sigmax2** - State update variance (>0)
8. **sigmau2** - Motor output variance (>0)
9. **sigma12** - Initial state variance (>0)

## Model Description

The two-state model assumes motor adaptation is governed by two hidden states:

- **Slow state**: High retention (aS ≈ 0.98), low learning rate (bS ≈ 0.1)
- **Fast state**: Low retention (aF ≈ 0.6), high learning rate (bF ≈ 0.3)

The EM algorithm iteratively:

1. **E-step**: Estimates hidden states using Kalman smoothing
2. **M-step**: Updates model parameters via constrained optimization (SLSQP)

## Example: notebook

Open the tutorial notebook for a complete, reproducible workflow:

```bash
pixi run notebook
```

## Testing

```bash
# Run all tests
pixi run test

# Or with pytest directly
pytest tests/
```

## Documentation

- **Notebook**: See `ipynb/tutorial.ipynb`
- **API**: Docstrings in source code
- **MATLAB reference**: `Matlab/` directory

## Citation

If you use this code, please cite:

Albert, S. T., & Shadmehr, R. (2016). The neural feedback response to error as a teaching signal for the motor learning system. *Journal of Neuroscience*, 36(17), 4832-4845.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- **Original MATLAB implementation**: Scott Albert (Johns Hopkins University)
- **Python port**: 2025
- **Lab**: Laboratory for Computational Motor Control
- **Advisor**: Reza Shadmehr
