# albert-em

Python package for fitting two-state models of sensorimotor adaptation using the Expectation-Maximization (EM) algorithm.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This package provides a complete, high-performance Python implementation of the Expectation-Maximization algorithm for fitting two-state models of motor learning. The implementation uses **Numba JIT compilation** to achieve C-like performance without requiring manual compilation of C++/MEX code.

**Original Work:**
- Author: Scott Albert
- Institution: Johns Hopkins University
- Lab: Laboratory for Computational Motor Control
- Advisor: Reza Shadmehr
- Date: July 25, 2017

## Features

✨ **High Performance**: Numba-optimized likelihood function provides near-C performance  
📦 **Easy Installation**: Install as a package via pip or pixi  
🔬 **Scientific Computing**: Built on NumPy, SciPy, and Numba  
📊 **Visualization**: Optional matplotlib integration for plotting  
🧪 **Well-Tested**: Includes benchmarks and convergence tests  
📖 **Comprehensive Docs**: Type hints and detailed documentation  

## Installation

### Using pixi (Recommended)

```bash
# Clone the repository
git clone https://github.com/Motor-Learning-Lab/albert-em-port.git
cd albert-em-port

# Install with pixi
pixi install

# Run example
pixi run example

# Run tests
pixi run test
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/Motor-Learning-Lab/albert-em-port.git
cd albert-em-port

# Install the package
pip install -e .

# Or with visualization support
pip install -e ".[viz]"

# Or for development
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from albert_em import generalized_expectation_maximization

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
parameters, likelihoods = generalized_expectation_maximization(
    initial_params, y, r, EC, EC_value, c, 
    search_space, constraints, num_iterations=100
)
```

## Repository Structure

```
albert-em-port/
├── src/albert_em/              # Main package
│   ├── __init__.py
│   ├── em.py                   # Main EM coordinator
│   ├── kalman_smoother.py      # E-step
│   ├── m_step.py               # M-step
│   ├── expected_complete_log_likelihood.py  # Numba-optimized
│   ├── incomplete_log_likelihood.py
│   └── simulation.py
├── examples/
│   └── tutorial.py             # Complete demonstration
├── tests/
│   └── test_benchmark.py       # Performance benchmarks
├── Matlab/                     # Original MATLAB code
├── pixi.toml                   # Pixi configuration
├── pyproject.toml              # Python package configuration
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

## Examples

See `examples/tutorial.py` for a complete working example including:
- Data simulation
- Model fitting
- Visualization of results

Run with:
```bash
pixi run example
# or
python examples/tutorial.py
```

## Testing

```bash
# Run all tests
pixi run test

# Run benchmarks
pixi run benchmark

# Or with pytest directly
pytest tests/
```

## Documentation

- **Examples**: See `examples/` directory
- **API Documentation**: See docstrings in source code
- **MATLAB Comparison**: See `Matlab/` directory for original implementation

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
