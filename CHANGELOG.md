# Changelog - Python Port

## Version 1.1.0 (2025-01-05)

### Added - One-State Model Support

#### Core One-State Components
- `kalman_smoother_one_state.py` - 1D Kalman filter + RTS smoother for one-state E-step
- `expected_complete_log_likelihood_one_state.py` - One-state M-step objective function
- `incomplete_log_likelihood_one_state.py` - One-state convergence monitoring
- `m_step_one_state.py` - One-state parameter optimization
- `params_one_state.py` - Parameter packing/unpacking utilities
- `fit_one_state()` - Main entry point for one-state EM fitting

#### Simulation
- `one_state_simulation_with_noise()` - Generate synthetic one-state data
- `one_state_simulation_without_noise()` - Deterministic one-state simulation

#### Testing
- `test_one_state.py` - Comprehensive test suite for one-state model
  - Parameter recovery test (convergence to true parameters)
  - EM monotonicity test (likelihood non-decreasing)
  - Integration tests (error-clamp handling, workflows)

#### Documentation
- `ONE_STATE_IMPLEMENTATION.md` - Detailed implementation guide
- `ONE_STATE_API_REFERENCE.md` - API documentation with usage examples
- Updated `README.md` with one-state quickstart examples
- Updated module docstring to reflect dual-model support

### Changed - Refactoring

#### Architecture Improvements
- Extracted shared `_run_em()` driver to eliminate EM loop duplication
- Both `fit_two_state()` and `fit_one_state()` now call shared driver via callbacks
- Improved consistency between one-state and two-state implementations

#### Documentation
- `__init__.py` module docstring updated to document both models
- `CHANGELOG.md` updated with one-state additions
- Fixed model equation in `params_one_state.py` (B*e_n, not B*y_n)

### Backward Compatibility

✅ **Full backward compatibility maintained:**
- `fit_two_state()` preserves original API
- `generalized_expectation_maximization()` behavior unchanged
- All existing two-state tests passing
- No breaking changes to public API

---

## Version 1.0.0 (2025)

### Added - Python Port

#### Core Algorithm Implementation
- `kalman_smoother.py` - Forward Kalman filter and backward smoother for E-step
- `expected_complete_log_likelihood.py` - Objective function for M-step optimization
- `incomplete_log_likelihood.py` - Convergence monitoring function
- `m_step.py` - Constrained parameter optimization using SciPy
- `generalized_expectation_maximization()` - Main EM algorithm coordinator

#### Simulation and Testing
- `two_state_simulation.py` - Behavior simulation with and without noise
- `tutorial.py` - Complete demonstration script with visualization
- `test_smoke.py` - Fast smoke tests and fallback mode verification

#### Documentation and Package
- `__init__.py` - Package initialization with exports
- `README.md` - Comprehensive Python documentation
- `pyproject.toml` - Modern Python project configuration
- `pixi.toml` - Reproducible environment specification

### Technical Details

#### Core Libraries Used
- NumPy (≥1.20.0) - Array operations and linear algebra
- SciPy (≥1.7.0) - Constrained optimization (SLSQP method)
- Numba (≥0.57.0) - JIT compilation with pure Python fallback
- Matplotlib (≥3.3.0) - Visualization and plotting

#### Key Translation Decisions
- MATLAB cell arrays → Python lists of NumPy arrays
- MATLAB fmincon → SciPy minimize with SLSQP
- 1-based indexing → 0-based indexing throughout
- Type hints added for better IDE support
- Comprehensive docstrings in NumPy style
- Numba JIT with automatic fallback for compatibility

#### Features Compared to MATLAB

**Maintained:**
- Complete algorithm fidelity
- All parameter constraints
- Convergence monitoring
- Error-clamp trial support
- Tutorial demonstrations

**Added:**
- Type hints for function signatures
- Package structure with `__init__.py`
- Numba JIT acceleration (with Python fallback)
- Comprehensive test suite

- Automated test script
- Modern Python idioms
- Enhanced documentation

**Not Ported:**
- MEX-compiled likelihood function (NumPy provides sufficient performance)
- Original `use_mex` parameter removed

#### Known Differences
- Different random number sequences (different RNG algorithms)
- Slightly different optimization paths (SLSQP vs interior-point)
- Results are statistically equivalent but not identical

### Validation

The Python port has been validated for:
- ✓ Correct imports and dependencies
- ✓ Function signature compatibility
- ✓ Monotonic likelihood increase
- ✓ End-to-end tutorial execution
- ✓ Reasonable parameter recovery

### Performance Notes

- Pure Python/NumPy implementation (no MEX required)
- Efficient for typical datasets (100-1000 trials)
- Optimization uses gradient-free SLSQP method
- All matrix operations fully vectorized

### Usage Example

```python
import numpy as np
from generalized_expectation_maximization import generalized_expectation_maximization

# Setup
parameters = np.array([0.95, 0.7, 0.05, 0.30, 0, 0, 2, 2, 5])
# ... define y, r, EC, EC_value, c, search_space, constraints

# Run EM
fitted_params, likelihoods = generalized_expectation_maximization(
    parameters, y, r, EC, EC_value, c, 
    search_space, constraints, num_iterations=100
)
```

### Future Work

Potential enhancements:
- [ ] Pytest-based unit test suite
- [ ] Jupyter notebook examples
- [ ] Parallel processing for multiple subjects
- [ ] Additional optimization algorithms
- [ ] Progress bar for long runs
- [ ] Data validation utilities
- [ ] Diagnostic plotting functions
- [ ] Pandas DataFrame support

### Credits

**Original MATLAB Implementation:**
- Author: Scott Albert
- Email: salbert8@jhu.edu
- Institution: Johns Hopkins University
- Lab: Laboratory for Computational Motor Control
- Advisor: Reza Shadmehr
- Date: July 25, 2017

**Python Port:**
- Date: 2025
- Based on MATLAB version 1.1

### References

Albert, S. T., & Shadmehr, R. (2016). The neural feedback response to error as a teaching signal for the motor learning system. *Journal of Neuroscience*, 36(17), 4832-4845.

### License

See LICENSE file in repository root.
