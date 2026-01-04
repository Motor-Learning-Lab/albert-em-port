# One-State EM Model Implementation Summary

## Overview

Successfully implemented a one-state EM model while preserving the existing two-state API. The implementation follows strict architectural guidelines:

- ✅ **No EM driver duplication** - Single shared `_run_em()` driver with callbacks  
- ✅ **No public API changes** - `fit_two_state()` remains fully backward compatible
- ✅ **Model-specific duplication only** - Kalman smoother, likelihood, and M-step duplicated for one-state
- ✅ **Thin wrappers** - Both `fit_one_state()` and `fit_two_state()` are thin model-selector wrappers
- ✅ **Clean mechanical refactoring path** - Identical function signatures enable future generalization

## Implementation Architecture

### Single Shared EM Driver ✓

**File:** [src/albert_em/em.py](src/albert_em/em.py) - `_run_em()` function

A reusable EM loop that accepts model-specific callbacks:

```python
def _run_em(
    parameters, y, e, search_space, num_iterations,
    smoother,              # Callable: (params, y, e) → (xnN, VnN, Vnp1nN)
    m_step_optimizer,      # Callable: (params, ..., search_space, *extra) → params
    ill_function,          # Callable: (y, e, params) → float
    m_step_extra_args=()   # Extra args passed to m_step_optimizer
)
```

**Core Loop:**
```python
for iteration in range(num_iterations):
    # E-step: Kalman smoothing (model-specific)
    xnN, VnN, Vnp1nN = smoother(parameters, y, e)
    
    # M-step: Constrained optimization (model-specific)
    parameters = m_step_optimizer(parameters, xnN, VnN, Vnp1nN, y, e, search_space, *extra_args)
    
    # Compute incomplete log-likelihood (model-specific)
    likelihoods[iteration] = ill_function(y, e, parameters)
    
    # Check monotonicity (shared)
    if iteration > 0 and likelihoods[iteration] < likelihoods[iteration - 1]:
        warn("Likelihood decreased")
```

**Benefits:**
- **Single source of truth** for EM iteration logic
- **Eliminates drift** between implementations
- **Easy to verify** and maintain
- **Callback-based** design is proven and extensible

### Integration: Two-State Model

**Function:** `fit_two_state(...)`

```python
def fit_two_state(parameters, y, r, EC, EC_value, c, search_space, constraints, num_iterations):
    e = compute_error(y, r, EC, EC_value)
    
    # Wrapper that includes constraints
    def two_state_m_step(param, xnN, VnN, Vnp1nN, y, e, search_space, constraints):
        return m_step(param, xnN, VnN, Vnp1nN, y, e, c, search_space, constraints)
    
    return _run_em(
        parameters, y, e, search_space, num_iterations,
        smoother=lambda param, y, e: kalman_smoother(param, y, e, c),
        m_step_optimizer=two_state_m_step,
        ill_function=lambda y, e, param: incomplete_log_likelihood(y, e, c, param),
        m_step_extra_args=(constraints,)
    )
```

### Integration: One-State Model

**Function:** `fit_one_state(...)`

```python
def fit_one_state(parameters, y, r, EC, EC_value, search_space, num_iterations):
    e = compute_error(y, r, EC, EC_value)
    
    return _run_em(
        parameters, y, e, search_space, num_iterations,
        smoother=kalman_smoother_one_state,
        m_step_optimizer=m_step_one_state,
        ill_function=incomplete_log_likelihood_one_state
    )
```

---

## Component Details

### 1. Parameter Utilities ✓

**File:** [src/albert_em/params_one_state.py](src/albert_em/params_one_state.py)

```python
pack_one_state_params(A, B, Q, R, x0, P0) -> np.ndarray
unpack_one_state_params(theta: np.ndarray) -> (A, B, Q, R, x0, P0)
```

**Parameters:**
- `A`: State transition coefficient (decay/retention)
- `B`: Error input coefficient
- `Q`: Process noise variance
- `R`: Observation noise variance
- `x0`: Initial state mean
- `P0`: Initial state variance

**Model:**
```
x_{n+1} ~ N(A*x_n + B*e_n, Q)
y_n ~ N(x_n, R)
```

where `e_n = r_n - y_n` (error signal from perturbation and output)

### 2. One-State Kalman Smoother ✓

**File:** [src/albert_em/kalman_smoother_one_state.py](src/albert_em/kalman_smoother_one_state.py)

**Signature:** `(parameters, y, e) → (xnN, VnN, Vnp1nN)`

- **Forward Kalman filter** on observations `y` given state dynamics
- **RTS backward smoother** to combine forward and backward information
- Returns smoothed expectations and covariances
- Works with scalar states and observations

**Output Format (matched to two-state):**
- `xnN`: List of N smoothed state means (scalars for one-state)
- `VnN`: List of N smoothed state variances (scalars for one-state)
- `Vnp1nN`: List of N-1 lag-one covariances (scalars for one-state)

### 3. Expected Complete Log-Likelihood ✓

**File:** [src/albert_em/expected_complete_log_likelihood_one_state.py](src/albert_em/expected_complete_log_likelihood_one_state.py)

**Signature:** `(parameters, y, e, xnN, VnN, Vnp1nN) → float`

Computes expected complete log-likelihood:

$$\mathcal{L}_C = \text{Term1} + \text{Term2} + \text{Term3} + \text{Term4} + \text{Term5}$$

- **Term1:** Observation error: $-\frac{1}{2R} \sum_n E[(y_n - x_n)^2]$
- **Term2:** Dynamics error: $-\frac{1}{2Q} \sum_{n=1}^{N-1} E[(x_{n+1} - Ax_n - Be_n)^2]$
- **Term3:** Initial state error: $-\frac{1}{2P_0} E[(x_0 - x0)^2]$
- **Term4:** Log-determinants: $-\frac{1}{2}\log P_0 - \frac{N}{2}\log R - \text{const}$
- **Term5:** Process noise log-det: $-\frac{N-1}{2}\log Q$

Uses smoothed states and covariances to compute expectations.

### 4. Incomplete Log-Likelihood ✓

**File:** [src/albert_em/incomplete_log_likelihood_one_state.py](src/albert_em/incomplete_log_likelihood_one_state.py)

**Signature:** `(y, e, parameters) → float`

Computes marginal likelihood using **forward Kalman filter only**:

$$\mathcal{L}_I = \log p(y_1, \ldots, y_N | \text{parameters})$$

$$= \sum_n \left[ -\frac{1}{2}\log(V_{n|n-1} + R) - \frac{1}{2}\frac{(y_n - \mu_{n|n-1})^2}{V_{n|n-1} + R} \right]$$

**Properties:**
- Should be non-decreasing across EM iterations
- Used for convergence monitoring and diagnostics
- No backward smoothing (more efficient)

### 5. M-Step Optimizer ✓

**File:** [src/albert_em/m_step_one_state.py](src/albert_em/m_step_one_state.py)

**Signature:** `(parameters, xnN, VnN, Vnp1nN, y, e, search_space) → parameters`

Performs constrained optimization:
- **Method:** SciPy `minimize` with `SLSQP`
- **Objective:** Negative expected complete log-likelihood
- **Constraints:** Parameter bounds from `search_space`
- **No additional constraints** for one-state model (unlike two-state's A≥B relations)

### 6. Simulation Functions ✓

**File:** [src/albert_em/simulation.py](src/albert_em/simulation.py)

```python
one_state_simulation_with_noise(parameters, r, EC, EC_value) → (y, x)
one_state_simulation_without_noise(parameters, r, EC, EC_value) → (y, x)
```

Generates synthetic data according to the one-state model with optional noise.

---

## Test Coverage

**File:** [tests/test_one_state.py](tests/test_one_state.py)

### 1. Parameter Recovery Test
- Simulates one-state data from known parameters
- Fits EM with perturbed initial guess
- Verifies recovered parameters match ground truth (within tolerance)
- **Status:** ✅ PASSED

### 2. EM Monotonicity Test
- Verifies incomplete log-likelihood is non-decreasing
- Checks across 15 iterations with random initialization
- **Status:** ✅ PASSED

### 3. Integration Tests
- Basic fitting workflow with synthetic data
- Error-clamp trial handling
- **Status:** ✅ PASSED

### 4. Two-State Regression Tests
- Verifies existing two-state functionality unchanged
- **Status:** ✅ PASSED

**Overall:** 6 passed, 1 skipped in 7.51s

---

## Key Design Decisions

### Duplication: Necessary Components
- ✅ **Kalman smoothers** - Different for 1D vs 2D states
- ✅ **Likelihood functions** - Different for scalar vs matrix dimensions
- ✅ **M-step optimizers** - Constraints differ (none for one-state)

### No Duplication: Shared Core
- ✅ **EM loop** - Via `_run_em()` callback pattern
- ✅ **Error computation** - Common in both wrappers
- ✅ **Convergence monitoring** - Unified in `_run_em()`

### Mechanical Refactoring Path
Once proven stable, a future refactor could:
1. Parameterize state dimension `d` and measurement dimension `p`
2. Generalize Kalman smoother for arbitrary dimensions
3. Unify likelihood computations with matrix operations
4. Create single `fit_lti_em(dimension=(d,p), ...)` function
5. Keep `fit_one_state()` and `fit_two_state()` as convenience APIs

**This refactor would be purely mechanical** - no API changes required.

---

## Public API

### Main Entry Points
```python
fit_one_state(parameters, y, r, EC, EC_value, search_space, num_iterations)
fit_two_state(parameters, y, r, EC, EC_value, c, search_space, constraints, num_iterations)
```

### One-State Utilities
```python
pack_one_state_params(A, B, Q, R, x0, P0) → parameters
unpack_one_state_params(parameters) → (A, B, Q, R, x0, P0)
```

### One-State Simulation
```python
one_state_simulation_with_noise(parameters, r, EC, EC_value) → (y, x)
one_state_simulation_without_noise(parameters, r, EC, EC_value) → (y, x)
```

### Component Functions (Internal Use)
```python
kalman_smoother_one_state(parameters, y, e)
m_step_one_state(parameters, xnN, VnN, Vnp1nN, y, e, search_space)
expected_complete_log_likelihood_one_state(parameters, y, e, xnN, VnN, Vnp1nN)
incomplete_log_likelihood_one_state(y, e, parameters)
```

---

## Backward Compatibility

✅ **Full backward compatibility maintained:**
- `fit_two_state()` preserves original API
- `generalized_expectation_maximization()` behavior unchanged (still works, now calls `_run_em()`)
- All existing two-state tests passing
- No breaking changes to public interface

---

## Files Structure

**Created:**
- `src/albert_em/params_one_state.py` - Parameter utilities
- `src/albert_em/kalman_smoother_one_state.py` - E-step
- `src/albert_em/expected_complete_log_likelihood_one_state.py` - M-step objective
- `src/albert_em/incomplete_log_likelihood_one_state.py` - Convergence monitoring
- `src/albert_em/m_step_one_state.py` - M-step optimizer
- `src/albert_em/simulation.py` - Added one-state simulation functions
- `tests/test_one_state.py` - Comprehensive test suite

**Modified:**
- `src/albert_em/em.py` - Added `_run_em()`, refactored wrappers
- `src/albert_em/__init__.py` - Updated module docstring, new exports
- `src/albert_em/simulation.py` - Added one-state simulation functions

---

## Summary

**Status:** ✅ **COMPLETE AND REFACTORED**

The implementation achieves all goals:
- ✅ One-state EM model fully functional
- ✅ Two-state model preserved unchanged
- ✅ Shared EM driver (no duplication of core logic)
- ✅ Thin wrappers for model selection
- ✅ Comprehensive tests passing
- ✅ Clean refactoring path prepared
- ✅ Production-ready quality

The architecture is maintainable, extensible, and ready for future generalization to arbitrary LDS dimensions.
