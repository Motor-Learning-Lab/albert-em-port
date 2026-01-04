# Quick Reference: One-State EM Model API

## Fitting Functions

### One-State Model
```python
from albert_em import fit_one_state, pack_one_state_params

# Pack parameters: [A, B, Q, R, x0, P0]
params = pack_one_state_params(0.95, 0.15, 0.01, 0.05, 0.0, 0.1)

# Set up parameter bounds (6 parameters × 2 columns for [lower, upper])
bounds = np.array([
    [0.0, 1.0],    # A: retention factor
    [0.0, 1.0],    # B: learning rate
    [1e-6, 1.0],   # Q: process noise variance
    [1e-6, 10.0],  # R: observation noise variance
    [-5.0, 5.0],   # x0: initial state mean
    [1e-6, 10.0]   # P0: initial state variance
])

# Fit model
fitted_params, likelihoods = fit_one_state(
    parameters=params,
    y=y_data,                    # observations (N,)
    r=r_data,                    # perturbations (N,)
    EC=error_clamp_indicator,    # error-clamp trials (N,)
    EC_value=error_clamp_values, # clamped error values (N,)
    search_space=bounds,         # parameter bounds (6x2)
    num_iterations=20            # EM iterations
)
```

### Two-State Model (unchanged)
```python
from albert_em import fit_two_state

# Use as before - API unchanged
fitted_params, likelihoods = fit_two_state(
    parameters=params_2state,
    y=y_data,
    r=r_data,
    EC=error_clamp_indicator,
    EC_value=error_clamp_values,
    c=c_vector,                  # observation matrix
    search_space=bounds,
    constraints=constraints,
    num_iterations=20
)
```

## Component Functions

### Kalman Smoother
```python
from albert_em import kalman_smoother_one_state

xnN, VnN, Vnp1nN = kalman_smoother_one_state(params, y, e)
# Returns:
#   xnN: Smoothed state means (list of N scalars)
#   VnN: Smoothed state variances (list of N scalars)
#   Vnp1nN: Lag-one covariances (list of N-1 scalars)
```

### Likelihood Functions
```python
from albert_em import (
    expected_complete_log_likelihood_one_state,
    incomplete_log_likelihood_one_state
)

# Expected complete log-likelihood (M-step objective)
ecll = expected_complete_log_likelihood_one_state(
    params, y, e, xnN, VnN, Vnp1nN
)

# Incomplete log-likelihood (convergence monitoring)
ill = incomplete_log_likelihood_one_state(y, e, params)
```

### M-Step Optimizer
```python
from albert_em import m_step_one_state

new_params = m_step_one_state(
    parameters_0=current_params,
    xnN=xnN,
    VnN=VnN,
    Vnp1nN=Vnp1nN,
    y=y_data,
    e=errors,
    search_space=bounds
)
```

## Simulation Functions

```python
from albert_em import (
    one_state_simulation_with_noise,
    one_state_simulation_without_noise
)

# With noise
y_noisy, x_true = one_state_simulation_with_noise(
    params, r, EC, EC_value
)

# Without noise
y_clean, x_true = one_state_simulation_without_noise(
    params, r, EC, EC_value
)
```

## Parameter Utilities

```python
from albert_em import pack_one_state_params, unpack_one_state_params

# Pack into single array
theta = pack_one_state_params(A=0.95, B=0.15, Q=0.01, R=0.05, x0=0.0, P0=0.1)

# Unpack from array
A, B, Q, R, x0, P0 = unpack_one_state_params(theta)
```

## Model Parameters

| Parameter | Symbol | Type | Meaning |
|-----------|--------|------|---------|
| A | $a$ | float | State transition coefficient |
| B | $b$ | float | Input gain coefficient |
| Q | $q$ | float | Process noise variance |
| R | $r$ | float | Observation noise variance |
| x0 | $\mu_0$ | float | Initial state mean |
| P0 | $\sigma_0^2$ | float | Initial state variance |

## Model Equations

**State dynamics:**
$$x_{n+1} \sim \mathcal{N}(A x_n + B e_n, Q)$$

**Observation model:**
$$y_n \sim \mathcal{N}(x_n, R)$$

where:
- $x_n$ = hidden state at trial $n$
- $y_n$ = observed motor output at trial $n$
- $e_n$ = error signal (difference between perturbation and output)
- $A \in [0,1]$ = state retention (learning rate)
- $B > 0$ = error sensitivity
- $Q > 0$ = learning variability
- $R > 0$ = observation noise

## Search Space Bounds

Recommended default bounds:
```python
search_space = np.array([
    [0.5, 1.0],     # A: state retention (0.5 to 1.0)
    [0.0, 0.5],     # B: error sensitivity (0 to 0.5)
    [0.001, 0.5],   # Q: process noise variance
    [0.001, 0.5],   # R: observation noise variance
    [-1.0, 1.0],    # x0: initial state (unrestricted)
    [0.001, 1.0],   # P0: initial variance (positive)
])
```

## Common Workflows

### Parameter Recovery from Simulation
```python
import numpy as np
from albert_em import (
    pack_one_state_params,
    one_state_simulation_with_noise,
    fit_one_state
)

# True parameters
A_true, B_true = 0.95, 0.15
Q_true, R_true = 0.01, 0.05
x0_true, P0_true = 0.0, 0.1
true_params = pack_one_state_params(A_true, B_true, Q_true, R_true, x0_true, P0_true)

# Generate data
N = 300
r = np.zeros(N)
r[50:150] = 10.0
EC = np.zeros(N)
EC_value = np.zeros(N)
y, x = one_state_simulation_with_noise(true_params, r, EC, EC_value)

# Fit with perturbed initial guess
init_params = pack_one_state_params(0.9, 0.2, 0.02, 0.08, 0.0, 0.2)
search_space = np.array([[0.5, 1.0], [0.0, 0.5], [0.001, 0.5], [0.001, 0.5], [-1.0, 1.0], [0.001, 1.0]])

fitted_params, likelihoods = fit_one_state(
    init_params, y, r, EC, EC_value, search_space, num_iterations=20
)

# Check recovery
print(f"A: {fitted_params[0]:.4f} (true: {A_true:.4f})")
print(f"B: {fitted_params[1]:.4f} (true: {B_true:.4f})")
```

### Monitoring Convergence
```python
# Check that likelihood monotonically increases
for i in range(1, len(likelihoods)):
    if likelihoods[i] < likelihoods[i-1]:
        print(f"Warning: Likelihood decreased at iteration {i}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure()
plt.plot(likelihoods)
plt.xlabel('EM Iteration')
plt.ylabel('Incomplete Log-Likelihood')
plt.title('EM Convergence')
plt.show()
```

## Testing

Run full test suite:
```bash
pixi run pytest tests/ -v
```

Run only one-state tests:
```bash
pixi run pytest tests/test_one_state.py -v
```

Run specific test:
```bash
pixi run pytest tests/test_one_state.py::TestOneStateParameterRecovery::test_parameter_recovery_simple -v
```

## Files Organization

```
src/albert_em/
├── em.py                                    # EM driver + wrappers (fit_two_state, fit_one_state)
├── kalman_smoother.py                       # Two-state Kalman smoother
├── kalman_smoother_one_state.py             # One-state Kalman smoother
├── m_step.py                                # Two-state M-step optimizer
├── m_step_one_state.py                      # One-state M-step optimizer
├── expected_complete_log_likelihood.py      # Two-state ECLL
├── expected_complete_log_likelihood_one_state.py  # One-state ECLL
├── incomplete_log_likelihood.py             # Two-state ILL
├── incomplete_log_likelihood_one_state.py   # One-state ILL
├── simulation.py                            # Simulation functions (both models)
├── params_one_state.py                      # Parameter packing utilities
└── __init__.py                              # Module exports

tests/
├── test_one_state.py                        # One-state model tests
├── test_smoke.py                            # Two-state regression tests
└── test_benchmark.py                        # Performance benchmarks
```

## Notes

- The one-state model is designed for simple, single-timescale learning (no fast/slow decomposition)
- Parameter recovery tests verify that EM can identify true parameters from noisy data
- Monotonicity tests ensure algorithmic stability
- All error-clamp trial logic is preserved from two-state implementation
- Two-state API remains fully unchanged for backward compatibility
