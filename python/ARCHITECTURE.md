# Python EM Algorithm Structure

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│         Generalized Expectation-Maximization                │
│                                                              │
│  Input: y (observations), r (perturbations), EC, c          │
│  Initial: parameters = [aS, aF, bS, bF, ...]               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   For each EM iteration:      │
            └───────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         E-STEP                        │
        │   (Kalman Smoother)                   │
        │                                       │
        │  1. Forward Kalman Filter             │
        │     → Compute xnn, Vnn               │
        │                                       │
        │  2. Backward Smoother                 │
        │     → Compute xnN, VnN, Vnp1nN       │
        │                                       │
        │  Output: Smoothed state estimates     │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         M-STEP                        │
        │   (Constrained Optimization)          │
        │                                       │
        │  Maximize:                            │
        │    Expected Complete Log-Likelihood   │
        │                                       │
        │  Subject to:                          │
        │    - Parameter bounds                 │
        │    - Linear constraints               │
        │      • aS >= aF + deltaA             │
        │      • bF >= bS + deltaB             │
        │                                       │
        │  Method: SciPy SLSQP                  │
        │                                       │
        │  Output: Updated parameters           │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   CONVERGENCE CHECK                   │
        │                                       │
        │  Compute Incomplete Log-Likelihood    │
        │  Check: L[n] >= L[n-1]               │
        └───────────────────────────────────────┘
                            │
                            ▼
                    Continue or Done?
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Output: Final parameters     │
            │  Output: Likelihood history   │
            └───────────────────────────────┘
```

## Module Dependencies

```
tutorial.py
    │
    ├──► generalized_expectation_maximization.py
    │       │
    │       ├──► kalman_smoother.py
    │       │       └──► numpy
    │       │
    │       ├──► m_step.py
    │       │       ├──► expected_complete_log_likelihood.py
    │       │       │       └──► numpy
    │       │       └──► scipy.optimize
    │       │
    │       └──► incomplete_log_likelihood.py
    │               └──► numpy
    │
    ├──► two_state_simulation.py
    │       └──► numpy
    │
    └──► matplotlib.pyplot
```

## Data Flow

```
Input Data:
┌────────────────────────────────────────────────────┐
│  y: [y₁, y₂, ..., yₙ]     Motor outputs           │
│  r: [r₁, r₂, ..., rₙ]     Perturbations           │
│  EC: [0, 0, 1, ...]        Error-clamp indicators  │
│  EC_value: [nan, nan, 0, ...] Clamped errors      │
│  c: [1, 1]                 State weights           │
└────────────────────────────────────────────────────┘
                    │
                    ▼
          Compute Errors: e[n]
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│              Kalman Smoother (E-step)              │
│                                                    │
│  Forward Pass:                                     │
│    xnnm1[n] → xnn[n]     (Prior → Posterior)     │
│    Vnnm1[n] → Vnn[n]                              │
│                                                    │
│  Backward Pass:                                    │
│    xnn[n] → xnN[n]       (Filtered → Smoothed)   │
│    Vnn[n] → VnN[n]                                │
│    Compute Vnp1nN[n]     (State covariances)      │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│              M-step Optimization                   │
│                                                    │
│  Input: xnN, VnN, Vnp1nN, y, e, c                │
│                                                    │
│  Compute: Expected Complete Log-Likelihood         │
│    = Σ [Term1 + Term2 + Term3 + Term4 + Term5]   │
│                                                    │
│  Optimize: parameters to maximize likelihood       │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│             Updated Parameters                     │
│                                                    │
│  θ = [aS, aF, bS, bF, xS1, xF1,                  │
│       sigmax2, sigmau2, sigma12]                  │
└────────────────────────────────────────────────────┘
```

## State-Space Model

```
States:
┌─────────────────┐
│  x[n] = [xS[n]] │  xS = Slow state
│         [xF[n]] │  xF = Fast state
└─────────────────┘

State Update:
x[n+1] = A·x[n] + b·e[n] + w[n]

where:
    A = [aS  0 ]    Retention factors
        [0   aF]

    b = [bS]        Error sensitivities
        [bF]

    e[n] = r[n] - y[n]    Error signal
    w[n] ~ N(0, Q)         Process noise
    Q = diag([sigmax2, sigmax2])

Observation:
y[n] = c'·x[n] + v[n]

where:
    c = [1]         State weights (typically)
        [1]

    v[n] ~ N(0, sigmau2)    Measurement noise
```

## Two-State Model Intuition

```
                    Fast State (xF)
                    ┌─────────────┐
                    │ ▲ High bF   │
                    │ │           │
Rapid learning ───► │ │ Low aF    │ ◄─── Rapid decay
                    │ ▼           │
                    └─────────────┘
                           │
                           │ c'·x
                           │
                           ▼
                    Motor Output (y)
                           ▲
                           │ c'·x
                           │
                    ┌─────────────┐
                    │ ▲ Low bS    │
Slow learning  ───► │ │           │ ◄─── Slow decay
                    │ │ High aS   │
                    │ ▼           │
                    └─────────────┘
                    Slow State (xS)

Key Properties:
- Fast state: Quick to learn (high bF), quick to forget (low aF)
- Slow state: Slow to learn (low bS), slow to forget (high aS)
- Constraint: aS > aF  (slow retains more)
- Constraint: bF > bS  (fast learns more)
```

## File Structure Summary

```
python/
├── Core Algorithm
│   ├── generalized_expectation_maximization.py  (Main coordinator)
│   ├── kalman_smoother.py                      (E-step)
│   ├── m_step.py                               (M-step)
│   ├── expected_complete_log_likelihood.py     (Objective function)
│   └── incomplete_log_likelihood.py            (Monitoring)
│
├── Simulation & Testing
│   ├── two_state_simulation.py                 (Data generation)
│   ├── tutorial.py                             (Demo script)
│   └── test_em.py                              (Quick test)
│
├── Documentation
│   ├── README.md                               (Main docs)
│   ├── QUICK_REFERENCE.py                      (Cheat sheet)
│   └── __init__.py                             (Package)
│
└── Configuration
    └── requirements.txt                        (Dependencies)
```
