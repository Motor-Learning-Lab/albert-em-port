"""Expected Complete Log-Likelihood (Numba + Pure Python fallback).

If Numba is available and not disabled (env var ALBERT_EM_DISABLE_NUMBA=1),
vectorized JIT-compiled routines are used. Otherwise a pure Python/NumPy
implementation executes without acceleration.

Environment variable to force fallback:
    ALBERT_EM_DISABLE_NUMBA=1
"""

from typing import List
import os
import numpy as np

# Attempt to enable numba; allow user/environment to disable.
USING_NUMBA = False
if os.getenv("ALBERT_EM_DISABLE_NUMBA", "0") != "1":
    try:
        from numba import jit  # type: ignore

        USING_NUMBA = True
    except Exception:  # numba not installed or incompatible
        USING_NUMBA = False

if USING_NUMBA:

    @jit(nopython=True, cache=True)
    def _compute_term1(y, c, xnN_array, VnN_array, sigmau2):
        N = len(y)
        term1 = 0.0
        for n in range(N):
            xnN = xnN_array[n].reshape(2, 1)
            VnN = VnN_array[n]
            c_col = c.reshape(2, 1)
            term1 += y[n] ** 2
            xnN_outer = xnN @ xnN.T
            mat_sum = VnN + xnN_outer
            term1 += (c_col.T @ mat_sum @ c_col)[0, 0]
            term1 -= 2.0 * y[n] * (c_col.T @ xnN)[0, 0]
        return -term1 / (2.0 * sigmau2)

    @jit(nopython=True, cache=True)
    def _compute_term2(e, xnN_array, VnN_array, Vnp1nN_array, aS, aF, bS, bF, sigmax2):
        N = len(e)
        A = np.array([[aS, 0.0], [0.0, aF]])
        b = np.array([[bS], [bF]])
        Q = np.array([[sigmax2, 0.0], [0.0, sigmax2]])
        Qinv = np.linalg.inv(Q)
        QinvA = Qinv @ A
        AtQinv = A.T @ Qinv
        AtQinvA = AtQinv @ A
        AtQinvb = A.T @ (Qinv @ b)
        btQinv = b.T @ Qinv
        btQinvA = btQinv @ A
        btQinvb = (b.T @ Qinv @ b)[0, 0]
        term2 = 0.0
        for n in range(N - 1):
            xnN = xnN_array[n].reshape(2, 1)
            xnp1N = xnN_array[n + 1].reshape(2, 1)
            Vnp1N = VnN_array[n + 1]
            Vnp1nN = Vnp1nN_array[n]
            term2 += (xnp1N.T @ Qinv @ xnp1N)[0, 0]
            term2 += np.trace(Qinv @ Vnp1N)
            term2 -= (xnp1N.T @ QinvA @ xnN)[0, 0]
            term2 -= np.trace(QinvA @ Vnp1nN.T)
            term2 -= (xnp1N.T @ (Qinv @ b))[0, 0] * e[n]
            term2 -= (xnN.T @ AtQinv @ xnp1N)[0, 0]
            term2 -= np.trace(AtQinv @ Vnp1nN)
            term2 += (xnN.T @ AtQinvA @ xnN)[0, 0]
            term2 += np.trace(AtQinvA @ VnN_array[n])
            term2 += (xnN.T @ AtQinvb)[0, 0] * e[n]
            term2 -= e[n] * (btQinv @ xnp1N)[0, 0]
            term2 += e[n] * (btQinvA @ xnN)[0, 0]
            term2 += e[n] * btQinvb * e[n]
        return -term2 / 2.0

    @jit(nopython=True, cache=True)
    def _compute_term3(xnN_0, VnN_0, x1, V1):
        V1inv = np.linalg.inv(V1)
        xnN_col = xnN_0.reshape(2, 1)
        x1_col = x1.reshape(2, 1)
        term3 = (xnN_col.T @ V1inv @ xnN_col)[0, 0]
        term3 += np.trace(V1inv @ VnN_0)
        term3 -= (xnN_col.T @ V1inv @ x1_col)[0, 0]
        term3 -= (x1_col.T @ V1inv @ xnN_col)[0, 0]
        term3 += (x1_col.T @ V1inv @ x1_col)[0, 0]
        return -term3 / 2.0

    @jit(nopython=True, cache=True)
    def _compute_term4_term5(N, sigma12, sigmau2, sigmax2):
        det_V1 = sigma12 * sigma12
        det_Q = sigmax2 * sigmax2
        term4 = (
            -0.5 * np.log(det_V1)
            - (N / 2.0) * np.log(sigmau2)
            - (3.0 / 2.0) * N * np.log(2.0 * np.pi)
        )
        term5 = -(N - 1) * np.log(det_Q) / 2.0
        return term4, term5

else:
    # Pure Python (vectorized NumPy) fallback implementations.
    def _compute_term1(y, c, xnN_array, VnN_array, sigmau2):
        c_col = c.reshape(2, 1)
        term1 = 0.0
        for xnN, VnN, yy in zip(xnN_array, VnN_array, y):
            xnN = xnN.reshape(2, 1)
            term1 += yy**2
            mat_sum = VnN + xnN @ xnN.T
            term1 += (c_col.T @ mat_sum @ c_col)[0, 0]
            term1 -= 2.0 * yy * (c_col.T @ xnN)[0, 0]
        return -term1 / (2.0 * sigmau2)

    def _compute_term2(e, xnN_array, VnN_array, Vnp1nN_array, aS, aF, bS, bF, sigmax2):
        A = np.array([[aS, 0.0], [0.0, aF]])
        b = np.array([[bS], [bF]])
        Q = np.array([[sigmax2, 0.0], [0.0, sigmax2]])
        Qinv = np.linalg.inv(Q)
        QinvA = Qinv @ A
        AtQinv = A.T @ Qinv
        AtQinvA = AtQinv @ A
        AtQinvb = A.T @ (Qinv @ b)
        btQinv = b.T @ Qinv
        btQinvA = btQinv @ A
        btQinvb = (b.T @ Qinv @ b)[0, 0]
        term2 = 0.0
        for n in range(len(e) - 1):
            xnN = xnN_array[n].reshape(2, 1)
            xnp1N = xnN_array[n + 1].reshape(2, 1)
            Vnp1N = VnN_array[n + 1]
            Vnp1nN = Vnp1nN_array[n]
            term2 += (xnp1N.T @ Qinv @ xnp1N)[0, 0]
            term2 += np.trace(Qinv @ Vnp1N)
            term2 -= (xnp1N.T @ QinvA @ xnN)[0, 0]
            term2 -= np.trace(QinvA @ Vnp1nN.T)
            term2 -= (xnp1N.T @ (Qinv @ b))[0, 0] * e[n]
            term2 -= (xnN.T @ AtQinv @ xnp1N)[0, 0]
            term2 -= np.trace(AtQinv @ Vnp1nN)
            term2 += (xnN.T @ AtQinvA @ xnN)[0, 0]
            term2 += np.trace(AtQinvA @ VnN_array[n])
            term2 += (xnN.T @ AtQinvb)[0, 0] * e[n]
            term2 -= e[n] * (btQinv @ xnp1N)[0, 0]
            term2 += e[n] * (btQinvA @ xnN)[0, 0]
            term2 += e[n] * btQinvb * e[n]
        return -term2 / 2.0

    def _compute_term3(xnN_0, VnN_0, x1, V1):
        V1inv = np.linalg.inv(V1)
        xnN_col = xnN_0.reshape(2, 1)
        x1_col = x1.reshape(2, 1)
        term3 = (xnN_col.T @ V1inv @ xnN_col)[0, 0]
        term3 += np.trace(V1inv @ VnN_0)
        term3 -= (xnN_col.T @ V1inv @ x1_col)[0, 0]
        term3 -= (x1_col.T @ V1inv @ xnN_col)[0, 0]
        term3 += (x1_col.T @ V1inv @ x1_col)[0, 0]
        return -term3 / 2.0

    def _compute_term4_term5(N, sigma12, sigmau2, sigmax2):
        det_V1 = sigma12 * sigma12
        det_Q = sigmax2 * sigmax2
        term4 = (
            -0.5 * np.log(det_V1)
            - (N / 2.0) * np.log(sigmau2)
            - (3.0 / 2.0) * N * np.log(2.0 * np.pi)
        )
        term5 = -(N - 1) * np.log(det_Q) / 2.0
        return term4, term5


def expected_complete_log_likelihood(
    parameters: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    c: np.ndarray,
    xnN: List[np.ndarray],
    VnN: List[np.ndarray],
    Vnp1nN: List[np.ndarray],
) -> float:
    """Compute expected complete log-likelihood using either Numba or pure Python.

    Returns total log-likelihood value (scalar).
    """
    aS, aF, bS, bF, xS1, xF1, sigmax2, sigmau2, sigma12 = parameters
    N = len(y)
    xnN_array = np.array([xnN[i] for i in range(N)])
    VnN_array = np.array([VnN[i] for i in range(N)])
    Vnp1nN_array = np.array([Vnp1nN[i] for i in range(N - 1)])
    x1 = np.array([xS1, xF1])
    V1 = np.array([[sigma12, 0.0], [0.0, sigma12]])
    term1 = _compute_term1(y, c, xnN_array, VnN_array, sigmau2)
    term2 = _compute_term2(
        e, xnN_array, VnN_array, Vnp1nN_array, aS, aF, bS, bF, sigmax2
    )
    term3 = _compute_term3(xnN_array[0], VnN_array[0], x1, V1)
    term4, term5 = _compute_term4_term5(N, sigma12, sigmau2, sigmax2)
    return term1 + term2 + term3 + term4 + term5
