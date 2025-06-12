"""
t.py  –  Multivariate Chebyshev basis for Smolyak grids (Numba version)
------------------------------------------------------------------------
• Keeps the original call signature:  chebyshev_T(y, s)
• Works in nopython / parallel regions (no Python objects inside).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from numba import njit

if TYPE_CHECKING:                     # avoid circular import at run-time
    from smolyak_step_1 import SmolyakStruct

__all__ = ["chebyshev_T"]

# ────────────────────────────────────────────────────────────────────────
#  1.  Small helper: make sure y is (N, d)
# ────────────────────────────────────────────────────────────────────────
def _ensure_y_orientation(y: NDArray, d: int) -> NDArray:
    """
    Accept either (d, N) or (N, d); return a *copy* in shape (N, d).

    Raises
    ------
    ValueError if y is not 2-D or neither dimension equals d.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2-D (got shape {y.shape})")
    if y.shape[1] == d:          # already (N, d)
        return y.copy()
    if y.shape[0] == d:          # MATLAB style (d, N) → transpose
        return y.T.copy()
    raise ValueError(f"y's 2nd dim must be d = {d} (got {y.shape})")


# ────────────────────────────────────────────────────────────────────────
#  2.  Numba core (no Python objects)
# ────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _chebyshev_T_nb(Y: NDArray[np.float64],
                    l: NDArray[np.int64],
                    max_order: int) -> NDArray[np.float64]:
    """
    Parameters
    ----------
    Y         : ndarray (N, d)     – points in [-1,1]^d
    l         : ndarray (nbasis, d) – multi-indices (1-based as in MATLAB)
    max_order : int  – max(l)

    Returns
    -------
    T  : ndarray (nbasis, N) – Chebyshev basis matrix
    """
    N, d   = Y.shape
    K      = l.shape[0]                  # nbasis
    Tprod  = np.ones((N, K), dtype=Y.dtype)

    # Univariate Chebyshev T_n(y), n = 0..max_order-1
    T_uni = np.ones((N, max_order, d), dtype=Y.dtype)
    if max_order > 1:
        T_uni[:, 1, :] = Y
    twoY = 2.0 * Y
    for n in range(2, max_order):
        T_uni[:, n, :] = twoY * T_uni[:, n - 1, :] - T_uni[:, n - 2, :]

    # Multiply across dimensions for every basis function
    for k in range(K):                  # loop over basis rows
        for dim in range(d):
            order = l[k, dim] - 1       # 1-based → 0-based
            if order > 0:               # skip T_0(y)=1
                Tprod[:, k] *= T_uni[:, order, dim]

    return Tprod.T                      # shape (nbasis, N)


# ────────────────────────────────────────────────────────────────────────
#  3.  Public wrapper – keeps old signature
# ────────────────────────────────────────────────────────────────────────
def chebyshev_T(y: NDArray, s: "SmolyakStruct") -> NDArray:
    """
    Evaluate the multivariate Chebyshev basis on a Smolyak grid.

    Parameters
    ----------
    y : ndarray (d, N) or (N, d) – scaled inputs in [-1,1]
    s : SmolyakStruct            – output of smolyak_step1

    Returns
    -------
    ndarray (nbasis, N) – basis matrix
    """
    # 1. Ensure y has the right orientation (N, d)
    Y = _ensure_y_orientation(y, s.d).astype(np.float64)

    # 2. Extract NumPy bits from the struct
    l_mat      = s.l.astype(np.int64)       # (nbasis, d)
    max_order  = int(np.max(s.M_mup1))

    # 3. Dispatch to nopython kernel
    return _chebyshev_T_nb(Y, l_mat, max_order)
