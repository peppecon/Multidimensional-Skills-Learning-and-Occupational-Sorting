"""T.py – Multivariate Chebyshev basis evaluation for Smolyak grids

Port of the MATLAB routine `T(y,s)` used in `smolyakapprox_step2/3`.
Given **y** (scaled inputs in [-1,1]^d) and the *structure* **s**
returned by ``smolyak_step1``, this function returns the tensor‑product
Chebyshev basis evaluated at all points.

*Input* dimensions*
------------------
``y`` : ndarray, shape (d, N) **or** (N, d)
    Coordinates in [-1,1].  If shape is (d, N) (MATLAB style), we
    transpose internally.
``s`` : SmolyakStruct
    Structure created by ``smolyak_step1``; must provide fields
    ``d``, ``k1``, ``l``, and ``M_mup1``.

*Output*
--------
``T`` : ndarray, shape (nbasis, N)
    Each *column* holds the product of univariate Chebyshev polynomials
    corresponding to one multi‑index **l** (rows are basis functions).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from smolyak_step_1 import SmolyakStruct

__all__ = ["chebyshev_T"]


def _ensure_y_orientation(y: NDArray, d: int) -> NDArray:
    """Return y in shape (N, d). Accepts (d, N) or (N, d)."""
    if y.ndim != 2:
        raise ValueError("y must be 2‑D (got shape %s)" % (y.shape,))
    if y.shape[0] == d:
        return y.T.copy()
    if y.shape[1] == d:
        return y.copy()
    raise ValueError("y's second dimension must match d=%d" % d)


def chebyshev_T(y: NDArray, s: "SmolyakStruct") -> NDArray:
    """Evaluate multivariate Chebyshev basis for all points in *y*.

    Parameters
    ----------
    y : ndarray
        Either (d, N) or (N, d) array of scaled inputs.
    s : SmolyakStruct
        Structure from ``smolyak_step1`` providing necessary indices.

    Returns
    -------
    ndarray
        Matrix ``T`` of shape (nbasis, N) where ``nbasis = s.l.shape[0]``.
    """
    d = s.d
    Y = _ensure_y_orientation(y, d)  # (N, d)
    N = Y.shape[0]

    max_order = int(np.max(s.M_mup1))

    # Univariate Chebyshev polynomials T_n(y) up to n=max_order
    T_uni = np.ones((N, max_order, d), dtype=Y.dtype)
    T_uni[:, 1, :] = Y  # T_1 = y  (note: index offset 1 vs MATLAB 2)

    twoY = 2.0 * Y
    for n in range(2, max_order):  # Python index 2 == MATLAB order 3
        T_uni[:, n, :] = twoY * T_uni[:, n - 1, :] - T_uni[:, n - 2, :]

    # Build multivariate products for each multi‑index l (rows of s.l)
    nbasis = s.l.shape[0]
    Tprod = np.ones((N, nbasis), dtype=Y.dtype)

    for dim in range(d):
        li = s.l[:, dim]  # 1‑based in MATLAB; keep as int array
        mask = li > 1     # ignore where order == 1 (T_0 = 1)
        if np.any(mask):
            # subtract 1 for 0‑based index into T_uni
            orders = li[mask] - 1
            Tprod[:, mask] *= T_uni[:, orders, dim]

    return Tprod.T  # (nbasis, N)
