"""smolyak_utils.py – helper utilities for Smolyak grids

Provides MATLAB‑style helper functions used when building sparse
Smolyak grids.

Functions
---------
smolyak_m1(i)
    m(i) = 2^(i-2)  with special‑case values m(1)=1, m(2)=2.
smolyak_m(i)
    m(i) = 2^(i-1) + 1  with special‑case value m(1)=1.

Both accept scalars, Python sequences, or NumPy arrays and raise
``ValueError`` if any element of *i* is < 1.  They return either a
scalar ``int`` or an ``ndarray[int]`` matching the input shape.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["smolyak_m1", "smolyak_m"]


# ---------------------------------------------------------------------------
# m₁(i) = 2^(i‑2)   (with MATLAB edge cases)                                 ──
# ---------------------------------------------------------------------------

def smolyak_m1(i: ArrayLike) -> Union[int, NDArray[np.int64]]:
    """Vectorised equivalent of MATLAB `smolyakM1`.

    m(i) = 1            if i == 1
           2            if i == 2
           2^(i-2)      otherwise
    """
    arr = np.asarray(i, dtype=np.int64)
    if np.any(arr < 1):
        raise ValueError("i must be >= 1")

    out = np.empty_like(arr, dtype=np.int64)
    mask1 = arr == 1
    mask2 = arr == 2
    mask_rest = ~(mask1 | mask2)

    out[mask1] = 1
    out[mask2] = 2
    out[mask_rest] = 2 ** (arr[mask_rest] - 2)

    return int(out) if np.isscalar(i) else out


# ---------------------------------------------------------------------------
# m(i) = 2^(i‑1) + 1                                                         ──
# ---------------------------------------------------------------------------

def smolyak_m(i: ArrayLike) -> Union[int, NDArray[np.int64]]:
    """Vectorised equivalent of MATLAB `smolyakM`.

    m(i) = 1                if i == 1
           2^(i-1) + 1      otherwise
    """
    arr = np.asarray(i, dtype=np.int64)
    if np.any(arr < 1):
        raise ValueError("i must be >= 1")

    out = np.empty_like(arr, dtype=np.int64)
    mask1 = arr == 1
    mask_rest = ~mask1

    out[mask1] = 1
    out[mask_rest] = 2 ** (arr[mask_rest] - 1) + 1

    return int(out) if np.isscalar(i) else out
