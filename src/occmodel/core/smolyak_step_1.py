"""smolyak_step1.py – Build Smolyak collocation grid (Step 1)

Python translation of the MATLAB routine `smolyakapprox_step1AS`.
It constructs the Smolyak grid points **x** (shape ``(d, N)``) for a
hypercube *[a, b]^d* at anisotropic level *mu* and returns a structure
`S` with all auxiliary indices later consumed by Step 2 / Step 3.

The implementation is *stand‑alone* but relies on a few helper
routines that are already (or will be) part of ``occmodel.core``:

* ``smolyak_m``      – see *smolyak_utils.py*
* ``smolyak_m1``     – idem
* ``smolyak_enumerate`` – enumerate non‑negative integer vectors of
  length ``d`` with fixed sum (simple compositions algorithm provided
  below)
* ``smolyak_g`` – 1‑D Clenshaw‑Curtis abscissae for level ``k``

If your project already implements alternative versions of these
helpers, simply swap the imports at the top.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np

from occmodel.core.smoliyak_utils import smolyak_m, smolyak_m1

__all__ = ["smolyak_step1"]


# ---------------------------------------------------------------------------
# Helper utilities                                                           ──
# ---------------------------------------------------------------------------

def smolyak_enumerate(d: int, m: int) -> np.ndarray:
    """Return all non‑negative integer vectors (length *d*) summing to *m*.

    Equivalent to MATLAB `smolyakEnumerate(d, m)` used in the original
    code.  The output is an (n_comb, d) array, each row a vector.
    """
    if m < 0:
        raise ValueError("m must be ≥ 0 (got m={m})")
    if d < 1:
        raise ValueError("d must be ≥ 1")

    # Using combinatorics composition formula via stars‑and‑bars
    # Generate all placements of d‑1 separators in m+d‑1 positions.
    combs = itertools.combinations(range(m + d - 1), d - 1)
    out: List[List[int]] = []
    for c in combs:
        # Convert separator positions to counts
        prev = -1
        vec = []
        for sep in c + (m + d - 1,):
            vec.append(sep - prev - 1)
            prev = sep
        out.append(vec)
    return np.asarray(out, dtype=int)


def smolyak_g(k: int) -> np.ndarray:
    """Return Clenshaw‑Curtis nodes in [-1,1] for level *k* (≥ 1).

    * k == 1 → [0]
    * k >  1 → cos(pi * (j‑1)/(k‑1))  for j = 1..k (reversed to match MATLAB)
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")
    if k == 1:
        return np.array([0.0])
    j = np.arange(1, k + 1)
    return np.cos(np.pi * (j - 1) / (k - 1))


def cartprod(a: np.ndarray | None, b: np.ndarray) -> np.ndarray:
    """Cartesian product helper (like MATLAB's custom `cartprod`)."""
    if a is None or a.size == 0:
        return b.reshape(-1, 1)
    a = a.reshape(-1, a.shape[-1])  # ensure 2‑D
    b = b.reshape(-1, 1)
    return np.hstack([np.repeat(a, len(b), axis=0), np.tile(b, (len(a), 1))])


# ---------------------------------------------------------------------------
# Data container for indices                                                 ──
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SmolyakStruct:
    d: int
    mu: np.ndarray
    a: np.ndarray
    b: np.ndarray

    # populated during construction
    mu_m: int = 0
    q: int = 0
    M_mup1: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    i: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    ibar: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    leni: int = 0
    k: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    k1: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    Lb: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    Ub: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    z: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    j: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    l: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    x: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


# ---------------------------------------------------------------------------
# Main routine                                                               ──
# ---------------------------------------------------------------------------

def smolyak_step1(d: int, mu: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, SmolyakStruct]:
    """Port of MATLAB `smolyakapprox_step1AS` (anisotropic Smolyak grid).

    Parameters
    ----------
    d  : int            – number of dimensions
    mu : ndarray[int]   – *d*‑vector of accuracy levels (non‑negative)
    a, b : ndarray[float] – lower / upper bounds for each dimension

    Returns
    -------
    x : ndarray, shape (d, N)
        Collocation points in **columns** (matches MATLAB output).
    S : `SmolyakStruct`
        Bag of indices used by later steps.
    """
    mu = np.asarray(mu, dtype=int).ravel()
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if mu.size != d:
        raise ValueError("mu must have length d")

    q = max(d, int(mu.max()) + 1)
    S = SmolyakStruct(d=d, mu=mu, a=a[:d], b=b[:d])
    S.mu_m = int(mu.max())
    S.q = q
    S.M_mup1 = smolyak_m(mu + 1)

    # Enumerate index vectors i with q ≤ |i| ≤ d + mu_m and element‑wise i_j ≤ mu_j + 1
    tmp_list: List[np.ndarray] = []
    for ibar in range(max(d, q), d + S.mu_m + 1):
        enum = smolyak_enumerate(d, ibar - d) + 1  # MATLAB has 1‑based indices
        for row in enum:
            if np.all(row <= mu + 1):
                tmp_list.append(row)
    if not tmp_list:
        raise RuntimeError("No admissible multi‑indices found.")
    S.i = np.unique(np.vstack(tmp_list), axis=0)
    S.leni = S.i.shape[0]
    S.ibar = S.i.sum(axis=1)

    # k and k1 arrays and block ranges
    S.k = smolyak_m(S.i)
    S.k1 = smolyak_m1(S.i)

    S.Lb = np.empty(S.leni, dtype=int)
    S.Ub = np.empty(S.leni, dtype=int)

    offset = 0
    for idx in range(S.leni):
        S.Lb[idx] = offset
        offset += int(np.prod(S.k[idx]))
        S.Ub[idx] = offset
    total_points = S.Ub[-1]

    S.z = np.empty((total_points, d))
    S.j = np.empty((total_points, d), dtype=int)

    for idx in range(S.leni):
        ztmp: np.ndarray | None = None
        jtmp: np.ndarray | None = None
        for dim in range(d):
            g = smolyak_g(int(S.k[idx, dim]))
            ztmp = cartprod(ztmp, g)
            jtmp = cartprod(jtmp, np.arange(1, int(S.k[idx, dim]) + 1))
        S.z[S.Lb[idx]:S.Ub[idx], :] = ztmp
        S.j[S.Lb[idx]:S.Ub[idx], :] = jtmp

    # unique points & scaling to [a,b]
    S.l = np.unique(S.j, axis=0)
    S.x = (S.z + 1.0) * (S.b - S.a) / 2.0 + S.a
    x = np.unique(S.x, axis=0).T  # columns as in MATLAB
    return x, S
