"""quadrature.py – Gaussian quadrature utilities

Implements Gauss–Hermite abscissae and weights using the symmetric‐
tridiagonal companion‑matrix approach (faithful to the MATLAB routine
`GaussHermite_2`).  Suitable for *any* order ``n ≥ 2``.

Place this file under ``src/occmodel/core/`` so you can later import with
``from occmodel.core.quadrature import gauss_hermite``.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["gauss_hermite"]


def gauss_hermite(n: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return abscissae *x* and weights *w* for *n*-point Gauss–Hermite quadrature.

    Parameters
    ----------
    n : int
        Quadrature order (must be ``n ≥ 2``).

    Returns
    -------
    x : ndarray, shape (n,)
        Sorted abscissae (roots of the *n*‑th degree Hermite polynomial).
    w : ndarray, shape (n,)
        Corresponding weights that satisfy ::

            ∫_{−∞}^{+∞} f(t) e^{−t²} dt ≈ Σ_{i=1}^{n} w_i f(x_i)

    Notes
    -----
    We construct the symmetric tridiagonal **companion matrix** of the
    Hermite polynomial and use its eigenvalue/eigenvector decomposition
    ("Golub–Welsch" algorithm).  This guarantees real roots and numerical
    stability.
    """
    if n < 2:
        raise ValueError("Gauss–Hermite requires n ≥ 2 (got n={n}).")

    # Build symmetric tridiagonal companion matrix
    i = np.arange(1, n, dtype=np.float64)
    a = np.sqrt(i / 2.0)
    CM = np.diag(a, k=1) + np.diag(a, k=-1)

    # Eigen‑decomposition (CM is symmetric ⇒ eigh gives sorted eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(CM)

    # Sort explicitly for clarity (they *should* already be ascending)
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    V = eigvecs[:, idx].T  # rows correspond to eigenvectors after sort

    # Weights derived from first component of each eigenvector
    w = math.sqrt(math.pi) * V[:, 0] ** 2

    return x, w
