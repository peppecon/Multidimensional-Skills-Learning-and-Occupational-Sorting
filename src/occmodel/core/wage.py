"""wage_and_variance.py – Wage utility & posterior variance update

Contains two key functions used in the dynamic occupational choice model:

1. **get_expected_wage_CARA** – expected CARA utility and human‑capital term.
2. **get_tilde_sigma_prime_corrparam** – Bayesian update of the 2×2 belief
   covariance matrix when observing a noisy wage signal in occupation *j*.

Both reflect the original MATLAB code and vectorised where possible for
speed.  Save this file under ``src/occmodel/core/wage_and_variance.py`` and
update your imports accordingly:

```python
from occmodel.core.wage_and_variance import (
    get_expected_wage_CARA,
    get_tilde_sigma_prime_corrparam,
)
```
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Expected CARA utility                                                      ──
# ---------------------------------------------------------------------------

def get_expected_wage_CARA(state_vec: NDArray[np.float64], params) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:  # noqa: N802
    """Compute expected CARA utility and human‑capital term for each occupation.

    Parameters
    ----------
    state_vec : ndarray, shape (7,)
        ``[θ_C, θ_M, σ_C, σ_M, ρ, τ_C, τ_M]``.
    params : object (e.g. ModelParams)
        Requires attributes ``O``, ``gamma``, ``lambda_mat``, ``beta_vec``,
        and ``varsigma``.

    Returns
    -------
    expected_utility_vec : ndarray, shape (O,)
    H_vec : ndarray, shape (O,)
    """
    theta_C, theta_M, sigmaC, sigmaM, rho, tau_C, tau_M = state_vec
    theta = np.array([theta_C, theta_M])
    tau = np.array([tau_C, tau_M])

    Sigma = np.array(
        [
            [sigmaC**2, rho * sigmaC * sigmaM],
            [rho * sigmaC * sigmaM, sigmaM**2],
        ]
    )

    lambda_mat = params.lambda_mat
    beta_mat = params.beta_vec
    sigma_vec = params.varsigma
    gamma = params.gamma
    O = params.O

    mu = lambda_mat @ theta + beta_mat @ tau
    H_vec = beta_mat @ tau

    var = np.einsum("ij,ij->i", lambda_mat @ Sigma, lambda_mat) + sigma_vec**2

    expected_utility_vec = -np.exp(-gamma * mu + 0.5 * gamma**2 * var)

    return expected_utility_vec.reshape(O), H_vec.reshape(O)


# ---------------------------------------------------------------------------
# Posterior covariance update                                               ──
# ---------------------------------------------------------------------------

def get_tilde_sigma_prime_corrparam(state_vec: NDArray[np.float64], j: int, params) -> NDArray[np.float64]:  # noqa: N802
    """Return updated 2×2 belief covariance matrix after observing wage in occ *j*.

    Parameters
    ----------
    state_vec : ndarray, shape (7,)
        Current belief state ``[θ_C, θ_M, σ_C, σ_M, ρ, τ_C, τ_M]``.
    j : int
        Occupation index (0‑based in Python, unlike 1‑based MATLAB).
    params : object
        Needs ``lambda_mat`` (O×2) and ``varsigma`` (O,) attributes.

    Notes
    -----
    Implements the information‑form update:

        Σ' = ( Σ⁻¹ + (1/σ²_ε) λ λᵀ )⁻¹

    Equivalent to Kalman filtering with measurement variance σ²_ε.
    Negative diagonals are clipped to a small positive constant; near‑
    singular covariance matrices are regularised by zeroing the off‑diag.
    """
    sigmaC, sigmaM, rho = state_vec[2], state_vec[3], state_vec[4]

    # Prior Σ
    Sigma = np.array(
        [
            [sigmaC**2, rho * sigmaC * sigmaM],
            [rho * sigmaC * sigmaM, sigmaM**2],
        ]
    )

    lambda_j = params.lambda_mat[j]  # shape (2,)
    varsigma_sq = params.varsigma[j] ** 2

    # Information‑form update: Σ' = (Σ⁻¹ + (1/σ²) λ λᵀ)⁻¹
    Sigma_inv = np.linalg.inv(Sigma)
    outer = np.outer(lambda_j, lambda_j)
    Sigma_new_inv = Sigma_inv + outer / varsigma_sq
    Sigma_prime = np.linalg.inv(Sigma_new_inv)

    # Numerical hygiene
    eps = 1e-8
    Sigma_prime[0, 0] = max(Sigma_prime[0, 0], eps)
    Sigma_prime[1, 1] = max(Sigma_prime[1, 1], eps)

    det_val = Sigma_prime[0, 0] * Sigma_prime[1, 1] - Sigma_prime[0, 1] ** 2
    if det_val < 1e-12:
        Sigma_prime[0, 1] = Sigma_prime[1, 0] = 0.0

    return Sigma_prime
