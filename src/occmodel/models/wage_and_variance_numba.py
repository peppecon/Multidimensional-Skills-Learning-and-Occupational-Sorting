# wage_and_variance_numba.py
# --------------------------------------------------------------
from __future__ import annotations
import math, numpy as np
from numba import njit
from numpy.typing import NDArray


# --------------------------------------------------------------
# 1. Expected utility (CARA)
# --------------------------------------------------------------
@njit(cache=True, fastmath=True)
def get_expected_wage_CARA_nb(state_vec,
                              lambda_mat, beta_mat,
                              varsigma, gamma):

    theta   = state_vec[:2]           # (2,)
    tau     = state_vec[5:7]          # (2,)

    sigmaC, sigmaM, rho = state_vec[2], state_vec[3], state_vec[4]
    Sigma = np.array([[sigmaC*sigmaC,      rho*sigmaC*sigmaM],
                      [rho*sigmaC*sigmaM,  sigmaM*sigmaM]])

    mu  = lambda_mat @ theta + beta_mat @ tau        # (O,)
    # ----------  only this block changed  ----------
    tmp = (lambda_mat @ Sigma) * lambda_mat          # O×2
    var = tmp.sum(axis=1) + varsigma*varsigma        # (O,)
    # ----------------------------------------------
    return -np.exp(-gamma*mu + 0.5*gamma*gamma*var)  # (O,)


# --------------------------------------------------------------
# 2. Posterior covariance update (Kalman)
# --------------------------------------------------------------
@njit(cache=True, fastmath=True)
def get_tilde_sigma_prime_corrparam_nb(state_vec: NDArray[np.float64],
                                       j:         int,
                                       lambda_mat: NDArray[np.float64],
                                       varsigma:   NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return Σ' (2×2) after observing occupation-j wage.
    """
    sigmaC, sigmaM, rho = state_vec[2], state_vec[3], state_vec[4]

    Sigma = np.array([[sigmaC*sigmaC,        rho*sigmaC*sigmaM],
                      [rho*sigmaC*sigmaM,    sigmaM*sigmaM]])

    lam_j   = lambda_mat[j]                  # (2,)
    sig2_e  = varsigma[j] ** 2               # scalar

    # Information-form update: Σ' = (Σ⁻¹ + lam lamᵀ / σ²_ε)⁻¹
    inv_S   = np.linalg.inv(Sigma)
    inv_new = inv_S + np.outer(lam_j, lam_j) / sig2_e
    Sigma_p = np.linalg.inv(inv_new)

    # Numerical hygiene
    eps = 1e-8
    if Sigma_p[0, 0] < eps: Sigma_p[0, 0] = eps
    if Sigma_p[1, 1] < eps: Sigma_p[1, 1] = eps
    if Sigma_p[0,0]*Sigma_p[1,1] - Sigma_p[0,1]**2 < 1e-12:
        Sigma_p[0,1] = Sigma_p[1,0] = 0.0

    return Sigma_p
