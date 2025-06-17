from numba import njit
import math
import numpy as np
from numpy.typing import NDArray

from t_numba import _chebyshev_T_nb
from wage_and_variance_numba import get_tilde_sigma_prime_corrparam_nb  # ← same import


# -------------------------------------------------------------------------
#  Expected continuation value  E[V | choose j]   (CARA utility version)
# -------------------------------------------------------------------------
@njit(cache=True)
def compute_cond_EV_CARA_nb(
    state_vec: NDArray,        # (7,)
    pm: NDArray, pj: NDArray,  # GH nodes   (K×K)
    wm: NDArray, wj: NDArray,  # GH weights (K×K)
    phi_new: NDArray,          # (N_basis,)
    bounds: NDArray,           # (7,2)
    l_mat: NDArray,            # (nbasis,d)  int64
    max_order: int,            # scalar
    params                     # ModelParams jit-class
) -> NDArray:

    O        = params.O
    psi      = params.psi
    bar_a    = params.bar_a
    gamma    = params.gamma      # only used for discount factor
    K        = pm.size
    weight_  = (wm * wj).ravel()
    disc     = params.beta * math.pi ** (-params.S / 2)
    theta_cur = state_vec[:2]

    # pre-stack GH nodes
    GH = np.empty((2, K), dtype=np.float64)
    GH[0, :] = pm.ravel()
    GH[1, :] = pj.ravel()
    sqrt2 = math.sqrt(2.0)

    expEV = np.empty(O, dtype=np.float64)

    for j in range(O):
        # 1) posterior Σ′  (new signature ➜ pass two extra arrays)
        Sigma_post = get_tilde_sigma_prime_corrparam_nb(
            state_vec,
            j,
            params.lambda_mat,    # <-- NEW ARG
            params.varsigma       # <-- NEW ARG
        )
        Sigma_post[0, 0] += 1e-10
        Sigma_post[1, 1] += 1e-10
        U = np.linalg.cholesky(Sigma_post)

        # 2-5) same as before  … -------------------------------------------------
        s_prime = np.empty((7, K), dtype=np.float64)
        for k in range(7):
            s_prime[k, :] = state_vec[k]
        s_prime[5, :] += params.b_mat[j, 0]
        s_prime[6, :] += params.b_mat[j, 1]

        sigC = math.sqrt(Sigma_post[0, 0])
        sigM = math.sqrt(Sigma_post[1, 1])
        rhoP = Sigma_post[0, 1] / (sigC * sigM + 1e-12)
        s_prime[2, :], s_prime[3, :], s_prime[4, :] = sigC, sigM, rhoP

        s_prime[:2, :] = theta_cur[:, None] + sqrt2 * (U @ GH)

        # 6) Chebyshev basis
        y = (2.0 * s_prime
             - (bounds[:, 0][:, None] + bounds[:, 1][:, None])) \
            / (bounds[:, 1][:, None] - bounds[:, 0][:, None])
        y = np.minimum(np.maximum(y, -1.0), 1.0)

        T_mat = _chebyshev_T_nb(y.T, l_mat, max_order)

        # 7-8) continuation value
        experience = s_prime[5, 0] + s_prime[6, 0]
        death_prob = 1.0 / (1.0 + math.exp(-psi * (experience - bar_a)))

        phi_vals = phi_new @ T_mat
        expEV[j] = (1.0 - death_prob) * disc * np.sum(weight_ * phi_vals)

    return expEV
