# core/ev_conditional.py
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from t import chebyshev_T as T            # core.T.py
from wage_and_variance import get_tilde_sigma_prime_corrparam


def compute_cond_EV_CARA(
    state_vec: NDArray,              # (7,)
    pm: NDArray, pj: NDArray,        # 5×5 Gauss–Hermite nodes (meshgrid form)
    wm: NDArray, wj: NDArray,        #   … corresponding weights
    phi_new: NDArray,                # (N_g,) Chebyshev coeffs of V̂
    bounds: NDArray,                 # (7,2) lower/upper bounds
    S,                               # SmolyakStruct from smolyak_step1
    params,
) -> NDArray:
    """
    Return expEV(j) for j=1..O   (shape: (O,))
    """
    O = params.O
    psi, bar_a = params.psi, params.bar_a
    n_comb = pm.size
    weights_prod = (wm * wj).ravel()      # (25,)

    sqrt2 = math.sqrt(2.0)
    disc  = params.beta * math.pi ** (-params.S / 2)

    # unpack current state for speed
    theta_cur = state_vec[:2]             # θ_C, θ_M
    tau_cur   = state_vec[5:7]            # τ_C, τ_M

    expEV = np.empty(O)

    for j in range(O):
        # ---------- 1) replicate state -------------------------------------
        s_prime = np.tile(state_vec[:, None], (1, n_comb))  # 7 × 25

        # ---------- 2) add tenure increment --------------------------------
        s_prime[5] += params.b_mat[j, 0]    # τ_C
        s_prime[6] += params.b_mat[j, 1]    # τ_M

        # ---------- 3–4) posterior covariance & moments --------------------
        Sigma_post = get_tilde_sigma_prime_corrparam(state_vec, j, params)
        sigC, sigM = math.sqrt(Sigma_post[0, 0]), math.sqrt(Sigma_post[1, 1])
        rho_prime  = Sigma_post[0, 1] / (sigC * sigM + 1e-12)

        s_prime[2], s_prime[3], s_prime[4] = sigC, sigM, rho_prime

        # ---------- 5) skill expansion via GH nodes ------------------------
        try:
            U = np.linalg.cholesky(Sigma_post)
        except np.linalg.LinAlgError:
            diag = np.clip(np.diag(Sigma_post), 1e-8, None)
            np.fill_diagonal(Sigma_post, diag)
            U = np.linalg.cholesky(Sigma_post)

        GH = np.vstack((pm.ravel(), pj.ravel()))        # 2 × 25
        s_prime[:2] = theta_cur[:, None] + sqrt2 * (U @ GH)

        # ---------- 6) Chebyshev basis evaluation --------------------------
        y_j = (2 * s_prime - (bounds[:, [0]] + bounds[:, [1]])) / (
            bounds[:, [1]] - bounds[:, [0]]
        )
        np.clip(y_j, -1.0, 1.0, out=y_j)
        T_mat = T(y_j, S)                               # (Nbasis × 25)

        # ---------- 7) death probability -----------------------------------
        experience = s_prime[5, 0] + s_prime[6, 0]
        death_prob = 1.0 / (1.0 + math.exp(-psi * (experience - bar_a)))

        # ---------- 8) expected continuation value -------------------------
        phi_vals = phi_new @ T_mat                      # (25,)
        expEV[j] = (1.0 - death_prob) * disc * np.sum(weights_prod * phi_vals)

    return expEV  # shape (O,)
