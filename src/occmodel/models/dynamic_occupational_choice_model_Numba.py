"""
dynamic_occupational_choice_model_numba.py
------------------------------------------
Numba-accelerated solver (parallel, nopython) for the dynamic
occupational-choice model.  Requires `numba >= 0.59`.
"""
from __future__ import annotations
import math, time, sys, pathlib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit, prange

# ──────────────────────────────────────────────
#  Local imports (same as before)
# ──────────────────────────────────────────────
_THIS_FILE   = pathlib.Path(__file__).resolve()
OCCMODEL_DIR = _THIS_FILE.parent.parent
if str(OCCMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(OCCMODEL_DIR))

from smolyak_step_1 import smolyak_step1, SmolyakStruct
from quadrature       import gauss_hermite
from t_numba                import _chebyshev_T_nb, chebyshev_T            # ➜ remember to JIT!
from wage_and_variance_numba import (
    get_expected_wage_CARA_nb,                         # ➜ remember to JIT!
    get_tilde_sigma_prime_corrparam_nb,                # ➜ remember to JIT!
)

# ──────────────────────────────────────────────
#  Params
# ──────────────────────────────────────────────
@dataclass()
class ModelParams:
    O: int  = 3
    S: int  = 2
    state_dim: int = 7

    beta_disc: float = 0.9
    varrho:    float = 0.01
    gamma:     float = 1.0

    varsigma:  np.ndarray = np.array([0.7, 0.7, 0.7])

    beta_vec:    np.ndarray = None
    lambda_mat:  np.ndarray = None
    b_mat:       np.ndarray = None

    psi: float = 2.0
    bar_a: float = 40.0

    quad_points: int = 5

    def __post_init__(self):
        beta_C = np.array([0.0, 0.025, 0.05])
        beta_M = np.array([0.05, 0.025, 0.0])
        self.beta_vec   = np.column_stack((beta_C, beta_M))
        self.lambda_mat = np.array([[0.2, 0.9], [0.6, 0.6], [1.0, 0.3]])
        self.b_mat      = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])

# ──────────────────────────────────────────────
#  Grid helpers (unchanged, pure Python)
# ──────────────────────────────────────────────
def build_grid(p: ModelParams) -> Tuple[np.ndarray, SmolyakStruct, np.ndarray]:
    d = p.state_dim
    bounds = np.zeros((d, 2))
    bounds[0:2] = np.array([-2, 5])
    bounds[2:4] = np.array([0.001, 1.0])
    bounds[4]   = np.array([-0.95, 0.95])
    bounds[5:7] = np.array([0, 50])

    mu = np.full(d, 3)
    x, S = smolyak_step1(d, mu, bounds[:, 0], bounds[:, 1])
    y    = (2*x - (bounds[:,[0]]+bounds[:,[1]])) / (bounds[:,[1]]-bounds[:,[0]])
    TT   = chebyshev_T(y, S)
    return bounds, x, TT, S


# ──────────────────────────────────────────────
#  Numba helpers
# ──────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def robust_cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky with fixed jitter (Numba-safe)."""
    jitter = 1e-10
    return np.linalg.cholesky(A + jitter*np.eye(A.shape[0]))


@njit(cache=True, fastmath=True)
def log_sum_exp(v: np.ndarray, varrho: float) -> float:
    m = np.max(v)
    return m + varrho * math.log(np.sum(np.exp((v - m)/varrho)))


# ──────────────────────────────────────────────
#  Core kernel – single collocation point
# ──────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _eval_point_nb(idx, x, bounds,
                   l_mat, max_order,
                   phi, beta_vec, lambda_mat, b_mat, varsigma,
                   gh_grid, wt_prod, scalars):

    # ────────── unpack scalars (all plain floats / ints) ──────────
    beta_disc, varrho, gamma, psi, bar_a, O, S = scalars

    # --------------------------------------------------------------
    # 1.  Current belief state
    theta_C, theta_M, sigmaC, sigmaM, rho_val, tenC, tenM = x[:, idx]
    state_vec = np.array([theta_C, theta_M, sigmaC, sigmaM,
                          rho_val, tenC, tenM])

    # --------------------------------------------------------------
    # 2.  Expected CARA utility   (no params object!)
    w_tilde_vec = get_expected_wage_CARA_nb(state_vec,
                                            lambda_mat,
                                            beta_vec,
                                            varsigma,
                                            gamma)        # ← gamma from scalars
    # Gauss-Hermite tensor grid (pre-vectorised)
    nComb = wt_prod.size
    sqrt2        = math.sqrt(2.0)
    beta_pi_term = beta_disc * math.pi ** (-S/2)

     # ------------------------------------------------------------------
    # inside _eval_point_nb  (everything above is unchanged)
    # ------------------------------------------------------------------
    exp_EV = np.zeros(O)

    for j in range(O):
        # ----- posterior covariance & Cholesky ------------------------
        Sigma_p = get_tilde_sigma_prime_corrparam_nb(
                      state_vec,                # ← current belief state
                      j,                        # ← occupation index
                      lambda_mat,
                      varsigma)
        U_p = robust_cholesky(Sigma_p)

        sigC  = math.sqrt(Sigma_p[0, 0])
        sigM  = math.sqrt(Sigma_p[1, 1])
        rhoP  = Sigma_p[0, 1] / (sigC * sigM)

        # ----- build s′ grid -----------------------------------------
        s_prime = np.empty((7, nComb))
        s_prime[:] = state_vec.reshape(7, 1)

        b_j = b_mat[j]                  # ← index with j, not jOcc
        s_prime[2] = sigC
        s_prime[3] = sigM
        s_prime[4] = rhoP
        s_prime[5] = tenC + b_j[0]
        s_prime[6] = tenM + b_j[1]

        # ---- inside _eval_point_nb ------------------------------------
        s_prime[:2] = sqrt2 * (U_p @ gh_grid) + state_vec[:2].reshape(2, 1)
        lo = bounds[:, 0].copy().reshape(-1, 1)   # contiguous
        hi = bounds[:, 1].copy().reshape(-1, 1)
        y_prime = (2.0 * s_prime - (lo + hi)) / (hi - lo)
        np.clip(y_prime, -1.0, 1.0, out=y_prime)

        T_prime = _chebyshev_T_nb(y_prime.T, l_mat, max_order)   # (nbasis, nComb)
        phi_vals = phi @ T_prime          # 1 × nComb

        experience = s_prime[5, 0] + s_prime[6, 0]
        death_prob = 1.0 / (1.0 + math.exp(-psi * (experience - bar_a)))

        exp_EV[j] = (1.0 - death_prob) * beta_pi_term * (wt_prod @ phi_vals)


    return log_sum_exp(w_tilde_vec + exp_EV, varrho)


# ──────────────────────────────────────────────
#  Vectorised, parallel loop over all points
# ──────────────────────────────────────────────
@njit(parallel=True, fastmath=True, cache=True)
def evaluate_all_nb(x, bounds, l_mat, max_order,
                    phi, beta_vec, lambda_mat, b_mat, varsigma,
                    gh_grid, wt_prod, scalars):

    N_g = x.shape[1]
    EV  = np.empty(N_g)

    for idx in prange(N_g):
        EV[idx] = _eval_point_nb(idx, x, bounds,
                                 l_mat, max_order,
                                 phi, beta_vec, lambda_mat, b_mat, varsigma,
                                 gh_grid, wt_prod, scalars)
    return EV


# ──────────────────────────────────────────────
#  Solver
# ──────────────────────────────────────────────
def solve_model(p: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    bounds, x, TT, S_struct = build_grid(params)

    # NEW: plain arrays for the JIT kernels
    l_mat     = S_struct.l.astype(np.int64)       # (nbasis, d)
    max_order = int(np.max(S_struct.M_mup1))
    N_g   = x.shape[1]
    phi   = np.full(N_g, 1e-3)

    gh_pts, gh_wts = gauss_hermite(p.quad_points)

    # 2-D grid (shape 2×nComb) and weight product (nComb,)
    pm, pj   = np.meshgrid(gh_pts, gh_pts, indexing="ij")
    gh_grid  = np.vstack((pm.ravel(), pj.ravel())).astype(np.float64)
    wt_prod  = np.outer(gh_wts, gh_wts).ravel().astype(np.float64)
    scalars = (p.beta_disc, p.varrho, p.gamma, p.psi,
               p.bar_a, p.O, p.S)

    err, it = 1e9, 0
    while it < 1500 and err > 1e-8:
        t0 = time.perf_counter()

        # Numba-parallel evaluation
         # pass into the parallel kernel
        EV = evaluate_all_nb(
                x, bounds, l_mat, max_order,
                phi, p.beta_vec, p.lambda_mat, p.b_mat, p.varsigma,
                gh_grid, wt_prod, scalars)

        # least-squares update  (BLAS – keep in Python)
        phi_new = np.linalg.lstsq(TT.T, EV, rcond=None)[0]
        err     = np.linalg.norm(phi_new - phi)
        phi     = phi_new
        it     += 1
        print(f"Iter {it:3d} | err = {err:9.2e} | {time.perf_counter()-t0:5.2f}s")

    return phi, EV


# ---------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import time
    t0 = time.perf_counter()          # ── start stopwatch ──

    params = ModelParams()
    phi_star, EV_grid = solve_model(params)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "fixed_point_estimates.npz",
             EV=EV_grid, phi=phi_star)

    dt = time.perf_counter() - t0     # ── elapsed seconds ──
    print(f"✔  Results saved to results/fixed_point_estimates.npz")
    print(f"⏱  Total run-time: {dt:,.1f} s")
