"""dynamic_occupational_choice_model_numba.py
────────────────────────────────────────────────────────
Numba‑accelerated EV solver (parallel, nopython) **for the
second‑step NFXP routine**.

Free parameters (passed in each NFXP iteration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **gamma** (γ)  – CARA risk‑aversion coefficient
* **eta_vec** (η) – non‑pecuniary reward vector (length = O)
* **varrho** (ϱ)  – scale of the Type‑I extreme‑value shock

All wage/variance primitives come from the first‑step EM and live in
`ModelParams`.  The solver infers the number of occupations **O** from
`beta_vec` (and checks that `eta_vec` has the same length).
"""
from __future__ import annotations
import math, time, sys, pathlib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit, prange

# ──────────────────────────────────────────────
#  Repository path & imports
# ──────────────────────────────────────────────
_THIS_FILE = pathlib.Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smolyak_step_1 import smolyak_step1, SmolyakStruct
from quadrature import gauss_hermite
from t_numba import _chebyshev_T_nb, chebyshev_T
from wage_and_variance_numba import (
    get_expected_wage_CARA_nb,
    get_tilde_sigma_prime_corrparam_nb,
)


# ──────────────────────────────────────────────
#  Parameters (fixed across outer‑loop calls)
# ──────────────────────────────────────────────
@dataclass
class ModelParams:
    beta_vec: np.ndarray | None = None  # (O, 2)
    lambda_mat: np.ndarray | None = None  # (O, 2)
    varsigma: np.ndarray | None = None  # (O,)
    b_mat: np.ndarray | None = None  # (O, 2)

    beta_disc: float = 0.9
    psi: float = 2.0
    bar_a: float = 40.0
    quad_points: int = 5

    state_dim: int = 7  # θ_C, θ_M, σ_C, σ_M, ρ, τ_C, τ_M

    def ensure_defaults(self):
        if self.beta_vec is None:
            self.beta_vec = np.array([[0.0, 0.0]])
        O = self.beta_vec.shape[0]
        if self.lambda_mat is None:
            self.lambda_mat = np.full((O, 2), 0.5)
        if self.varsigma is None:
            self.varsigma = np.full(O, 0.5)
        if self.b_mat is None:
            self.b_mat = np.zeros((O, 2))


# ──────────────────────────────────────────────
#  Grid construction
# ──────────────────────────────────────────────

def build_grid(p: ModelParams):
    d = p.state_dim
    bounds = np.zeros((d, 2))
    bounds[0:2] = [-2, 5]  # θ_C, θ_M
    bounds[2:4] = [0.001, 1.0]  # σ_C, σ_M
    bounds[4] = [-0.95, 0.95]  # ρ
    bounds[5:7] = [0, 50]  # τ_C, τ_M

    mu = np.full(d, 3)
    x, S = smolyak_step1(d, mu, bounds[:, 0], bounds[:, 1])
    y = (2 * x - (bounds[:, [0]] + bounds[:, [1]])) / (bounds[:, [1]] - bounds[:, [0]])
    TT = chebyshev_T(y, S)
    return bounds, x, TT, S


# ──────────────────────────────────────────────
#  Numba helpers
# ──────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def robust_cholesky(A):
    return np.linalg.cholesky(A + 1e-10 * np.eye(A.shape[0]))


@njit(cache=True, fastmath=True)
def log_sum_exp(v, varrho):
    m = np.max(v)
    return m + varrho * math.log(np.sum(np.exp((v - m) / varrho)))


# ──────────────────────────────────────────────
#  Single‑point kernel
# ──────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _eval_point_nb(idx, x, bounds,
                   l_mat, max_order,
                   phi, beta_vec, lambda_mat, b_mat, varsigma,
                   eta_vec, gh_grid, wt_prod,
                   beta_disc, varrho, gamma, psi, bar_a, O, S):
    θ_C, θ_M, σ_C, σ_M, ρ, τ_C, τ_M = x[:, idx]
    state_vec = np.array([θ_C, θ_M, σ_C, σ_M, ρ, τ_C, τ_M])

    # Stage utility
    wage_util = get_expected_wage_CARA_nb(state_vec, lambda_mat, beta_vec, varsigma, gamma)
    util_vec = wage_util + eta_vec

    # Continuation value
    nComb = wt_prod.size
    sqrt2 = math.sqrt(2.0)
    beta_pi_term = beta_disc * math.pi ** (-S / 2)

    exp_EV = np.zeros(O)
    for j in range(O):
        Σ_p = get_tilde_sigma_prime_corrparam_nb(state_vec, j, lambda_mat, varsigma)
        U = robust_cholesky(Σ_p)
        σC_next, σM_next = math.sqrt(Σ_p[0, 0]), math.sqrt(Σ_p[1, 1])
        ρ_next = Σ_p[0, 1] / (σC_next * σM_next)

        s_prime = np.empty((7, nComb))
        s_prime[:] = state_vec.reshape(7, 1)
        b_j = b_mat[j]
        s_prime[2], s_prime[3] = σC_next, σM_next
        s_prime[4] = ρ_next
        s_prime[5] = τ_C + b_j[0]
        s_prime[6] = τ_M + b_j[1]
        s_prime[:2] = sqrt2 * (U @ gh_grid) + state_vec[:2].reshape(2, 1)

        lo = bounds[:, 0].copy().reshape(-1, 1)   # contiguous
        hi = bounds[:, 1].copy().reshape(-1, 1)
        y_prime = (2.0 * s_prime - (lo + hi)) / (hi - lo)
        np.clip(y_prime, -1.0, 1.0, out=y_prime)
        T_prime = _chebyshev_T_nb(y_prime.T, l_mat, max_order)
        φ_vals = phi @ T_prime

        experience = s_prime[5, 0] + s_prime[6, 0]
        death_prob = 1.0 / (1.0 + math.exp(-psi * (experience - bar_a)))
        exp_EV[j] = (1.0 - death_prob) * beta_pi_term * (wt_prod @ φ_vals)

    return log_sum_exp(util_vec + exp_EV, varrho)


# ──────────────────────────────────────────────
#  Vectorised evaluation
# ──────────────────────────────────────────────
@njit(parallel=True, fastmath=True, cache=True)
def evaluate_all_nb(x, bounds, l_mat, max_order,
                    phi, beta_vec, lambda_mat, b_mat, varsigma,
                    eta_vec, gh_grid, wt_prod,
                    beta_disc, varrho, gamma, psi, bar_a, O, S):
    N_g = x.shape[1]
    EV = np.empty(N_g)
    for idx in prange(N_g):
        EV[idx] = _eval_point_nb(idx, x, bounds, l_mat, max_order,
                                 phi, beta_vec, lambda_mat, b_mat, varsigma,
                                 eta_vec, gh_grid, wt_prod,
                                 beta_disc, varrho, gamma, psi, bar_a, O, S)
    return EV


# ──────────────────────────────────────────────
#  Public solver
# ──────────────────────────────────────────────

def solve_model(p: ModelParams,
                gamma: float,
                eta_vec: np.ndarray,
                varrho: float,
                tol: float = 1e-8,
                max_iter: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fixed‑point coefficients φ and EV grid for current θ."""
    p.ensure_defaults()
    O, S = p.beta_vec.shape[0], p.beta_vec.shape[1]
    assert eta_vec.shape == (O,), "eta_vec length mismatch with occupations"

    bounds, x, TT, S_struct = build_grid(p)
    l_mat = S_struct.l.astype(np.int64)
    max_order = int(np.max(S_struct.M_mup1))

    N_g = x.shape[1]
    phi = np.full(N_g, 1e-3)

    gh_pts, gh_wts = gauss_hermite(p.quad_points)
    pm, pj = np.meshgrid(gh_pts, gh_pts, indexing="ij")
    gh_grid = np.vstack((pm.ravel(), pj.ravel())).astype(np.float64)
    wt_prod = np.outer(gh_wts, gh_wts).ravel().astype(np.float64)

    it, err = 0, 1.0
    scalars = (p.beta_disc, varrho, gamma, p.psi, p.bar_a, O, S)

    # start total timer
    t_start = time.perf_counter()

    it = 0
    err = np.inf
    while it < max_iter and err > tol:
        # start iteration timer
        # t0 = time.perf_counter()

        EV = evaluate_all_nb(
            x, bounds, l_mat, max_order,
            phi, p.beta_vec, p.lambda_mat, p.b_mat, p.varsigma,
            eta_vec, gh_grid, wt_prod,
            *scalars
        )
        phi_new = np.linalg.lstsq(TT.T, EV, rcond=None)[0]
        err = np.linalg.norm(phi_new - phi)
        phi = phi_new
        it += 1

        # per‐iteration timing
        # print(f"Iter {it:3d} | err = {err:9.2e} | {time.perf_counter() - t0:5.2f}s")

    # total timing
    total_time = time.perf_counter() - t_start
    print(f"Completed {it} iterations in {total_time:.2f}s (avg {total_time/it:.2f}s/iter)")

    return phi, EV
