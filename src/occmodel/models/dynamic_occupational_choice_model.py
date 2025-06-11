"""dynamic_occupational_choice_model.py
================================================
Self‑contained solver for the dynamic occupational‑choice model **without
requiring package installation**.  It works in plain PyCharm “Run File”,
Jupyter `%run`, or CLI execution.

Key design choices
------------------
* **Path bootstrap** – at runtime we prepend the *parent* folder that
  contains the sibling `core/` directory to `sys.path`.  That makes the
  `core` package importable from anywhere.
* **Robust imports** – we try both naming conventions that appear in
  your tree (`smolyak_step1.py` **or** `smolyak_step_1.py`, `T.py` **or**
  `t.py`).  No more `ModuleNotFoundError`.
* **No duplicate import blocks** – one clean set of imports.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor

import sys
import pathlib
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap import path so *core/* is visible                                ──
# ─────────────────────────────────────────────────────────────────────────────
_THIS_FILE = pathlib.Path(__file__).resolve()        # …/occmodel/models/…
OCCMODEL_DIR = _THIS_FILE.parent.parent              # …/occmodel
if str(OCCMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(OCCMODEL_DIR))            # prepend once

# ─────────────────────────────────────────────────────────────────────────────
#  Imports – try both naming conventions that exist in your folder            ──
# ─────────────────────────────────────────────────────────────────────────────
from smolyak_step_1 import smolyak_step1  # type: ignore
from smolyak_step_1 import SmolyakStruct                    # type: ignore

from t import chebyshev_T as T  # fallback to t.py (lower‑case)

from quadrature import gauss_hermite
from wage_and_variance import (
    get_expected_wage_CARA,
    get_tilde_sigma_prime_corrparam,
)
# ---------------------------------------------------------------------------
#  Parameter container                                                        ──                                                        ──
# ---------------------------------------------------------------------------
@dataclass()
class ModelParams:
    O: int = 3
    S: int = 2
    state_dim: int = 7

    beta_disc: float = 0.9
    varrho: float = 0.01
    gamma: float = 1.0

    # occupational data (filled below)
    varsigma: np.ndarray = np.array([0.7, 0.7, 0.7])
    beta_vec: np.ndarray = None
    lambda_mat: np.ndarray = None
    b_mat: np.ndarray = None

    # death‑process parameters
    psi: float = 2.0
    bar_a: float = 40.0

    quad_points: int = 5

    def __post_init__(self):
        # returns to tenure
        beta_C = np.array([0.0, 0.025, 0.05])
        beta_M = np.array([0.05, 0.025, 0.0])
        self.beta_vec = np.column_stack((beta_C, beta_M))

        # skill prices
        self.lambda_mat = np.array([[0.2, 0.9], [0.6, 0.6], [1.0, 0.3]])

        # skill accumulation
        self.b_mat = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])


# ---------------------------------------------------------------------------
#  Collocation‑grid helpers                                                   ──
# ---------------------------------------------------------------------------

def build_grid(params: ModelParams) -> Tuple[np.ndarray, SmolyakStruct]:
    """Construct Smolyak grid & auxiliary structure."""
    d = params.state_dim

    # bounds in the same order as MATLAB comments
    bounds = np.zeros((d, 2))
    bounds[0:2] = np.array([-2, 5])            # θ_C, θ_M
    bounds[2:4] = np.array([0.001, 1.0])       # σ_C, σ_M
    bounds[4] = np.array([-0.95, 0.95])        # ρ
    bounds[5:7] = np.array([0, 50])            # tenure

    mu = np.full(d, 3)  # Smolyak accuracy levels (order‑1)
    x, S = smolyak_step1(d, mu, bounds[:, 0], bounds[:, 1])

    # y in [−1,1]^d  (columns)
    y = (2 * x - (bounds[:, [0]] + bounds[:, [1]])) / (bounds[:, [1]] - bounds[:, [0]])
    TT = T(y, S)
    return bounds, x, y, TT, S


# ---------------------------------------------------------------------------
#  Single collocation evaluation (for parallelism)                            ──
# ---------------------------------------------------------------------------

def _evaluate_point(idx: int, x: np.ndarray, bounds: np.ndarray, S: SmolyakStruct,
                    phi: np.ndarray, params: ModelParams,
                    gh_points: np.ndarray, gh_weights: np.ndarray) -> float:
    """Return expected value at collocation index *idx* (CPU intensive)."""
    # unpack state
    theta_C, theta_M, sigmaC, sigmaM, rho_val, tenC, tenM = x[:, idx]
    state_vec = np.array([theta_C, theta_M, sigmaC, sigmaM, rho_val, tenC, tenM])

    # expected wages/utilities (vectorised function returns tuple)
    w_tilde_vec, _ = get_expected_wage_CARA(state_vec, params)

    # preview posterior covariances & Cholesky factors
    U_prime = []
    tilde_sigma_prime_list = []
    for jOcc in range(params.O):
        Sigma_p = get_tilde_sigma_prime_corrparam(state_vec, jOcc, params)
        tilde_sigma_prime_list.append(Sigma_p)
        try:
            U_prime.append(np.linalg.cholesky(Sigma_p))
        except np.linalg.LinAlgError:
            # basic fix
            diag = np.clip(np.diag(Sigma_p), 1e-8, None)
            np.fill_diagonal(Sigma_p, diag)
            U_prime.append(np.linalg.cholesky(Sigma_p))

    # quadrature setup (5×5) – pre‑flatten
    pm, pj = np.meshgrid(gh_points, gh_points, indexing="ij")
    wm, wj = np.meshgrid(gh_weights, gh_weights, indexing="ij")
    weights_prod = (wm * wj).ravel()
    nComb = weights_prod.size

    beta_pi_term = params.beta_disc * math.pi ** (-params.S / 2)
    sqrt2 = math.sqrt(2.0)

    exp_EV = np.zeros(params.O)
    for jOcc in range(params.O):
        # build s′ grid (7×nComb)
        s_prime = np.tile(state_vec[:, None], (1, nComb))

        sigCprime = math.sqrt(tilde_sigma_prime_list[jOcc][0, 0])
        sigMprime = math.sqrt(tilde_sigma_prime_list[jOcc][1, 1])
        covPrime = tilde_sigma_prime_list[jOcc][0, 1]
        rhoPrime = covPrime / (sigCprime * sigMprime)

        s_prime[2] = sigCprime
        s_prime[3] = sigMprime
        s_prime[4] = rhoPrime

        b_j = params.b_mat[jOcc]
        s_prime[5] = tenC + b_j[0]
        s_prime[6] = tenM + b_j[1]

        skill_mean = state_vec[:2]
        skill_new = sqrt2 * (U_prime[jOcc] @ np.vstack((pm.ravel(), pj.ravel()))) + skill_mean[:, None]
        s_prime[:2] = skill_new

        # evaluate Chebyshev basis at s′
        y_prime = (2 * s_prime - (bounds[:, [0]] + bounds[:, [1]])) / (bounds[:, [1]] - bounds[:, [0]])
        np.clip(y_prime, -1.0, 1.0, out=y_prime)
        T_prime = T(y_prime, S)
        phi_vals = phi @ T_prime

        experience = s_prime[5, 0] + s_prime[6, 0]
        death_prob = 1.0 / (1.0 + math.exp(-params.psi * (experience - params.bar_a)))
        exp_EV[jOcc] = (1 - death_prob) * beta_pi_term * np.sum(weights_prod * phi_vals)

    # log‑sum‑exp
    v_vec = w_tilde_vec + exp_EV
    max_val = np.max(v_vec)
    EV = max_val + params.varrho * math.log(np.sum(np.exp((v_vec - max_val) / params.varrho)))
    return EV


# ---------------------------------------------------------------------------
#  Main solver loop                                                          ──
# ---------------------------------------------------------------------------

def solve_model(params: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    bounds, x, y, TT, S = build_grid(params)
    N_g = x.shape[1]

    phi = np.full(N_g, 0.001)
    err, check, iter_ = 100.0, -10.0, 0

    gh_points, gh_weights = gauss_hermite(params.quad_points)

    while iter_ < 1500 and err > 1e-8 and check < 0:
        t0 = time.perf_counter()
        print(f"Iteration {iter_}")

        # parallel evaluation
        with ProcessPoolExecutor() as ex:
            EV = np.fromiter(
                ex.map(
                    _evaluate_point,
                    range(N_g),
                    [x] * N_g,
                    [bounds] * N_g,
                    [S] * N_g,
                    [phi] * N_g,
                    [params] * N_g,
                    [gh_points] * N_g,
                    [gh_weights] * N_g,
                ),
                dtype=float,
                count=N_g,
            )

        # least‑squares update of phi
        phi_new = np.linalg.lstsq(TT.T, EV, rcond=None)[0]
        err_new = np.linalg.norm(phi_new - phi)
        check = err_new - err
        err = err_new
        phi = phi_new

        iter_ += 1
        print(f"  error = {err:.3e}  |  dt = {time.perf_counter() - t0:.1f}s")

    return phi, EV


# ---------------------------------------------------------------------------
#  CLI entry‑point                                                           ──
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = ModelParams()
    phi_star, EV_grid = solve_model(params)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "fixed_point_estimates.npz", EV=EV_grid, phi=phi_star)
    print("✔  Results saved to results/fixed_point_estimates.npz")
