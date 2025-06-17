"""
forward_simulation_vectorized.py
================================
Loop-free forward simulation for the multidimensional-skills model.

Key points
----------
* No per-worker Python loops – all operations are batched with NumPy.
* Compatible with your existing scalar helpers; see the four “*_batch”
  wrappers below.
* Ready for optional Numba: uncomment the decorators at the very bottom
  to JIT-compile the heavy kernels in one shot.
"""

from __future__ import annotations

import math, time, warnings, sys, pathlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------------------------------------------------------------------
#  Path bootstrap so imports work without installation
# ---------------------------------------------------------------------------
_THIS = pathlib.Path(__file__).resolve()
SRC_DIR = _THIS.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
#  Core helpers
# ---------------------------------------------------------------------------
from smolyak_step_1 import smolyak_step1, SmolyakStruct
from quadrature    import gauss_hermite
from t             import chebyshev_T as T
from wage_and_variance import (
    get_expected_wage_CARA,
    get_tilde_sigma_prime_corrparam,
)
from ev_conditional import compute_cond_EV_CARA
# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
@dataclass
class ModelParams:
    O: int = 3
    S: int = 2
    d: int = 7
    beta: float = 0.9
    varrho: float = 0.01
    gamma: float = 1.0
    varsigma: NDArray = np.array([0.7, 0.7, 0.7])
    psi: float = 2.0
    bar_a: float = 40.0
    quad_points: int = 5

    beta_vec: NDArray | None = None
    lambda_mat: NDArray | None = None
    b_mat: NDArray | None = None

    def __post_init__(self):
        self.beta_vec = np.column_stack((np.array([0.0, 0.025, 0.05]),
                                         np.array([0.05, 0.025, 0.0])))
        self.lambda_mat = np.array([[0.2, 0.9], [0.6, 0.6], [1.0, 0.3]])
        self.b_mat = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])


# --------------------------------------------------------------------------
#  Vector-friendly wrappers around your scalar helpers
#  (They just loop once internally; refactor later for even more speed.)
# --------------------------------------------------------------------------

def get_expected_wage_CARA_batch(states: NDArray, params) -> NDArray:
    """
    Parameters
    ----------
    states : (7, n_workers) belief state for all workers
    Returns
    -------
    w_det  : (n_workers, params.O) deterministic wage component
    """
    return np.stack(
        [get_expected_wage_CARA(states[:, i], params)[0]      # keep only w_det
         for i in range(states.shape[1])],
        axis=0
    )


def compute_cond_EV_CARA_batch(states: NDArray,
                               pm, pj, wm, wj,
                               phi, bounds, S, params) -> NDArray:
    """
    Same contract as the scalar `compute_cond_EV_CARA`, batched over workers.
    """
    return np.stack(
        [compute_cond_EV_CARA(states[:, i], pm, pj, wm, wj, phi, bounds, S, params)
         for i in range(states.shape[1])],
        axis=0
    )


def get_tilde_sigma_prime_corrparam_batch(states: NDArray,
                                          j_idx: NDArray,
                                          params) -> NDArray:
    """
    Batched version of `get_tilde_sigma_prime_corrparam`.

    Parameters
    ----------
    states : (7, n_workers)
    j_idx  : (n_workers,) chosen occupation index (0-based)
    Returns
    -------
    Sigma_post : (n_workers, 2, 2)
    """
    return np.stack(
        [get_tilde_sigma_prime_corrparam(states[:, i], j_idx[i], params)
         for i in range(states.shape[1])],
        axis=0
    )


def update_posterior_mean_batch(theta_old: NDArray,      # (n, 2)
                                Sigma_old: NDArray,      # (n, 2, 2)
                                Sigma_new: NDArray,      # (n, 2, 2)
                                lam: NDArray,            # (n, 2)
                                signal: NDArray,         # (n,)
                                sigma_eps_sq: NDArray) -> NDArray:
    """
    Vectorised Kalman update (closed form).
    """
    # Kalman gain
    K   = (Sigma_old @ lam[..., None])[:, :, 0]          # (n, 2)
    Szz = (lam * K).sum(1) + sigma_eps_sq               # (n,)
    gain = K / Szz[:, None]                             # (n, 2)

    # residual needs shape (n, 1) to broadcast with (n, 2)
    resid = signal - np.einsum('ij,ij->i', lam, theta_old)   # (n,)
    return theta_old + gain * resid[:, None]                # (n, 2)
# ---------------------------------------------------------------------------
#  Grid helper
# ---------------------------------------------------------------------------


def build_grid(params: ModelParams):
    bounds = np.zeros((params.d, 2))
    bounds[0:2] = np.array([-2, 5])
    bounds[2:4] = np.array([0.001, 1.0])
    bounds[4] = np.array([-0.95, 0.95])
    bounds[5:7] = np.array([0, 50])
    order = np.full(params.d, 3)
    _, S = smolyak_step1(params.d, order, bounds[:, 0], bounds[:, 1])
    return bounds, S
# --------------------------------------------------------------------------
#  Main simulation
# --------------------------------------------------------------------------

def forward_simulation_vec(phi: NDArray,
                           params, *,
                           P: int = 150,
                           n_workers: int = 300):
    """
    Fully vectorised forward simulation.

    Returns
    -------
    occ   : (n_workers, P)      chosen occupations (1-based)
    xst   : (7, n_workers, P)   belief paths
    wages : (n_workers, P)      realised wages
    death : (n_workers, P)      indicator for “life-cycle reset”
    """
    # 0. Pre-computed grids / constants -----------------------------------
    bounds, S = build_grid(params)

    x0 = np.array([0.6, 0.9, 0.6, 0.6, 0.35, 0.0, 0.0])          # (7,)

    lam_mat  = params.lambda_mat                                 # (O, 2)
    beta_mat = params.beta_vec                                   # (O, 2)
    b_mat    = params.b_mat                                      # (O, 2)
    eps_std  = np.asarray(params.varsigma) ** 0.5                # (O,)

    gh_pts, gh_wts = gauss_hermite(params.quad_points)
    pm, pj = np.meshgrid(gh_pts, gh_pts, indexing="ij")
    wm, wj = np.meshgrid(gh_wts, gh_wts, indexing="ij")

    # 1. Allocate result arrays ------------------------------------------
    occ   = np.zeros((n_workers, P), dtype=np.int8)
    xst   = np.empty((7, n_workers, P))
    xst[:, :, 0] = x0[:, None]
    wages = np.zeros((n_workers, P))
    death = np.zeros((n_workers, P), dtype=np.int8)

    # 2. Draw true (latent) skills ---------------------------------------
    Sigma0 = np.array([[0.6**2, 0.35*0.6*0.6],
                       [0.35*0.6*0.6, 0.6**2]])
    theta_true = np.random.multivariate_normal([0.6, 0.9],
                                               Sigma0, n_workers).T  # (2, n)

    # 3. --- Period 0 choice ---------------------------------------------
    w_det0  = get_expected_wage_CARA(x0, params)[0]               # (O,)
    expEV0  = compute_cond_EV_CARA(x0, pm, pj, wm, wj,
                                   phi, bounds, S, params)        # (O,)
    xi0     = params.varrho * (-np.log(-np.log(
                 np.random.rand(n_workers, params.O))))
    occ[:, 0] = np.argmax(w_det0 + expEV0 + xi0, axis=1) + 1
    j_idx = occ[:, 0] - 1                                         # (n,)

    # 3b. Period-0 wage draw & posterior update --------------------------
    lam    = lam_mat[j_idx]                                       # (n, 2)
    beta   = beta_mat[j_idx]                                      # (n, 2)
    mu0    = np.einsum('ij,ij->i', lam, theta_true.T) \
           + np.dot(beta, x0[5:7])                                # (n,)
    wages[:, 0] = mu0 + eps_std[j_idx] * np.random.randn(n_workers)

    signal = mu0 - np.dot(beta, x0[5:7])
    Sigma_post0 = get_tilde_sigma_prime_corrparam_batch(
        x0[:, None].repeat(n_workers, axis=1),   #  ← fixed
        j_idx,
        params
    )
    theta_post0 = update_posterior_mean_batch(
        np.repeat(x0[:2][None, :], n_workers, axis=0),
        np.repeat(Sigma0[None, :, :], n_workers, axis=0),
        Sigma_post0, lam, signal, eps_std[j_idx]**2
    )

    sigC0 = np.sqrt(Sigma_post0[:, 0, 0])
    sigM0 = np.sqrt(Sigma_post0[:, 1, 1])
    rho0  = Sigma_post0[:, 0, 1] / (sigC0 * sigM0)
    tau1  = x0[5:7] + b_mat[j_idx]                                # (n, 2)

    xst[:, :, 1] = np.vstack([
        theta_post0.T,
        sigC0, sigM0, rho0,
        tau1.T
    ])

    # 4. --- Main loop ----------------------------------------------------
    for t in range(1, P-1):
        st = xst[:, :, t]                                         # (7, n)

        # choice ---------------------------------------------------------
        w_det  = get_expected_wage_CARA_batch(st, params)         # (n, O)
        expEV  = compute_cond_EV_CARA_batch(st, pm, pj, wm, wj,
                                            phi, bounds, S, params)  # (n, O)
        xi     = params.varrho * (-np.log(-np.log(
                    np.random.rand(n_workers, params.O))))
        occ[:, t] = np.argmax(w_det + expEV + xi, axis=1) + 1
        j_idx = occ[:, t] - 1                                     # (n,)

        # wage draw ------------------------------------------------------
        lam   = lam_mat[j_idx]
        beta  = beta_mat[j_idx]
        mu    = np.einsum('ij,ij->i', lam, theta_true.T) \
              + np.einsum('ij,ij->i', beta, st[5:7].T)
        wages[:, t] = mu + eps_std[j_idx] * np.random.randn(n_workers)
        signal = mu - np.einsum('ij,ij->i', beta, st[5:7].T)

        # posterior update ----------------------------------------------
        Sigma_old          = np.empty((n_workers, 2, 2))
        Sigma_old[:, 0, 0] = st[2]**2
        Sigma_old[:, 1, 1] = st[3]**2
        Sigma_old[:, 0, 1] = Sigma_old[:, 1, 0] = st[4]*st[2]*st[3]

        Sigma_post = get_tilde_sigma_prime_corrparam_batch(st, j_idx, params)

        theta_post         = update_posterior_mean_batch(
                                st[:2].T, Sigma_old, Sigma_post,
                                lam, signal, eps_std[j_idx]**2
                             )

        sigC  = np.sqrt(Sigma_post[:, 0, 0])
        sigM  = np.sqrt(Sigma_post[:, 1, 1])
        rho   = Sigma_post[:, 0, 1] / (sigC * sigM)
        tau   = st[5:7].T + b_mat[j_idx]                          # (n, 2)

        xst[:, :, t+1] = np.vstack([
            theta_post.T,
            sigC, sigM, rho,
            tau.T
        ])

        # life-cycle reset (“death”) ------------------------------------
        p_die            = 1 / (1 + np.exp(-params.psi * (tau.sum(1) - params.bar_a)))
        died             = np.random.rand(n_workers) < p_die
        xst[:, died, t+1] = x0[:, None]
        death[died, t]    = 1

        share = (occ[:, t][:, None] == np.arange(1, params.O+1)).mean(0)
        print(f"t={t+1} | shares {np.round(share, 3).tolist()}")

    return occ, xst, wages, death


# --------------------------------------------------------------------------
#  Optional: uncomment to JIT-compile the heavy parts with Numba
# --------------------------------------------------------------------------
#
# from numba import njit, prange
#
# get_expected_wage_CARA_batch       = njit(parallel=True, fastmath=True)(get_expected_wage_CARA_batch)
# compute_cond_EV_CARA_batch         = njit(parallel=True, fastmath=True)(compute_cond_EV_CARA_batch)
# get_tilde_sigma_prime_corrparam_batch = njit(parallel=True, fastmath=True)(get_tilde_sigma_prime_corrparam_batch)
# update_posterior_mean_batch        = njit(parallel=True, fastmath=True)(update_posterior_mean_batch)
# forward_simulation_vec             = njit(parallel=True, fastmath=True)(forward_simulation_vec)
#
# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
from pathlib import Path
def main(P: int, n_workers: int, seed: int):
    # 1.  Reproducibility
    np.random.seed(seed)

    # 2.  Load fixed-point coefficients ϕ (or initialise zeros)
    fp_file = Path("results/fixed_point_estimates.npz")
    if fp_file.exists():
        fp_data = np.load(fp_file)
        phi     = fp_data["phi"]
        print(f"✔  Loaded φ from {fp_file}")
    else:
        raise FileNotFoundError(
            "Could not find results/fixed_point_estimates.npz – "
            "run the fixed-point solver first.")

    # 3.  Model primitives
    params = ModelParams()

    # 4.  Run simulation
    print(f"⇢  Simulating {n_workers:,} workers for {P} periods ...")
    occ, xst, wages, death = forward_simulation_vec(
        phi, params, P=P, n_workers=n_workers
    )
    print("✔  Simulation finished")

    # 5.  Persist outputs
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"forward_sim_out_P{P}_N{n_workers}.npz"
    np.savez_compressed(out_file,
                        occ=occ, xst=xst, wages=wages, death=death)
    print(f"✔  Saved outputs to {out_file}")


# ---------------------------------------------------------------------------
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the forward simulation in vectorised mode."
    )
    parser.add_argument("--P", type=int, default=150,
                        help="Number of periods to simulate (default: 150)")
    parser.add_argument("--n_workers", type=int, default=300,
                        help="Number of synthetic workers (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # override from CLI if you like; hard-coded here per your request
    args.n_workers = 300

    t0 = time.perf_counter()                  #  ← start timer
    main(args.P, args.n_workers, args.seed)
    elapsed = time.perf_counter() - t0        #  ← stop timer

    print(f"🏁  Completed in {elapsed:,.2f} seconds")
