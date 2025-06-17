"""forward_simulation.py – Forward simulation **plus** U‑shape & hazard plots

Features
--------
1. Loads `phi` (and optionally `EV`) from
   `../results/fixed_point_estimates.npz` if it exists.
2. Runs the forward simulation (defaults: `P=150`, `n_workers=300`).
3. Generates two figures **after** the simulation:
   * **U‑shape** – switching probability by wage‑percentile bin, smoothed
     with LOWESS (`frac=0.20`).  Saved to `fig/u_shape.png`.
   * **Hazard rate** – smoothed hazard of switching vs. experience using
     LOESS (`span = 0.20 · #bins`).  Saved to `fig/hazard_rate.png`.

Run:
    python -m occmodel.models.forward_simulation
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


# ---------------------------------------------------------------------------
#  Posterior mean update helper
# ---------------------------------------------------------------------------

def update_posterior_mean(
    theta_old: NDArray,        # (2,)
    Sigma_old: NDArray,        # 2×2 prior covariance
    Sigma_post: NDArray,       # 2×2 posterior covariance (from Kalman update)
    lam_j: NDArray,            # (2,) loading vector
    signal: float,             # wage signal minus beta·tau
    sigma_eps_sq: float,       # measurement variance σ²_ε
) -> NDArray:
    """Exact Bayesian mean update used in the MATLAB code.

    θ' = Σ_post ( Σ_old⁻¹ θ_old  +  λ · signal / σ²_ε )
    """
    rhs = np.linalg.inv(Sigma_old) @ theta_old + lam_j * (signal / sigma_eps_sq)
    return Sigma_post @ rhs


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

# ---------------------------------------------------------------------------
#  Forward simulation (with mean updates)
# ---------------------------------------------------------------------------


def forward_simulation(phi: NDArray, params: ModelParams, *, P=150, n_workers=300):
    bounds, S = build_grid(params)

    # initial belief state
    x0 = np.array([0.6, 0.9, 0.6, 0.6, 0.35, 0.0, 0.0])

    occ = np.zeros((n_workers, P), dtype=np.int8)
    xst = np.tile(x0[:, None, None], (1, n_workers, P))
    wages = np.zeros((n_workers, P))
    death = np.zeros((n_workers, P), dtype=np.int8)

    # true skills
    Sigma0 = np.array([[0.6**2, 0.35*0.6*0.6], [0.35*0.6*0.6, 0.6**2]])
    theta_true = np.random.multivariate_normal([0.6, 0.9], Sigma0, n_workers).T

    gh_points, gh_weights = gauss_hermite(params.quad_points)
    pm, pj = np.meshgrid(gh_points, gh_points, indexing="ij")
    wm, wj = np.meshgrid(gh_weights, gh_weights, indexing="ij")

    # ---------- t = 0 choice ---------------------------------------------
    w_det, _ = get_expected_wage_CARA(x0, params)
    expEV0 = compute_cond_EV_CARA(x0, pm, pj, wm, wj, phi, bounds, S, params)
    xi0 = params.varrho * (-np.log(-np.log(np.random.rand(n_workers, params.O))))
    occ[:, 0] = (w_det + expEV0 + xi0).argmax(axis=1) + 1

    # update beliefs and draw wage ---------------------------------------
    for i in range(n_workers):
        j = occ[i, 0] - 1
        lam_j, beta_j = params.lambda_mat[j], params.beta_vec[j]
        mu_j = lam_j @ theta_true[:, i] + beta_j @ x0[5:7]
        eps = math.sqrt(params.varsigma[j]) * np.random.randn()
        wages[i, 0] = mu_j + eps
        signal = mu_j - beta_j @ x0[5:7]

        Sigma_post = get_tilde_sigma_prime_corrparam(x0, j, params)
        theta_post = update_posterior_mean(
            theta_old=x0[:2],
            Sigma_old=Sigma0,
            Sigma_post=Sigma_post,
            lam_j=lam_j,
            signal=signal,
            sigma_eps_sq=params.varsigma[j] ** 2,
        )
        sigC, sigM = math.sqrt(Sigma_post[0,0]), math.sqrt(Sigma_post[1,1])
        rhoP = Sigma_post[0,1]/(sigC*sigM)
        xst[:, i, 1] = np.array([theta_post[0], theta_post[1], sigC, sigM, rhoP, * (x0[5:7] + params.b_mat[j])])

    # ---------- main loop -------------------------------------------------
    for t in range(1, P-1):
        xi = params.varrho * (-np.log(-np.log(np.random.rand(n_workers, params.O))))
        for i in range(n_workers):
            st = xst[:, i, t]
            w_det, _ = get_expected_wage_CARA(st, params)
            expEV = compute_cond_EV_CARA(st, pm, pj, wm, wj, phi, bounds, S, params)
            j = (w_det + expEV + xi[i]).argmax()
            occ[i, t] = j + 1

            lam_j, beta_j = params.lambda_mat[j], params.beta_vec[j]
            mu_j = lam_j @ theta_true[:, i] + beta_j @ st[5:7]
            eps = math.sqrt(params.varsigma[j]) * np.random.randn()
            wages[i, t] = mu_j + eps
            signal = mu_j - beta_j @ st[5:7]

            Sigma_old = np.array([[st[2]**2, st[4]*st[2]*st[3]], [st[4]*st[2]*st[3], st[3]**2]])
            Sigma_post = get_tilde_sigma_prime_corrparam(st, j, params)
            theta_post = update_posterior_mean(
                theta_old=st[:2],
                Sigma_old=Sigma_old,
                Sigma_post=Sigma_post,
                lam_j=lam_j,
                signal=signal,
                sigma_eps_sq=params.varsigma[j] ** 2,
            )
            sigC, sigM = math.sqrt(Sigma_post[0,0]), math.sqrt(Sigma_post[1,1])
            rhoP = Sigma_post[0,1]/(sigC*sigM)
            new_tau = st[5:7] + params.b_mat[j]
            xst[:, i, t+1] = np.array([theta_post[0], theta_post[1], sigC, sigM, rhoP, *new_tau])

            if np.random.rand() < 1/(1+math.exp(-params.psi*(new_tau.sum()-params.bar_a))):
                xst[:, i, t+1] = x0
                death[i, t] = 1

        # if (t+1) % 20 == 0:
        share = [np.mean(occ[:, t]==k+1) for k in range(params.O)]
        print(f"t={t+1} | shares {share}")

    return occ, xst, wages, death


# ---------------------------------------------------------------------------
#  Plotting helpers                                                           ──
# ---------------------------------------------------------------------------

def plot_u_shape(
    occ: NDArray, wages: NDArray, death: NDArray,
    *, nbins: int = 20, drop_first_bin: bool = False,
    out: pathlib.Path | None = None,
):
    """Plot raw bin switching probabilities (no LOWESS).

    Parameters
    ----------
    occ : (N, P) int array of occupation IDs.
    wages : (N, P) float array of wages.
    death : (N, P) 0/1 array; 0 = alive.
    nbins : number of percentile bins.
    drop_first_bin : if True, omit the very lowest percentile group.
    out : optional Path to PNG.
    """
    n_workers, P = occ.shape
    O = int(occ.max())

    if death.shape[1] < P:
        death = np.hstack((death, np.zeros((n_workers, 1), dtype=death.dtype)))

    bins = np.linspace(0, 1, nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if drop_first_bin:
        bins = bins[1:]
        bin_centers = bin_centers[1:]

    fig, axs = plt.subplots(1, O, figsize=(4 * O, 4), sharey=True)
    for j in range(1, O + 1):
        probs = np.full(len(bin_centers), np.nan)
        for t in range(P - 1):
            alive = death[:, t] == 0
            idx_j = (occ[:, t] == j) & alive
            if idx_j.sum() == 0:
                continue
            w = wages[idx_j, t]
            r = (np.argsort(np.argsort(w)) + 1) / len(w)  # (0,1]
            sw = (occ[idx_j, t + 1] != j)
            for b in range(len(bin_centers)):
                in_bin = (r >= bins[b]) & (r < bins[b + 1])
                if in_bin.any():
                    if np.isnan(probs[b]):
                        probs[b] = 0.0
                    probs[b] += sw[in_bin].mean()
        # average over time periods where defined
        counts = ~np.isnan(probs)
        if counts.any():
            probs[counts] = probs[counts] / (P - 1)  # crude average
        axs[j - 1].plot(bin_centers[~np.isnan(probs)] * 100, probs[~np.isnan(probs)], '-o')
        axs[j - 1].set_title(f'Occ {j}')
        axs[j - 1].set_xlabel('Wage percentile')
        axs[j - 1].grid(True, linestyle=':')
    axs[0].set_ylabel('Switch prob')
    fig.tight_layout()
    if out:
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out, dpi=150)
    plt.show()


def plot_hazard(occ: NDArray, exp_mat: NDArray, death: NDArray, *, nbins=50, out: pathlib.Path | None = None):
    # exp_mat shape n×P (sum of tau)
    n, P = exp_mat.shape
    mask_alive = (death==0)
    switch = (occ[:,:-1]!= occ[:,1:]) & mask_alive[:,:-1]
    total_exp = exp_mat[:,:-1][mask_alive[:,:-1]]
    switch_exp = exp_mat[:,:-1][switch]
    bins = np.linspace(total_exp.min(), total_exp.max(), nbins+1)
    centers=(bins[:-1]+bins[1:])/2
    hazard = []
    for b in range(nbins):
        denom = ((total_exp>=bins[b])&(total_exp<bins[b+1])).sum()
        num = ((switch_exp>=bins[b])&(switch_exp<bins[b+1])).sum()
        hazard.append(num/denom if denom>0 else np.nan)
    hazard=np.array(hazard)
    sm = lowess(hazard, centers, frac=0.20, return_sorted=True)
    plt.figure(figsize=(6,4))
    plt.plot(sm[:,0], sm[:,1], '-o')
    plt.xlabel('Experience')
    plt.ylabel('Hazard of switch')
    plt.grid(True, linestyle=':')
    plt.savefig(out,dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
#  Main run                                                                   ──
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time                       # ← add
    params = ModelParams()

    # try loading fixed-point estimates -------------------------------------
    res_path = _THIS.parents[2] / "occmodel" / "models" / "results" / "fixed_point_estimates.npz"
    if res_path.exists():
        data = np.load(res_path)
        phi_loaded = data["phi"]
        print(f"Loaded phi from {res_path.relative_to(_THIS.parents[2])}")
    else:
        warnings.warn("phi not found – using zeros (simulation will be rough)")
        phi_loaded = np.zeros(100)    # make sure length matches your basis

    # ----------------- run simulation --------------------------------------
    t0 = time.perf_counter()          # ← start timer
    occ, xst, wages, death = forward_simulation(phi_loaded, params,
                                                P=150, n_workers=300)
    sim_time = time.perf_counter() - t0
    print(f"Simulation finished in {sim_time:,.2f} s – building plots …")

    # total experience (τ_C + τ_M)
    exp_mat = xst[5] + xst[6]         # shape (n, P)

    # --------- drop first P_x periods before plotting ----------------------
    P_x = 0                           # number of periods to skip at the start
    occ_plot   = occ[:, P_x:-1]
    wages_plot = wages[:, P_x:-1]
    death_plot = death[:, P_x:-1]
    exp_plot   = exp_mat[:, P_x:-1]

    FIG_DIR = _THIS.parents[2] / "fig"
    plot_u_shape(occ_plot, wages_plot, death_plot, out=FIG_DIR / "u_shape.png")
    plot_hazard(occ_plot, exp_plot, death_plot, out=FIG_DIR / "hazard_rate.png")

    total_time = time.perf_counter() - t0
    print(f"✓  Figures saved to {FIG_DIR}")
    print(f"🏁  End-to-end runtime: {total_time:,.2f} s")
