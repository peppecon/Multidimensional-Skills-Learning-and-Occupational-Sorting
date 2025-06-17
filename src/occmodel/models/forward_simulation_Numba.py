# ────────────────────────────────────────────────────────────────────────────
#  forward_simulation_numba.py      (fully Numba-compatible edition)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import pathlib, sys, math, time, warnings
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, int64, float64
from numba.experimental import jitclass

# ── path bootstrap so local imports work without pip install ────────────────
_THIS = pathlib.Path(__file__).resolve()
SRC_DIR = _THIS.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── **NB-versions** of every helper we need ─────────────────────────────────
from smolyak_step_1 import smolyak_step1                                # ✅ jitted
from quadrature import gauss_hermite                                    # ✅ jitted
from t_numba import _chebyshev_T_nb, chebyshev_T                        # ✅ jitted
from wage_and_variance_numba import (
    get_expected_wage_CARA_nb,              # ✅ jitted
    get_tilde_sigma_prime_corrparam_nb,     # ✅ jitted
)
from ev_conditional_nb import compute_cond_EV_CARA_nb                # ✅ jitted

# ────────────────────────────────────────────────────────────────────────────
#  1. Parameters as jit-class
# ────────────────────────────────────────────────────────────────────────────
spec = [
    ('O',           int64),    ('S', int64),     ('d', int64),
    ('beta',        float64),  ('varrho', float64), ('gamma', float64),
    ('varsigma',    float64[:]),
    ('psi',         float64),  ('bar_a', float64),
    ('quad_points', int64),
    ('beta_vec',    float64[:, :]),
    ('lambda_mat',  float64[:, :]),
    ('b_mat',       float64[:, :]),
]

@jitclass(spec)
class ModelParams:
    def __init__(self):
        self.O, self.S, self.d = 3, 2, 7
        self.beta, self.varrho, self.gamma = 0.9, 0.01, 1.0
        self.varsigma = np.array([0.7, 0.7, 0.7], dtype=np.float64)
        self.psi, self.bar_a, self.quad_points = 2.0, 40.0, 5

        self.beta_vec = np.empty((3, 2), dtype=np.float64)
        self.beta_vec[:, 0] = (0.00, 0.025, 0.050)
        self.beta_vec[:, 1] = (0.05, 0.025, 0.000)

        self.lambda_mat = np.array([[0.2, 0.9],
                                    [0.6, 0.6],
                                    [1.0, 0.3]], dtype=np.float64)

        self.b_mat = np.array([[0.2, 0.8],
                               [0.5, 0.5],
                               [0.8, 0.2]], dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
#  2. Tiny helpers
# ────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def update_posterior_mean(theta_old, Sigma_old, Sigma_post,
                          lam_j, signal, sigma_eps_sq):
    rhs = np.linalg.solve(Sigma_old, theta_old) + lam_j * (signal / sigma_eps_sq)
    return Sigma_post @ rhs


def build_grid(d: int):
    bounds = np.zeros((d, 2))
    bounds[0:2] = (-2.0, 5.0)
    bounds[2:4] = (0.001, 1.0)
    bounds[4]   = (-0.95, 0.95)
    bounds[5:7] = (0, 50)
    order_arr   = np.full(d, 3, dtype=np.int64)
    _, S = smolyak_step1(d, order_arr, bounds[:, 0], bounds[:, 1])
    return bounds.astype(np.float64), S


def extract_lmat_maxorder(S):
    """Return (l_mat_int64, max_order_int) from a SmolyakStruct‐like object."""
    # 1️⃣  Try the most common attribute names
    for attr in ("l_mat", "l", "index", "_l", "l_mat_full"):
        if hasattr(S, attr):
            l_raw = getattr(S, attr)
            break
    else:                 # not found – maybe S is a namedtuple (index is first)
        try:
            l_raw = S[0]  # first element
        except Exception as e:
            raise AttributeError("Cannot locate multi-index matrix in S") from e

    # 2️⃣  Convert to a contiguous int64 NumPy array
    l_mat = np.asarray(l_raw, dtype=np.int64)
    if l_mat.ndim != 2:
        raise ValueError("l_mat should be 2-D (nbasis × d)")

    max_order = int(l_mat.max())
    return l_mat, max_order


# ────────────────────────────────────────────────────────────────────────────
#  3. Forward-simulation kernel  (Numba parallel, literal-array free)
# ────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def forward_simulation(phi, params,
                       pm, pj, wm, wj,
                       bounds, l_mat, max_order,
                       P: int = 150, n_workers: int = 300):

    # ----- constant initial state ------------------------------------------
    x0 = np.empty(7, dtype=np.float64)
    x0[:] = (0.6, 0.9, 0.6, 0.6, 0.35, 0.0, 0.0)

    # ----- allocate tensors -------------------------------------------------
    xst   = np.zeros((params.d, n_workers, P), dtype=np.float64)
    occ   = np.zeros((n_workers, P),           dtype=np.int8)
    wages = np.zeros((n_workers, P),           dtype=np.float64)
    death = np.zeros((n_workers, P),           dtype=np.int8)

    for w in range(n_workers):
        for k in range(P):
            xst[:, w, k] = x0

    # ----- Σ0 and θtrue -----------------------------------------------------
    Sigma0 = np.empty((2, 2), dtype=np.float64)
    Sigma0[0, 0] = 0.36
    Sigma0[0, 1] = 0.126
    Sigma0[1, 0] = 0.126
    Sigma0[1, 1] = 0.36
    L = np.linalg.cholesky(Sigma0)

    offset = np.empty((2, 1), dtype=np.float64)
    offset[0, 0] = 0.6
    offset[1, 0] = 0.9
    theta_true = (L @ np.random.randn(2, n_workers)) + offset

    # ── t = 0 choice --------------------------------------------------------
    w_det0 = get_expected_wage_CARA_nb(
        x0, params.lambda_mat, params.beta_vec, params.varsigma, params.gamma
    )
    expEV0 = compute_cond_EV_CARA_nb(
        x0, pm, pj, wm, wj, phi, bounds, l_mat, max_order, params
    )
    xi0 = params.varrho * (-np.log(-np.log(np.random.rand(n_workers, params.O))))
    occ[:, 0] = (w_det0 + expEV0 + xi0).argmax() + 1

    # ── belief update t = 0 -------------------------------------------------
    for i in range(n_workers):
        j = occ[i, 0] - 1
        lam_j  = params.lambda_mat[j]
        beta_j = params.beta_vec[j]

        mu_j   = lam_j @ theta_true[:, i] + beta_j @ x0[5:7]
        wages[i, 0] = mu_j + math.sqrt(params.varsigma[j]) * np.random.randn()
        signal = mu_j - beta_j @ x0[5:7]

        Sigma_post = get_tilde_sigma_prime_corrparam_nb(
            x0, j, params.lambda_mat, params.varsigma
        )
        theta_post = update_posterior_mean(
            x0[:2], Sigma0, Sigma_post, lam_j, signal, params.varsigma[j] ** 2
        )

        sigC = math.sqrt(Sigma_post[0, 0])
        sigM = math.sqrt(Sigma_post[1, 1])
        rhoP = Sigma_post[0, 1] / (sigC * sigM + 1e-12)

        row = xst[:, i, 1]          # write elements one-by-one
        row[0] = theta_post[0]
        row[1] = theta_post[1]
        row[2] = sigC
        row[3] = sigM
        row[4] = rhoP
        row[5] = x0[5] + params.b_mat[j, 0]
        row[6] = x0[6] + params.b_mat[j, 1]

    # ── main loop  t = 1 … P–2 ---------------------------------------------
    for t in range(1, P - 1):
        xi = params.varrho * (-np.log(-np.log(np.random.rand(n_workers, params.O))))
        for i in range(n_workers):
            st = xst[:, i, t]

            w_det = get_expected_wage_CARA_nb(
                st, params.lambda_mat, params.beta_vec,
                params.varsigma, params.gamma
            )
            expEV = compute_cond_EV_CARA_nb(
                st, pm, pj, wm, wj, phi, bounds, l_mat, max_order, params
            )
            j = (w_det + expEV + xi[i]).argmax()
            occ[i, t] = j + 1

            lam_j  = params.lambda_mat[j]
            beta_j = params.beta_vec[j]
            mu_j   = lam_j @ theta_true[:, i] + beta_j @ st[5:7]
            wages[i, t] = mu_j + math.sqrt(params.varsigma[j]) * np.random.randn()
            signal = mu_j - beta_j @ st[5:7]

            Sigma_old = np.empty((2, 2), dtype=np.float64)
            Sigma_old[0, 0] = st[2] * st[2]
            Sigma_old[0, 1] = st[4] * st[2] * st[3]
            Sigma_old[1, 0] = Sigma_old[0, 1]
            Sigma_old[1, 1] = st[3] * st[3]

            Sigma_post = get_tilde_sigma_prime_corrparam_nb(
                st, j, params.lambda_mat, params.varsigma
            )
            theta_post = update_posterior_mean(
                st[:2], Sigma_old, Sigma_post,
                lam_j, signal, params.varsigma[j] ** 2
            )

            sigC = math.sqrt(Sigma_post[0, 0])
            sigM = math.sqrt(Sigma_post[1, 1])
            rhoP = Sigma_post[0, 1] / (sigC * sigM + 1e-12)

            new_tau0 = st[5] + params.b_mat[j, 0]
            new_tau1 = st[6] + params.b_mat[j, 1]
            row = xst[:, i, t + 1]
            row[0] = theta_post[0]
            row[1] = theta_post[1]
            row[2] = sigC
            row[3] = sigM
            row[4] = rhoP
            row[5] = new_tau0
            row[6] = new_tau1

            # death reset ---------------------------------------------------
            if np.random.rand() < 1.0 / (1.0 + math.exp(
                    -params.psi * (new_tau0 + new_tau1 - params.bar_a))):
                xst[:, i, t + 1] = x0
                death[i, t] = 1
                # if (t+1) % 20 == 0:
        share = [np.mean(occ[:, t]==k+1) for k in range(params.O)]
        print(f"t={t+1} | shares {share}")


    return occ, xst, wages, death


# ────────────────────────────────────────────────────────────────────────────
#  4. Plotting (unchanged – pure Python)
# ────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_u_shape(occ, wages, death, nbins=20, out=None):
    n_workers, P = occ.shape
    O = int(occ.max())
    if death.shape[1] < P:
        death = np.hstack((death, np.zeros((n_workers, 1), dtype=death.dtype)))
    bins = np.linspace(0, 1, nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axs = plt.subplots(1, O, figsize=(4 * O, 4), sharey=True)
    for j in range(1, O + 1):
        probs = np.full(len(bin_centers), np.nan)
        for t in range(P - 1):
            alive = death[:, t] == 0
            idx = (occ[:, t] == j) & alive
            if idx.sum() == 0:
                continue
            w = wages[idx, t]
            r = (np.argsort(np.argsort(w)) + 1) / len(w)
            sw = (occ[idx, t + 1] != j)
            for b in range(len(bin_centers)):
                in_bin = (r >= bins[b]) & (r < bins[b + 1])
                if in_bin.any():
                    if np.isnan(probs[b]):
                        probs[b] = 0.0
                    probs[b] += sw[in_bin].mean()
        m = ~np.isnan(probs)
        probs[m] /= (P - 1)
        axs[j - 1].plot(bin_centers[m] * 100, probs[m], '-o')
        axs[j - 1].set_title(f'Occ {j}')
        axs[j - 1].set_xlabel('Wage percentile')
        axs[j - 1].grid(True, linestyle=':')
    axs[0].set_ylabel('Switch prob')
    fig.tight_layout()
    if out:
        out = pathlib.Path(out)
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out, dpi=150)
    plt.show()


def plot_hazard(occ, exp_mat, death, nbins=50, out=None):
    n, P = exp_mat.shape
    mask_alive = (death == 0)
    switch = (occ[:, :-1] != occ[:, 1:]) & mask_alive[:, :-1]
    total = exp_mat[:, :-1][mask_alive[:, :-1]]
    sw_exp = exp_mat[:, :-1][switch]
    bins = np.linspace(total.min(), total.max(), nbins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    hazard = np.empty(nbins, dtype=np.float64)
    for b in range(nbins):
        denom = ((total >= bins[b]) & (total < bins[b + 1])).sum()
        num = ((sw_exp >= bins[b]) & (sw_exp < bins[b + 1])).sum()
        hazard[b] = num / denom if denom else np.nan
    sm = lowess(hazard, centers, frac=0.20, return_sorted=True)
    plt.figure(figsize=(6, 4))
    plt.plot(sm[:, 0], sm[:, 1], '-o')
    plt.xlabel('Experience')
    plt.ylabel('Hazard of switch')
    plt.grid(True, linestyle=':')
    if out:
        out = pathlib.Path(out)
        out.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out, dpi=150)
    plt.show()


# ------------------------------------------------------------------
#  Driver code  (pure Python, outside @njit)
# ------------------------------------------------------------------
if __name__ == "__main__":
    params = ModelParams()

    # 1. Chebyshev/Smolyak grid (only ONCE)
    bounds, S = build_grid(params.d)                  # jit-compiled, returns np array + jitclass

    # 2. Gauss–Hermite grid (only ONCE)
    gh_pts, gh_w = gauss_hermite(params.quad_points)
    gh_pts = gh_pts.astype(np.float64)
    gh_w   = gh_w.astype(np.float64)
    pm, pj = np.meshgrid(gh_pts, gh_pts, indexing="ij")
    wm, wj = np.meshgrid(gh_w,   gh_w,   indexing="ij")

    # 3. Load φ and run the simulation
    res_path = _THIS.parents[2] / "occmodel" / "models" / "results" / "fixed_point_estimates.npz"
    phi_loaded = np.load(res_path)["phi"] if res_path.exists() else np.zeros(100, dtype=np.float64)

    t0 = time.time()
    l_mat, max_order = extract_lmat_maxorder(S)  # ← robust extraction
    occ, xst, wages, death = forward_simulation(
        phi_loaded, params,
        pm, pj, wm, wj,              # quadrature
        bounds, l_mat, max_order,    # ← new
        P=150, n_workers=500
    )
    print(f"Forward simulation finished in {time.time() - t0:,.2f} s.")

    exp_mat = xst[5] + xst[6]

    FIG_DIR = _THIS.parents[2] / "fig"
    plot_u_shape(occ, wages, death, out=FIG_DIR / "u_shape.png")
    plot_hazard(occ, exp_mat, death, out=FIG_DIR / "hazard_rate.png")
    print("✓ Figures saved to", FIG_DIR)
