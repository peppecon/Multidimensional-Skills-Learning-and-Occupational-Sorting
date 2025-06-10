"""
Dynamic Occupational Choice Model (Python Rewrite)
---------------------------------------------------
This script is a line‑by‑line Python rewrite of the MATLAB code
provided by the user.  The numerical approach and the overall
structure are preserved, but the implementation relies on common
scientific‑Python libraries (NumPy/SciPy) and idiomatic Python
constructs (e.g. dataclasses, type hints, context managers, and
concurrent.futures for parallelism).

⚠️  NOTE
-----
• The external helper routines from the original project (Smolyak grid
  generators, quadrature nodes, wage / variance update helpers, etc.)
  are referenced here as *stubs*.  Replace them with concrete
  implementations or imports from your code‑base.
• The script targets Python 3.11+ and assumes a recent SciPy / NumPy
  stack.  Performance‑critical parts (parfor loops, basis evaluation)
  can later be numba‑accelerated.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
# (1) Helper‑function STUBS – replace / implement!                         ──
# ---------------------------------------------------------------------------

def smolyak_grid(dim: int, order: NDArray[np.int_], lb: NDArray, ub: NDArray) -> tuple[NDArray, dict]:
    """Return the Smolyak collocation nodes and a structure needed for basis construction.
    Placeholder that MUST be replaced by your own implementation.
    """
    raise NotImplementedError


def T_chebyshev(y: NDArray, structure: dict) -> NDArray:
    """Evaluate multivariate Chebyshev polynomials on *y* given *structure* (from smolyak_grid)."""
    raise NotImplementedError


def gauss_hermite(n: int) -> tuple[NDArray, NDArray]:
    """Return (points, weights) for *n*‑point univariate Gauss‑Hermite quadrature."""
    raise NotImplementedError


def get_expected_wage_CARA(state: NDArray, params: "ModelParams") -> NDArray:  # noqa: N802
    raise NotImplementedError


def get_tilde_sigma_prime_corrparam(state: NDArray, occ: int, params: "ModelParams") -> NDArray:  # noqa: N802
    raise NotImplementedError


# ---------------------------------------------------------------------------
# (2) Core data‑structures                                                  ──
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ModelParams:
    """Container for calibrations used throughout the model."""

    # Occupations / skills / state dimension
    O: int = 3
    S: int = 2
    state_dim: int = 7  #  θC, θM, σC, σM, ρ, tenC, tenM

    # Preferences & technology
    beta_disc: float = 0.9
    varrho: float = 0.01  # EV‑type‑I shock variance (scale)
    gamma: float = 1.0    # CARA coefficient

    # death process
    psi: float = 2.0
    bar_a: float = 40.0

    # (vectors defined below are filled in __post_init__)
    varsigma: NDArray = np.array([0.7, 0.7, 0.7])
    quad_points: int = 5
    beta_vec: NDArray = None  # shape (O,2)
    lambda_mat: NDArray = None  # shape (O,S)
    b_mat: NDArray = None  # shape (O,S)

    def __post_init__(self):
        # Occupation‑specific returns to tenure (β_C, β_M rows)
        self.beta_vec = np.array([[0.0, 0.025, 0.05], [0.05, 0.025, 0.0]]).T
        # Skill prices
        self.lambda_mat = np.array([[0.2, 0.9], [0.6, 0.6], [1.0, 0.3]])
        # Skill accumulation rates
        self.b_mat = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])


# ---------------------------------------------------------------------------
# (3) Main fixed‑point iteration                                            ──
# ---------------------------------------------------------------------------

def _evaluate_collocation_point(idx: int, x: NDArray, y: NDArray, bounds: NDArray,
                                TT: NDArray, phi: NDArray, params: ModelParams, points: NDArray, weights: NDArray,
                                sqrt2: float) -> float:
    """Compute expected value at a single collocation point.  Designed for parallel execution."""
    O, S = params.O, params.S

    # Unpack the state vector
    theta_C, theta_M, sigmaC, sigmaM, rho_val, tenC, tenM = x[:, idx]
    state_vec = np.array([theta_C, theta_M, sigmaC, sigmaM, rho_val, tenC, tenM])

    # (a) occupation‑specific wages
    w_tilde_vec = get_expected_wage_CARA(state_vec, params)

    # (b) pre‑compute quadrature products
    pm, pj = np.meshgrid(points, points, indexing="ij")
    wm, wj = np.meshgrid(weights, weights, indexing="ij")
    weights_prod = (wm * wj).ravel()
    nComb = weights_prod.size

    beta_pi_term = params.beta_disc * math.pi ** (‑S / 2)
    exp_EV = np.zeros(O)

    # loop over occupations (small – keep python level)
    for jOcc in range(O):
        # Next‑period variance update
        tilde_sigma_prime = get_tilde_sigma_prime_corrparam(state_vec, jOcc, params)
        try:
            U_prime = cholesky(tilde_sigma_prime, lower=True)
        except np.linalg.LinAlgError:
            # Basic fix: force positive on diagonal
            diag = np.clip(np.diag(tilde_sigma_prime), a_min=np.finfo(float).eps, a_max=None)
            tilde_sigma_prime[np.diag_indices(2)] = diag
            U_prime = cholesky(tilde_sigma_prime, lower=True)

        # Build s' grid
        s_prime = np.tile(state_vec[:, None], (1, nComb))

        sigCprime = math.sqrt(tilde_sigma_prime[0, 0])
        sigMprime = math.sqrt(tilde_sigma_prime[1, 1])
        covPrime = tilde_sigma_prime[0, 1]
        rhoPrime = covPrime / (sigCprime * sigMprime)
        s_prime[2] = sigCprime
        s_prime[3] = sigMprime
        s_prime[4] = rhoPrime

        # tenure update
        b_j = params.b_mat[jOcc]
        s_prime[5] = tenC + b_j[0]
        s_prime[6] = tenM + b_j[1]

        # skill innovations
        skill_new = sqrt2 * (U_prime @ np.vstack((pm.ravel(), pj.ravel()))) + state_vec[:2, None]
        s_prime[:2] = skill_new

        # transform -> Chebyshev domain and evaluate
        y_j = (2 * s_prime ‑ bounds[:, 0:1] ‑ bounds[:, 1:2]) / (bounds[:, 1:2] ‑ bounds[:, 0:1])
        np.clip(y_j, ‑1, 1, out=y_j)
        T_j = T_chebyshev(y_j, structure)
        phiVals = phi @ T_j

        # death probability
        experience = s_prime[5] + s_prime[6]
        death_proba = 1 / (1 + np.exp(‑params.psi * (experience ‑ params.bar_a)))
        exp_EV[jOcc] = (1 ‑ death_proba) * beta_pi_term * np.sum(weights_prod * phiVals)

    # log‑sum‑exp expected value
    v_vec = w_tilde_vec + exp_EV
    max_val = np.max(v_vec)
    EV = max_val + params.varrho * math.log(np.sum(np.exp((v_vec ‑ max_val) / params.varrho)))
    return EV


def solve_model(params: ModelParams) -> tuple[NDArray, NDArray]:
    """Main driver replicating the MATLAB fixed‑point loop."""
    # (1) Collocation bounds (replicates the MATLAB block)
    sigma_min, sigma_max = 0.001, 1.0
    corr_min, corr_max = ‑0.95, 0.95

    bounds = np.zeros((params.state_dim, 2))
    bounds[0:params.S] = np.array([‑2, 5])        # skill means
    bounds[params.S:2*params.S] = np.array([sigma_min, sigma_max])  # stdev
    bounds[2*params.S] = np.array([corr_min, corr_max])             # correlation
    bounds[3*params.S:] = np.array([0, 50])        # tenure

    order = np.full(params.state_dim, 3)

    # (2) Smolyak grid / basis
    coll_points, structure = smolyak_grid(params.state_dim, order, bounds[:, 0], bounds[:, 1])
    N_g = coll_points.shape[1]
    y = (2 * coll_points ‑ bounds[:, [0]] ‑ bounds[:, [1]]) / (bounds[:, [1]] ‑ bounds[:, [0]])
    TT = T_chebyshev(y, structure)

    # (3) initial guesses
    phi = np.full(N_g, 0.001)
    err, iter_, n_iters, check = 100.0, 0, 1500, ‑10.0

    points, weights = gauss_hermite(params.quad_points)
    sqrt2 = math.sqrt(2.0)

    while iter_ < n_iters and err > 1e‑8 and check < 0:
        t0 = time.perf_counter()
        print(f"Iteration #{iter_}")

        # parallel evaluation
        with ProcessPoolExecutor() as ex:
            EV_list = list(ex.map(_evaluate_collocation_point,
                                   range(N_g),
                                   [coll_points]*N_g,
                                   [y]*N_g,
                                   [bounds]*N_g,
                                   [TT]*N_g,
                                   [phi]*N_g,
                                   [params]*N_g,
                                   [points]*N_g,
                                   [weights]*N_g,
                                   [sqrt2]*N_g))
        EV = np.array(EV_list)

        # normal‑equation update (TT^T phi = EV)
        phi_new = np.linalg.lstsq(TT.T, EV, rcond=None)[0]
        xi = 1.0
        phi_new = xi * phi_new + (1 ‑ xi) * phi

        err_new = np.linalg.norm(phi_new ‑ phi)
        check = err_new ‑ err
        err = err_new
        phi = phi_new
        iter_ += 1
        dt = time.perf_counter() ‑ t0
        print(f"Error: {err:.3e}   |   dt = {dt:.2f}s")

    return phi, EV


if __name__ == "__main__":
    params = ModelParams()
    phi_star, EV_grid = solve_model(params)
    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "fixed_point_estimates.npz", EV=EV_grid, phi=phi_star)
    print("✔ Results saved to results/fixed_point_estimates.npz")
