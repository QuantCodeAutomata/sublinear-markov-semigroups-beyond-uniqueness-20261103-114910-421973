"""
Canonical semigroup V^G via Monte Carlo upper expectations.

V_t^G f(x) = sup_{P in P_x(G)} E^P[f(X_t) 1_{t < tau_kill}]

Paper: Section 1.3 semigroup map; Section 5.2 upper Perron representation.
"""
from __future__ import annotations
from typing import Callable, Optional, List
import numpy as np
from src.simulator import _simulate_with_kill_fn, simulate_absorbing_branch, simulate_zero_sojourn_branch


def make_control_grid(
    sigma_vals: np.ndarray,
    b_vals: np.ndarray,
    c_vals: np.ndarray,
) -> list[dict]:
    """Return list of control dicts with keys 'sigma', 'b', 'c'."""
    return [{"sigma": float(s), "b": float(b), "c": float(c)}
            for s, b, c in zip(sigma_vals, b_vals, c_vals)]


def make_default_control_grid(
    sigma_min: float = 0.2,
    sigma_max: float = 1.0,
    b_min: float = -0.5,
    b_max: float = 0.5,
    c_max: float = 0.3,
    n_theta: int = 25,
) -> list[dict]:
    """Create a discrete control grid by sampling sigma, b, c uniformly.

    Returns a list of n_theta control dicts, each with 'sigma', 'b', 'c'.
    c <= 0 (killing), so we sample c in [-c_max, 0].

    Paper: Section 1.3 — admissible coefficient set A_G.
    """
    rng = np.random.default_rng(0)
    sigmas = rng.uniform(sigma_min, sigma_max, n_theta)
    bs = rng.uniform(b_min, b_max, n_theta)
    cs = rng.uniform(-c_max, 0.0, n_theta)
    return make_control_grid(sigmas, bs, cs)


def compute_VG(
    x0_vals: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list[dict],
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate V_t^G f(x) for each x in x0_vals.

    V_t^G f(x) = sup_{theta in controls} E^theta[f(X_t) 1_{t < tau_kill}]

    Paper: Section 1.3 — upper-expectation semigroup over admissible laws.

    Parameters
    ----------
    x0_vals  : (N,) initial states
    t        : time horizon
    payoff_fn: bounded measurable payoff f
    controls : list of dicts with 'sigma', 'b', 'c' keys
    n_paths  : Monte Carlo paths per control
    dt       : Euler-Maruyama step
    rng      : numpy Generator

    Returns
    -------
    V : (N,) array of semigroup values
    """
    N = len(x0_vals)
    V = np.full(N, -np.inf)
    for ctrl in controls:
        sigma_c = float(ctrl["sigma"])
        b_c = float(ctrl["b"])
        c_c = float(ctrl["c"])
        killing_fn = (
            (lambda x, _k=-c_c: np.full(len(x), _k)) if c_c < 0.0 else None
        )
        ctrl_vals = np.zeros(N)
        for i, x0 in enumerate(x0_vals):
            X_end, alive = _simulate_with_kill_fn(
                x0=x0,
                t_end=t,
                dt=dt,
                b_fn=lambda x, _b=b_c: np.full(len(x), _b),
                sigma_fn=lambda x, _s=sigma_c: np.full(len(x), _s),
                killing_fn=killing_fn,
                n_paths=n_paths,
                rng=rng,
            )
            payoffs = np.zeros(n_paths)
            if np.any(alive):
                payoffs[alive] = payoff_fn(X_end[alive])
            ctrl_vals[i] = float(np.mean(payoffs))
        V = np.maximum(V, ctrl_vals)
    return V


def check_semigroup_property(
    x_pts: np.ndarray,
    s: float,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list[dict],
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Check V_{s+t} f(x) approx= V_s(V_t f)(x).

    Paper: Section 3.2 — time-consistency of the canonical semigroup.

    Returns dict with 'max_abs_err', 'mean_abs_err', 'V_st', 'V_s_Vt'.
    """
    # Direct: V_{s+t} f
    rng1 = np.random.default_rng(rng.integers(2**31))
    V_st = compute_VG(x_pts, s + t, payoff_fn, controls, n_paths, dt, rng1)

    # Composed: V_s(V_t f)
    rng2 = np.random.default_rng(rng.integers(2**31))
    V_t_f = compute_VG(x_pts, t, payoff_fn, controls, n_paths, dt, rng2)
    # Build the composed payoff: interpolate V_t_f as a function of x
    from scipy.interpolate import interp1d
    interp = interp1d(x_pts, V_t_f, kind="linear", fill_value="extrapolate")
    composed_payoff = lambda x: interp(x)
    rng3 = np.random.default_rng(rng.integers(2**31))
    V_s_Vt = compute_VG(x_pts, s, composed_payoff, controls, n_paths, dt, rng3)

    abs_err = np.abs(V_st - V_s_Vt)
    return {
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "V_st": V_st,
        "V_s_Vt": V_s_Vt,
    }


def check_semigroup_sublinearity(
    x_pts: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list[dict],
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Check positive homogeneity and subadditivity of V^G.

    Paper: Definition 2.1 (V3) — sublinearity.
    """
    lam = 2.0
    rng1 = np.random.default_rng(rng.integers(2**31))
    V_f = compute_VG(x_pts, t, payoff_fn, controls, n_paths, dt, rng1)
    rng2 = np.random.default_rng(rng.integers(2**31))
    V_lf = compute_VG(x_pts, t, lambda x: lam * payoff_fn(x), controls, n_paths, dt, rng2)
    homogeneity_err = float(np.max(np.abs(V_lf - lam * V_f)))

    f2 = lambda x: 0.5 * payoff_fn(x)
    rng3 = np.random.default_rng(rng.integers(2**31))
    V_f2 = compute_VG(x_pts, t, f2, controls, n_paths, dt, rng3)
    rng4 = np.random.default_rng(rng.integers(2**31))
    V_sum = compute_VG(x_pts, t, lambda x: payoff_fn(x) + f2(x), controls, n_paths, dt, rng4)
    subadditivity_violation = float(np.max(np.maximum(V_sum - V_f - V_f2, 0.0)))

    return {
        "homogeneity_err": homogeneity_err,
        "subadditivity_violation": subadditivity_violation,
        "is_sublinear": homogeneity_err < 0.1 and subadditivity_violation < 0.1,
    }


def compute_VG_alpha(
    x0: float,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Upper envelope over absorbing and zero-sojourn branches for Example 4.1.

    V_t^{G_alpha} f(x0) = max(E^{abs}[f], E^{zs}[f])

    Paper: Section 4.2, p. 30 — nonuniqueness example; canonical semigroup
    is the upper envelope over all admissible laws.
    """
    X_abs, alive_abs = simulate_absorbing_branch(x0, t, dt, alpha, n_paths, rng)
    X_zs, alive_zs = simulate_zero_sojourn_branch(x0, t, dt, alpha, n_paths, rng)
    payoffs_abs = np.zeros(n_paths)
    if np.any(alive_abs):
        payoffs_abs[alive_abs] = payoff_fn(X_abs[alive_abs])
    payoffs_zs = np.zeros(n_paths)
    if np.any(alive_zs):
        payoffs_zs[alive_zs] = payoff_fn(X_zs[alive_zs])
    return float(max(np.mean(payoffs_abs), np.mean(payoffs_zs)))


def compute_VG_diffusion(
    x0_vals: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    sigma_fn: Callable[[np.ndarray], np.ndarray],
    b_fn: Callable[[np.ndarray], np.ndarray],
    killing_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """V^G for a single-control diffusion with given coefficient functions.

    Used in exp6 to compare a sub-family semigroup against the full V^G.

    Paper: Section 5.2 — stochastic representation via admissible law family.
    """
    N = len(x0_vals)
    V = np.zeros(N)
    for i, x0 in enumerate(x0_vals):
        X_end, alive = _simulate_with_kill_fn(
            x0=x0, t_end=t, dt=dt,
            b_fn=b_fn, sigma_fn=sigma_fn, killing_fn=killing_fn,
            n_paths=n_paths, rng=rng,
        )
        payoffs = np.zeros(n_paths)
        if np.any(alive):
            payoffs[alive] = payoff_fn(X_end[alive])
        V[i] = float(np.mean(payoffs))
    return V


def compute_canonical_upper_perron(
    x0_vals: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list[dict],
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
    branch_abs: bool = True,
    branch_zs: bool = True,
    alpha: float = 0.3,
) -> dict:
    """Upper Perron envelope over absorbing and zero-sojourn branches (Example 4.1).

    Paper: Section 5.2 — V^G is the canonical greatest semigroup.
    """
    N = len(x0_vals)
    V_abs = np.zeros(N)
    V_zs = np.zeros(N)
    for i, x0 in enumerate(x0_vals):
        if branch_abs:
            X_abs, alive_abs = simulate_absorbing_branch(x0, t, dt, alpha, n_paths, rng)
            payoffs_abs = np.zeros(n_paths)
            if np.any(alive_abs):
                payoffs_abs[alive_abs] = payoff_fn(X_abs[alive_abs])
            V_abs[i] = float(np.mean(payoffs_abs))
        if branch_zs:
            X_zs, alive_zs = simulate_zero_sojourn_branch(x0, t, dt, alpha, n_paths, rng)
            payoffs_zs = np.zeros(n_paths)
            if np.any(alive_zs):
                payoffs_zs[alive_zs] = payoff_fn(X_zs[alive_zs])
            V_zs[i] = float(np.mean(payoffs_zs))
    V_ctrl = compute_VG(x0_vals, t, payoff_fn, controls, n_paths, dt, rng)
    envelope = np.maximum(V_abs, np.maximum(V_zs, V_ctrl))
    return {
        "V_abs": V_abs,
        "V_zs": V_zs,
        "V_ctrl": V_ctrl,
        "V_envelope": envelope,
    }
