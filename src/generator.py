"""
Generator estimation for sublinear semigroups.

The upper generator of T at f is:
    G_T(x, J^2 f) = limsup_{h->0+} [T_h f(x) - f(x)] / h

Paper: Definition 2.2 — upper generating function; Section 1.3 overview.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence
import numpy as np


def estimate_generator(
    x_pts: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    h_vals: List[float],
    controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Estimate Delta_h^T f(x) = (T_h f(x) - f(x)) / h for multiple h values.

    Paper: Definition 2.2 — upper generator via limsup of difference quotients.

    Parameters
    ----------
    x_pts    : (N,) state grid
    payoff_fn: test function f
    h_vals   : list of h > 0 (decreasing)
    controls : list of control dicts for V^G
    n_paths  : Monte Carlo paths
    dt       : time step
    rng      : numpy Generator

    Returns
    -------
    dict with 'gen_estimates' (dict h -> (N,) array), 'G_T_estimate' (N,).
    """
    from src.upper_expectation import compute_VG
    f0 = payoff_fn(x_pts)
    gen_estimates = {}
    for h in h_vals:
        T_h_f = compute_VG(x_pts, h, payoff_fn, controls, n_paths, dt, rng)
        gen_estimates[h] = (T_h_f - f0) / h
    G_T_est = gen_estimates[min(h_vals)] if h_vals else np.zeros_like(x_pts)
    return {
        "gen_estimates": gen_estimates,
        "G_T_estimate": G_T_est,
        "h_vals": list(h_vals),
    }


def estimate_generator_analytic(
    x_pts: np.ndarray,
    sigma: float,
    b: float,
    f_expr: str,
) -> dict:
    """Analytic generator G f = 0.5 sigma^2 f'' + b f' using sympy.

    Paper: Definition 2.2 — for a single constant-coefficient diffusion
    the generator is exactly 0.5 sigma^2 D^2 f + b D f.

    Parameters
    ----------
    x_pts  : (N,) evaluation points
    sigma  : diffusion coefficient
    b      : drift coefficient
    f_expr : sympy-compatible expression string in variable 'x'

    Returns
    -------
    dict with 'gen_values' (N,), 'f_prime_vals', 'f_double_prime_vals'.
    """
    import sympy as sp
    x_sym = sp.Symbol("x")
    f_sym = sp.sympify(f_expr)
    fp = sp.diff(f_sym, x_sym)
    fpp = sp.diff(fp, x_sym)
    gen_sym = sp.Rational(1, 2) * sigma**2 * fpp + b * fp

    gen_fn = sp.lambdify(x_sym, gen_sym, "numpy")
    fp_fn  = sp.lambdify(x_sym, fp, "numpy")
    fpp_fn = sp.lambdify(x_sym, fpp, "numpy")

    gen_vals = np.asarray(gen_fn(x_pts), dtype=float)
    if gen_vals.ndim == 0:
        gen_vals = np.full(len(x_pts), float(gen_vals))
    return {
        "gen_values": gen_vals,
        "f_prime_vals": np.asarray(fp_fn(x_pts), dtype=float),
        "f_double_prime_vals": np.asarray(fpp_fn(x_pts), dtype=float),
    }


def check_generator_bound(
    x_pts: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    h: float,
    controls_T: list,
    controls_VG: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Check G_T(x, J^2 f) <= G_VG(x, J^2 f) pointwise.

    Paper: Section 4.1 — G_T <= G iff T <= V^G (Corollary 4.8).

    Parameters
    ----------
    x_pts      : (N,) state grid
    payoff_fn  : test function
    h          : time increment for finite-difference estimate
    controls_T : controls defining semigroup T
    controls_VG: controls defining V^G
    n_paths    : Monte Carlo paths
    dt         : time step
    rng        : numpy Generator

    Returns
    -------
    dict with 'bound_satisfied' (bool), 'max_excess', 'G_T', 'G_VG'.
    """
    from src.upper_expectation import compute_VG
    f0 = payoff_fn(x_pts)
    rng1 = np.random.default_rng(rng.integers(2**31))
    rng2 = np.random.default_rng(rng.integers(2**31))
    T_h_T  = compute_VG(x_pts, h, payoff_fn, controls_T, n_paths, dt, rng1)
    T_h_VG = compute_VG(x_pts, h, payoff_fn, controls_VG, n_paths, dt, rng2)
    G_T  = (T_h_T  - f0) / h
    G_VG = (T_h_VG - f0) / h
    excess = G_T - G_VG  # should be <= 0
    max_excess = float(np.max(excess))
    return {
        "bound_satisfied": max_excess < 0.1,
        "max_excess": max_excess,
        "G_T": G_T,
        "G_VG": G_VG,
    }


def compare_generators_to_spec(
    x_pts: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    h_vals: List[float],
    controls: list,
    spec_fn: Callable,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Compare the numerically estimated generator of V^G with the spec G.

    Paper: Section 5.1 — exact generation: G_{V^G} f = G(., J^2 f) in the limit.

    Parameters
    ----------
    x_pts   : (N,) state grid
    payoff_fn: test function f
    h_vals  : list of h values for finite-difference estimates
    controls: control dicts defining V^G
    spec_fn : G(x, z, p, H) — the local specification
    n_paths : Monte Carlo paths
    dt      : time step
    rng     : numpy Generator

    Returns
    -------
    dict with 'mean_abs_diff', 'max_abs_diff', 'G_T_estimate', 'G_spec'.
    """
    res = estimate_generator(x_pts, payoff_fn, h_vals, controls, n_paths, dt, rng)
    G_T = res["G_T_estimate"]

    # Analytically compute G(x, J^2 f) using finite differences for derivatives
    dx = float(np.diff(x_pts).mean()) if len(x_pts) > 1 else 1e-4
    eps = dx * 0.01
    f0 = payoff_fn(x_pts)
    fp = payoff_fn(x_pts + eps)
    fm = payoff_fn(x_pts - eps)
    f_prime = (fp - fm) / (2 * eps)
    f_double = (fp - 2 * f0 + fm) / eps**2
    G_spec = np.array([
        spec_fn(float(x), float(f0[i]), float(f_prime[i]), float(f_double[i]))
        for i, x in enumerate(x_pts)
    ])
    abs_diff = np.abs(G_T - G_spec)
    return {
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "G_T_estimate": G_T,
        "G_spec": G_spec,
    }
