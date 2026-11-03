"""
Generator estimation for sublinear semigroups.

The upper generator of T at f is:
    G_T(x, J^2 f) = limsup_{h->0+} [T_h f(x) - f(x)] / h

Paper: Definition 2.2 — upper generating function; Section 1.3 overview.
"""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np


def test_function_library(x_grid: np.ndarray) -> list[dict]:
    """Return a library of smooth test functions with analytic derivatives.

    Each dict has keys:
      'name' : str
      'f'    : callable (N,) -> (N,)
      'df'   : callable — first derivative
      'd2f'  : callable — second derivative
      'f_arr': (N,) values on x_grid

    Paper: Definition 2.2 — test class for upper generator estimation.
    """
    import numpy as np

    def _make(name, f_fn, df_fn, d2f_fn):
        return {
            "name": name,
            "f": f_fn,
            "df": df_fn,
            "d2f": d2f_fn,
            "f_arr": f_fn(x_grid),
        }

    # f1: f(x) = x^2 / (1 + x^2)
    fns = [
        _make(
            "f_plus",
            lambda x: x ** 2 / (1.0 + x ** 2),
            lambda x: 2 * x / (1.0 + x ** 2) ** 2,
            lambda x: (2.0 * (1.0 + x**2)**2 - 2*x * 2*(1.0+x**2)*2*x)
                       / (1.0 + x**2)**4,
        ),
        # f2: f(x) = exp(-x^2/2)
        _make(
            "gaussian",
            lambda x: np.exp(-x ** 2 / 2.0),
            lambda x: -x * np.exp(-x ** 2 / 2.0),
            lambda x: (x ** 2 - 1.0) * np.exp(-x ** 2 / 2.0),
        ),
        # f3: f(x) = tanh(x)
        _make(
            "tanh",
            lambda x: np.tanh(x),
            lambda x: 1.0 / np.cosh(x) ** 2,
            lambda x: -2.0 * np.tanh(x) / np.cosh(x) ** 2,
        ),
        # f4: f(x) = sin(x) * exp(-x^2/4)
        _make(
            "sin_env",
            lambda x: np.sin(x) * np.exp(-x ** 2 / 4.0),
            lambda x: (np.cos(x) - x * np.sin(x) / 2.0) * np.exp(-x ** 2 / 4.0),
            lambda x: (
                -np.sin(x) - np.cos(x) * x / 2.0
                - (np.cos(x) - x * np.sin(x) / 2.0) * x / 2.0
            ) * np.exp(-x ** 2 / 4.0)
            + (np.cos(x) - x * np.sin(x) / 2.0) * (-x / 2.0) * np.exp(-x ** 2 / 4.0),
        ),
        # f5: f(x) = 0.5 (constant, trivial test)
        _make(
            "const",
            lambda x: 0.5 * np.ones_like(x),
            lambda x: np.zeros_like(x),
            lambda x: np.zeros_like(x),
        ),
    ]
    return fns


def estimate_generator(
    semigroup_type,
    x_pts: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    h_vals: List[float],
    eval_kwargs: dict,
) -> dict:
    """Estimate Delta_h^T f(x) = (T_h f(x) - f(x)) / h for multiple h values.

    Paper: Definition 2.2 — upper generator via limsup of difference quotients.

    Parameters
    ----------
    semigroup_type : SemigroupType or string — selects CONTROLLED (V^G) evaluation
    x_pts          : (N,) state grid
    payoff_fn      : test function f
    h_vals         : decreasing sequence of time increments
    eval_kwargs    : dict with keys 'controls', 'n_paths', 'dt', 'rng'

    Returns
    -------
    dict with 'gen_estimates' (dict h -> (N,) array), 'G_T_estimate' (N,).
    """
    from src.upper_expectation import compute_VG
    controls = eval_kwargs["controls"]
    n_paths  = eval_kwargs["n_paths"]
    dt       = eval_kwargs["dt"]
    rng      = eval_kwargs["rng"]
    f0 = payoff_fn(x_pts)
    gen_estimates: dict = {}
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
    fn_dict: dict,
    spec_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """Evaluate G(x, J^2 f) analytically using fn_dict derivatives and spec_fn.

    Paper: Definition 2.2 — upper generator consistency check.

    Parameters
    ----------
    x_pts   : (N,) evaluation points
    fn_dict : dict with callables 'f', 'df', 'd2f' (from test_function_library)
    spec_fn : G(x, z, p, H, **kwargs) or G(x, z, p, H) callable
    **kwargs: extra keyword arguments forwarded to spec_fn (e.g. alpha=, sigma_vals=, ...)

    Returns
    -------
    (N,) array of G(x, J^2 f(x)) values.
    """
    x = np.asarray(x_pts, dtype=float)
    f_fn  = fn_dict["f"]
    df_fn = fn_dict["df"]
    d2f_fn = fn_dict["d2f"]
    z = f_fn(x)
    p = df_fn(x)
    H = d2f_fn(x)
    # Call spec with broadcasting; accept both scalar-per-x and vectorized signatures
    try:
        result = spec_fn(x, z, p, H, **kwargs)
    except TypeError:
        # Fallback: call per element for spec functions with positional-only scalar API
        result = np.array([
            float(spec_fn(np.array([xi]), np.array([zi]), np.array([pi]), np.array([Hi]), **kwargs))
            for xi, zi, pi, Hi in zip(x, z, p, H)
        ])
    return np.asarray(result, dtype=float)


def check_generator_bound(
    x_pts: np.ndarray | None = None,
    payoff_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    h: float = 0.05,
    controls_T: list | None = None,
    controls_VG: list | None = None,
    n_paths: int = 500,
    dt: float = 0.01,
    rng: np.random.Generator | None = None,
    *,
    sg_type=None,
    x_grid: np.ndarray | None = None,
    g_fn: Callable | None = None,
    g_kwargs: dict | None = None,
    h_vals: list | None = None,
    eval_kwargs: dict | None = None,
) -> dict:
    """Check G_T(x, J^2 f) <= G(x, J^2 f) pointwise.

    Paper: Section 4.1 — G_T <= G iff T <= V^G (Corollary 4.8).

    Supports two call conventions:

    Old API (used by tests):
        check_generator_bound(x_pts, payoff_fn, h, controls_T, controls_VG,
                              n_paths, dt, rng)
        Returns dict with 'bound_satisfied', 'max_excess', 'G_T', 'G_VG'.

    New API (used by exp3):
        check_generator_bound(sg_type=..., x_grid=..., g_fn=..., g_kwargs=...,
                              h_vals=..., eval_kwargs=...)
        Returns dict with 'bound_satisfied', 'overall_max_violation'.
    """
    from src.upper_expectation import compute_VG

    if sg_type is not None or x_grid is not None:
        # New API: compare G_T (subfam semigroup) against G spec analytically.
        _x = x_grid if x_grid is not None else x_pts
        _gkw = g_kwargs or {}
        _h_vals = h_vals or [0.05, 0.02]
        _ek = eval_kwargs or {}
        controls = _ek.get("controls", controls_T or [])
        _n_paths = _ek.get("n_paths", n_paths)
        _dt = _ek.get("dt", dt)
        _rng = _ek.get("rng", np.random.default_rng(0))

        # Use a Gaussian test payoff to estimate G_T numerically.
        test_payoff = lambda x: np.exp(-0.5 * x ** 2)
        f0 = test_payoff(_x)
        # Analytical derivatives of Gaussian: f'(x)=-x*f, f''(x)=(x^2-1)*f
        grad_f = -_x * f0
        hess_f = (_x ** 2 - 1.0) * f0
        z_vals = f0

        max_viol = -np.inf
        for _h in _h_vals:
            _r = np.random.default_rng(_rng.integers(2**31))
            T_h = compute_VG(_x, _h, test_payoff, controls, _n_paths, _dt, _r)
            G_T_est = (T_h - f0) / _h
            if g_fn is not None:
                G_spec = g_fn(_x, z_vals, grad_f, hess_f, **_gkw)
            else:
                G_spec = np.zeros_like(_x)
            excess = G_T_est - G_spec
            max_viol = max(max_viol, float(np.max(excess)))
        return {
            "bound_satisfied": max_viol < 0.2,
            "overall_max_violation": max_viol,
            "max_excess": max_viol,
        }

    # Old API
    if rng is None:
        rng = np.random.default_rng(0)
    f0 = payoff_fn(x_pts)
    rng1 = np.random.default_rng(rng.integers(2**31))
    rng2 = np.random.default_rng(rng.integers(2**31))
    T_h_T  = compute_VG(x_pts, h, payoff_fn, controls_T or [], n_paths, dt, rng1)
    T_h_VG = compute_VG(x_pts, h, payoff_fn, controls_VG or [], n_paths, dt, rng2)
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
    x_pts    : (N,) state grid
    payoff_fn: test function f
    h_vals   : list of h values for finite-difference estimates
    controls : control dicts defining V^G
    spec_fn  : G(x, z, p, H) — the local specification
    n_paths  : Monte Carlo paths
    dt       : time step
    rng      : numpy Generator

    Returns
    -------
    dict with 'mean_abs_diff', 'max_abs_diff', 'G_T_estimate', 'G_spec'.
    """
    from src.semigroup import SemigroupType
    eval_kw = {"controls": controls, "n_paths": n_paths, "dt": dt, "rng": rng}
    res = estimate_generator(SemigroupType.CONTROLLED, x_pts, payoff_fn, h_vals, eval_kw)
    G_T = res["G_T_estimate"]

    eps = 1e-5
    f0 = payoff_fn(x_pts)
    fp = payoff_fn(x_pts + eps)
    fm = payoff_fn(x_pts - eps)
    f_prime = (fp - fm) / (2 * eps)
    f_double = (fp - 2 * f0 + fm) / eps ** 2
    G_spec = np.array([
        float(spec_fn(np.array([float(x)]), np.array([float(f0[i])]),
                      np.array([float(f_prime[i])]), np.array([float(f_double[i])])))
        for i, x in enumerate(x_pts)
    ])
    abs_diff = np.abs(G_T - G_spec)
    return {
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "G_T_estimate": G_T,
        "G_spec": G_spec,
    }
