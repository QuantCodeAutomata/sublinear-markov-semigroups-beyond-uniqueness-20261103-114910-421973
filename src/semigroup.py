"""
Semigroup evaluation and property verification.

Paper: Definition 2.1 (axioms V1-V4), Section 1.3 (semigroup map),
       Example 4.1 (Section 4.2), Example 4.2 (Section 4.3).
"""
from __future__ import annotations
from enum import Enum
from typing import Callable, List, Optional
import numpy as np
from src.simulator import (
    simulate_absorbing_branch,
    simulate_zero_sojourn_branch,
    simulate_deterministic_flow,
)


class SemigroupType(Enum):
    ABSORBING = "absorbing"       # T^abs (Example 4.1)
    ZERO_SOJOURN = "zero_sojourn" # T^0   (Example 4.1)
    DETERMINISTIC = "deterministic"  # V^G   (Example 4.2)
    CONTROLLED = "controlled"     # V^G via Monte Carlo supremum


def eval_semigroup(
    x0_vals: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    *,
    stype: SemigroupType,
    controls: Optional[list] = None,
    alpha: float = 0.3,
    n_paths: int = 500,
    dt: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Evaluate T_t f(x) for each x in x0_vals.

    Paper: Section 1.3 — semigroup map; Section 4.2 Example 4.1; Section 4.3 Example 4.2.

    Parameters
    ----------
    x0_vals  : (N,) initial states
    t        : time horizon
    payoff_fn: bounded payoff f
    stype    : which semigroup branch to use
    controls : list of control dicts (for CONTROLLED type)
    alpha    : diffusion exponent (for ABSORBING/ZERO_SOJOURN)
    n_paths  : Monte Carlo paths
    dt       : Euler-Maruyama step
    rng      : numpy Generator

    Returns
    -------
    T_t_f : (N,) array
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(x0_vals)

    if stype == SemigroupType.DETERMINISTIC:
        X_end = simulate_deterministic_flow(x0_vals, t, n_paths=1)
        return payoff_fn(X_end)

    if stype == SemigroupType.CONTROLLED:
        if controls is None:
            raise ValueError("controls must be provided for CONTROLLED semigroup")
        from src.upper_expectation import compute_VG
        return compute_VG(x0_vals, t, payoff_fn, controls, n_paths, dt, rng)

    result = np.zeros(N)
    for i, x0 in enumerate(x0_vals):
        if stype == SemigroupType.ABSORBING:
            X_end, alive = simulate_absorbing_branch(x0, t, dt, alpha, n_paths, rng)
        elif stype == SemigroupType.ZERO_SOJOURN:
            X_end, alive = simulate_zero_sojourn_branch(x0, t, dt, alpha, n_paths, rng)
        else:
            raise ValueError(f"Unknown semigroup type: {stype}")
        payoffs = np.zeros(n_paths)
        if np.any(alive):
            payoffs[alive] = payoff_fn(X_end[alive])
        result[i] = float(np.mean(payoffs))
    return result


def semigroup_orbit_at_time(
    x0: float,
    R: float,
    g: Callable[[np.ndarray], np.ndarray],
    s_vals: np.ndarray,
    controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Compute the semigroup orbit Y_s = T_{R-s} g(X_s) for s in s_vals.

    Paper: p. 4-5 — R^T defined by orbits being P-supermartingales.
    Y_s^{T;R,g} = T_{R-s} g(X_s) 1_{s < tau_inf}.

    Parameters
    ----------
    x0      : initial state
    R       : terminal horizon
    g       : payoff function
    s_vals  : array of times in [0, R]
    controls: control dicts for the semigroup
    n_paths : Monte Carlo paths
    dt      : time step
    rng     : numpy Generator

    Returns
    -------
    dict with 'orbit_means' (means of Y_s at each s), 'orbit_vals' (list of arrays)
    """
    from src.upper_expectation import compute_VG
    orbit_means = []
    for s in s_vals:
        remaining = R - float(s)
        if remaining < 1e-10:
            # Y_R = g(x0) at starting point
            orbit_means.append(float(g(np.array([x0]))[0]))
            continue
        # T_{R-s} g evaluated at x0 (and neighbours to estimate mean)
        x_pts = np.array([x0])
        Y_s = compute_VG(x_pts, remaining, g, controls, n_paths, dt, rng)
        orbit_means.append(float(Y_s[0]))
    return {"orbit_means": orbit_means, "s_vals": list(s_vals)}


def check_domination(
    x_pts: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    full_controls: list,
    sub_controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Numerically test T_sub <= V^G (sub-family dominated by full family).

    Paper: Corollary 4.8 — T <= V^G iff Loc(T) <= G.

    Returns dict with 'dominated', 'max_violation', 'V_full', 'V_sub'.
    """
    from src.upper_expectation import compute_VG
    rng1 = np.random.default_rng(rng.integers(2**31))
    rng2 = np.random.default_rng(rng.integers(2**31))
    V_full = compute_VG(x_pts, t, payoff_fn, full_controls, n_paths, dt, rng1)
    V_sub = compute_VG(x_pts, t, payoff_fn, sub_controls, n_paths, dt, rng2)
    diff = V_full - V_sub
    max_violation = float(np.max(-diff))  # positive = V_sub > V_full (violation)
    return {
        "dominated": max_violation < 0.05,
        "max_violation": max_violation,
        "V_full": V_full,
        "V_sub": V_sub,
    }


def check_feller(
    x_pts: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Check C_b-Feller property: x -> V_t f(x) is continuous.

    Paper: Section 5.3 — V^G maps C_b(D) to C_b(D) under comparison conditions.

    Returns dict with 'is_continuous', 'max_jump', 'V_vals'.
    """
    from src.upper_expectation import compute_VG
    V_vals = compute_VG(x_pts, t, payoff_fn, controls, n_paths, dt, rng)
    # Continuity: max absolute difference between adjacent x-points
    jumps = np.abs(np.diff(V_vals))
    max_jump = float(np.max(jumps)) if len(jumps) > 0 else 0.0
    dx = float(np.diff(x_pts).mean()) if len(x_pts) > 1 else 1.0
    # Threshold: jump > 5*dx is a discontinuity (MC noise tolerance)
    is_continuous = max_jump < 5.0 * dx + 0.2
    return {
        "is_continuous": is_continuous,
        "max_jump": max_jump,
        "V_vals": V_vals,
    }
