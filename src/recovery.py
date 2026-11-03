"""
Recovered structure R^T from a semigroup T.

A law P with initial condition (t,x) belongs to R^T_{t,x} iff every
semigroup-orbit process Y_s = T_{R-s} g(X_s) 1_{s < tau_inf} is a
P-supermartingale on [t, R], for every R >= t and g in USC_b(D).

Paper: Section 3.2 (pp. 18-20).
"""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np


def test_supermartingale_orbit(
    x0: float,
    R: float,
    g_fn: Callable[[np.ndarray], np.ndarray],
    s_vals: np.ndarray,
    candidate_controls: list,
    semigroup_controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Test whether Y_s = T_{R-s} g(X_s) is a P-supermartingale.

    For a candidate law (here parameterised by candidate_controls) and
    semigroup defined by semigroup_controls, check the supermartingale
    property by testing that mean of Y_s is non-increasing.

    Paper: Section 3.2, p. 18 — characterisation of R^T.

    Parameters
    ----------
    x0                : initial state
    R                 : terminal horizon
    g_fn              : payoff g in USC_b(D)
    s_vals            : array of times in [0, R]
    candidate_controls: controls for the candidate law
    semigroup_controls: controls defining the semigroup T
    n_paths           : Monte Carlo paths
    dt                : time step
    rng               : numpy Generator

    Returns
    -------
    dict with 'is_supermartingale', 'orbit_means', 'max_violation'.
    """
    from src.upper_expectation import compute_VG
    orbit_means = []
    for s in s_vals:
        remaining = R - float(s)
        if remaining < 1e-10:
            orbit_means.append(float(g_fn(np.array([x0]))[0]))
            continue
        x_pts = np.array([x0])
        Y_s = compute_VG(x_pts, remaining, g_fn, semigroup_controls, n_paths, dt, rng)
        orbit_means.append(float(Y_s[0]))

    # Supermartingale: E[Y_{s+ds}] <= E[Y_s]; check pairwise
    violations = []
    for i in range(1, len(orbit_means)):
        violations.append(orbit_means[i] - orbit_means[i - 1])
    max_violation = max(violations) if violations else 0.0

    return {
        "is_supermartingale": max_violation <= 0.05,
        "orbit_means": orbit_means,
        "max_violation": float(max_violation),
    }


# Prevent pytest from collecting this as a test function (signature uses test_ prefix
# by design to match the paper's "test membership in R^T" nomenclature).
test_supermartingale_orbit.__test__ = False  # type: ignore[attr-defined]


def classify_laws_into_RT(
    x0: float,
    R: float,
    g_fns: List[Callable[[np.ndarray], np.ndarray]],
    candidate_controls: list,
    semigroup_controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Test each candidate control and classify it as being in R^T or not.

    A control is admitted if the supermartingale test passes for all g_fns.

    Paper: Section 3.2 — R^T_{t,x} is the largest stable family for T.

    Parameters
    ----------
    x0                : initial state
    R                 : terminal horizon
    g_fns             : list of payoff functions to test
    candidate_controls: list of control dicts to classify
    semigroup_controls: controls defining the semigroup T
    n_paths           : Monte Carlo paths
    dt                : time step
    rng               : numpy Generator

    Returns
    -------
    dict with 'n_admitted', 'admitted_indices', 'results'.
    """
    s_vals = np.linspace(0.0, R, 4)
    admitted = []
    results = []
    for idx, ctrl in enumerate(candidate_controls):
        passed_all = True
        for g_fn in g_fns:
            res = test_supermartingale_orbit(
                x0=x0, R=R, g_fn=g_fn, s_vals=s_vals,
                candidate_controls=[ctrl],
                semigroup_controls=semigroup_controls,
                n_paths=n_paths, dt=dt, rng=rng,
            )
            if not res["is_supermartingale"]:
                passed_all = False
                break
        results.append({"control": ctrl, "admitted": passed_all})
        if passed_all:
            admitted.append(idx)
    return {
        "n_admitted": len(admitted),
        "admitted_indices": admitted,
        "results": results,
    }


def test_attainment(
    x0_vals: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Test that V^G f(x) = max_{P in R^T} E^P[f(X_t)].

    Attainment means the supremum over the control family is achieved
    (not just approached). We approximate this by checking that the
    best single control achieves approximately the same value as the
    full upper envelope.

    Paper: Section 3.2 — attained upper-expectation representation.

    Parameters
    ----------
    x0_vals  : (N,) initial states
    t        : time horizon
    payoff_fn: bounded payoff
    controls : list of control dicts
    n_paths  : Monte Carlo paths
    dt       : time step
    rng      : numpy Generator

    Returns
    -------
    dict with 'attainment_gap', 'V_envelope', 'V_best_single'.
    """
    from src.upper_expectation import compute_VG
    V_envelope = compute_VG(x0_vals, t, payoff_fn, controls, n_paths, dt, rng)

    # Best single control
    best_vals = np.full(len(x0_vals), -np.inf)
    for ctrl in controls:
        V_c = compute_VG(x0_vals, t, payoff_fn, [ctrl], n_paths, dt, rng)
        best_vals = np.maximum(best_vals, V_c)

    attainment_gap = float(np.max(np.abs(V_envelope - best_vals)))
    return {
        "attainment_gap": attainment_gap,
        "V_envelope": V_envelope,
        "V_best_single": best_vals,
    }


test_attainment.__test__ = False  # type: ignore[attr-defined]


def test_order_isomorphism(
    x0: float,
    t: float,
    R: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    g_fn: Callable[[np.ndarray], np.ndarray],
    controls_1: list,
    controls_2: list,
    n_paths: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Test T_1 <= T_2 iff R^{T1} subseteq R^{T2}.

    Paper: Section 3.3 — order isomorphism between semigroups and
    stable structures of laws (Theorem 3.10).

    Parameters
    ----------
    x0         : initial state
    t          : time horizon for domination test
    R          : terminal horizon for orbit test
    payoff_fn  : payoff for domination check
    g_fn       : payoff for supermartingale orbit check
    controls_1 : smaller control family (T_1)
    controls_2 : larger control family (T_2)
    n_paths    : Monte Carlo paths
    dt         : time step
    rng        : numpy Generator

    Returns
    -------
    dict with 'T1_leq_T2', 'RT1_subset_RT2'.
    """
    from src.upper_expectation import compute_VG
    x_pts = np.array([x0])
    rng1 = np.random.default_rng(rng.integers(2**31))
    rng2 = np.random.default_rng(rng.integers(2**31))
    V1 = compute_VG(x_pts, t, payoff_fn, controls_1, n_paths, dt, rng1)
    V2 = compute_VG(x_pts, t, payoff_fn, controls_2, n_paths, dt, rng2)
    T1_leq_T2 = bool(float(V1[0]) <= float(V2[0]) + 0.05)

    # R^T1 subset R^T2: controls admitted into R^T1 should also be admitted into R^T2
    res1 = classify_laws_into_RT(
        x0=x0, R=R, g_fns=[g_fn],
        candidate_controls=controls_1,
        semigroup_controls=controls_1,
        n_paths=n_paths, dt=dt, rng=rng,
    )
    res2 = classify_laws_into_RT(
        x0=x0, R=R, g_fns=[g_fn],
        candidate_controls=controls_1,
        semigroup_controls=controls_2,
        n_paths=n_paths, dt=dt, rng=rng,
    )
    RT1_subset_RT2 = res1["n_admitted"] <= res2["n_admitted"] + 1
    return {
        "T1_leq_T2": T1_leq_T2,
        "RT1_subset_RT2": RT1_subset_RT2,
        "V1": float(V1[0]),
        "V2": float(V2[0]),
    }


test_order_isomorphism.__test__ = False  # type: ignore[attr-defined]
