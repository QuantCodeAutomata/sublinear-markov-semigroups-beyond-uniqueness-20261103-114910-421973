"""
Monotone finite-difference HJB solver for the Cauchy problem
    partial_t u - G(x, u, Du, D^2 u) = 0   on (0,T) x D
    u(0, .) = f                              (initial condition)

The scheme is explicit in time with a monotone discretisation of the
second-order term (upwind for first-order terms and central for second-order).
A discrete control set Theta is used; at each grid node the supremum over
controls defines the right-hand side.

Paper: Section 5.2 (upper Perron characterisation, Theorem 5.3, p. 35).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 1-D HJB solver for controlled diffusions
# ---------------------------------------------------------------------------

def solve_hjb_controlled(
    f_fn: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    t_max: float,
    n_steps: int,
    sigma_vals: np.ndarray,
    b_vals: np.ndarray,
    c_vals: np.ndarray,
    boundary_val: float = 0.0,
) -> dict:
    """Solve the HJB Cauchy problem for a 1-D controlled diffusion.

    PDE: partial_t u - max_{theta in Theta} [0.5*sigma^2*u_xx + b*u_x + c*u] = 0
    subject to u(0, .) = f  (note: t here is forward time; u(t, .) = V_t^G f(x)).

    The scheme is explicit-in-time with:
      - central differences for u_xx
      - upwind differences for u_x (sign determined by b)
      - local control supremum

    CFL condition: dt <= dx^2 / max(sigma^2).

    Paper: Section 5.2 — PDE characterisation and Theorem 5.3.

    Parameters
    ----------
    f_fn        : initial payoff f(x)
    x_grid      : (N_x,) uniform spatial grid
    t_max       : terminal time T
    n_steps     : number of time steps
    sigma_vals  : (N_theta,) diffusion coefficients
    b_vals      : (N_theta,) drift values
    c_vals      : (N_theta,) <= 0 killing rates
    boundary_val: Dirichlet BC at domain boundaries (approximate)

    Returns
    -------
    dict with 'u_final' (N_x,), 'u_history' (n_steps+1, N_x), 't_grid'.
    """
    dx = x_grid[1] - x_grid[0]
    dt = t_max / n_steps
    Nx = len(x_grid)

    # CFL stability check: dt <= 0.5*dx^2 / max(sigma^2)
    max_sigma2 = float(np.max(sigma_vals ** 2))
    dt_cfl = 0.5 * dx ** 2 / max(max_sigma2, 1e-8)
    if dt > dt_cfl:
        # Reduce dt to satisfy CFL
        n_steps = max(int(np.ceil(t_max / dt_cfl)) + 1, n_steps)
        dt = t_max / n_steps

    u = f_fn(x_grid).copy()
    u_history = [u.copy()]
    t_grid = np.linspace(0.0, t_max, n_steps + 1)

    # Boundary values from initial condition (preserve f at domain edges)
    bc_left  = float(f_fn(np.array([x_grid[0]]))[0])
    bc_right = float(f_fn(np.array([x_grid[-1]]))[0])

    # Pre-compute finite-difference operators
    # Interior indices
    interior = np.arange(1, Nx - 1)

    for step in range(n_steps):
        u_old = u.copy()

        # Central second differences: u_xx at interior points
        u_xx = (u_old[interior + 1] - 2.0 * u_old[interior] + u_old[interior - 1]) / dx ** 2

        # Upwind first differences: u_x at interior points
        u_x_forward = (u_old[interior + 1] - u_old[interior]) / dx
        u_x_backward = (u_old[interior] - u_old[interior - 1]) / dx

        # Compute local HJB right-hand side: sup_{theta} [0.5*sigma^2*u_xx + b*u_x + c*u]
        rhs = np.full(len(interior), -np.inf)
        for sigma_c, b_c, c_c in zip(sigma_vals, b_vals, c_vals):
            # Upwind for b: use forward diff if b>0, backward if b<0
            u_x = np.where(b_c >= 0.0, u_x_forward, u_x_backward)
            contrib = (
                0.5 * sigma_c ** 2 * u_xx
                + b_c * u_x
                + c_c * u_old[interior]
            )
            rhs = np.maximum(rhs, contrib)

        # Explicit Euler update
        u[interior] = u_old[interior] + dt * rhs

        # Boundary conditions: hold at initial-condition value
        u[0]  = bc_left
        u[-1] = bc_right

        # Clamp to payoff bounds (sub-Markov: ||T_t f|| <= ||f||)
        max_f = float(np.max(np.abs(f_fn(x_grid))))
        u = np.clip(u, -max_f, max_f)

        u_history.append(u.copy())

    return {
        "u_final": u,
        "u_history": np.array(u_history),
        "t_grid": t_grid,
        "x_grid": x_grid,
        "n_steps": n_steps,
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# 1-D HJB solver for Example 4.2 (first-order transport PDE)
# ---------------------------------------------------------------------------

def solve_transport_pde(
    f_fn: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    t_max: float,
    n_steps: int,
) -> dict:
    """Solve partial_t u - u_x = 0 (transport, velocity 1) with upwind scheme.

    This is the HJB equation for Example 4.2: G^downarrow(x,z,p,H) = p = u_x.
    The exact solution is u(t, x) = f(x + t).

    Paper: Example 4.2, p. 31 — V_t^G f(x) = f(x + t); G^downarrow = p.

    Returns
    -------
    dict with 'u_final', 'u_exact', 'u_history', 't_grid', 'max_error'.
    """
    dx = x_grid[1] - x_grid[0]
    dt = t_max / n_steps
    Nx = len(x_grid)

    # CFL for upwind: dt <= dx
    dt_cfl = dx
    if dt > dt_cfl:
        n_steps = max(int(np.ceil(t_max / dt_cfl)) + 1, n_steps)
        dt = t_max / n_steps

    u = f_fn(x_grid).copy()
    t_grid = np.linspace(0.0, t_max, n_steps + 1)
    u_history = [u.copy()]

    for step in range(n_steps):
        u_old = u.copy()
        # Upwind: PDE is u_t = u_x (drift = -1), solution u(t,x) = f(x+t).
        # For left-moving wave (c=-1 in u_t + c*u_x=0), upwind uses forward differences.
        u_interior = np.arange(0, Nx - 1)
        u[u_interior] = u_old[u_interior] + dt / dx * (u_old[u_interior + 1] - u_old[u_interior])
        # Right boundary: use inflow from analytic continuation
        u[-1] = f_fn(np.array([x_grid[-1] + dt * (step + 1)]))[0]
        u_history.append(u.copy())

    # Exact solution
    u_exact = f_fn(x_grid + t_max)

    return {
        "u_final": u,
        "u_exact": u_exact,
        "u_history": np.array(u_history),
        "t_grid": t_grid,
        "x_grid": x_grid,
        "max_error": float(np.max(np.abs(u - u_exact))),
    }


# ---------------------------------------------------------------------------
# Comparison: PDE solution vs stochastic upper-expectation
# ---------------------------------------------------------------------------

def compare_pde_stochastic(
    x_pts: np.ndarray,
    t: float,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    controls: list,
    n_paths: int,
    dt: float,
    n_pde_steps: int,
    rng: np.random.Generator,
) -> dict:
    """Compare PDE solution with stochastic upper-expectation semigroup.

    Paper: Section 5.2 — V^G equals the greatest bounded USC viscosity subsolution
    of the HJB Cauchy problem.

    Parameters
    ----------
    x_pts      : (N,) state grid
    t          : time horizon
    payoff_fn  : bounded payoff (initial condition for PDE)
    controls   : list of control dicts for stochastic V^G
    n_paths    : Monte Carlo paths
    dt         : SDE time step
    n_pde_steps: number of PDE time steps
    rng        : numpy Generator

    Returns
    -------
    dict with 'max_abs_err', 'mean_abs_err', 'u_pde', 'u_stoch'.
    """
    from src.upper_expectation import compute_VG
    sigma_vals = np.array([float(c["sigma"]) for c in controls])
    b_vals     = np.array([float(c["b"]) for c in controls])
    c_vals     = np.array([float(c["c"]) for c in controls])
    pde_res = solve_hjb_controlled(
        f_fn=payoff_fn,
        x_grid=x_pts,
        t_max=t,
        n_steps=n_pde_steps,
        sigma_vals=sigma_vals,
        b_vals=b_vals,
        c_vals=c_vals,
    )
    u_pde = pde_res["u_final"]
    u_stoch = compute_VG(x_pts, t, payoff_fn, controls, n_paths, dt, rng)
    abs_err = np.abs(u_pde - u_stoch)
    return {
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "u_pde": u_pde,
        "u_stoch": u_stoch,
    }
