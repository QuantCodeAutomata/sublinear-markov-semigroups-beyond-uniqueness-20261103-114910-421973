"""
Synthetic data generation for the experiments.

Generates controlled-diffusion path ensembles, test payoff functions,
and control grids used across experiments.  No external data is required;
all inputs are analytically or numerically defined.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from src import config
from src.local_spec import (
    sigma_alpha,
    a_alpha,
    phi_quadratic,
    grad_phi_quadratic,
    hess_phi_quadratic,
)
from src.simulator import simulate_paths_with_kill, simulate_deterministic_flow
from src.upper_expectation import make_default_control_grid


# ---------------------------------------------------------------------------
# Control grid factory
# ---------------------------------------------------------------------------

def make_control_grid(full_run: bool = False) -> list[dict]:
    """Create the discrete control grid for the controlled-diffusion model.

    Returns a list of {'sigma', 'b', 'c'} dicts.  Size controlled by config.
    """
    n_theta = config.N_THETA_FULL if full_run else config.N_THETA
    return make_default_control_grid(
        sigma_min=config.CTRL_SIGMA_MIN,
        sigma_max=config.CTRL_SIGMA_MAX,
        b_min=config.CTRL_B_MIN,
        b_max=config.CTRL_B_MAX,
        c_max=config.CTRL_C_MAX,
        n_theta=n_theta,
    )


# ---------------------------------------------------------------------------
# Standard payoff functions
# ---------------------------------------------------------------------------

def payoff_f_plus(x: np.ndarray) -> np.ndarray:
    """f_+(x) = x^2 / (1 + x^2) — test payoff from Example 4.1.

    Paper: Section 4.2, p. 31.
    """
    return x ** 2 / (1.0 + x ** 2)


def payoff_gaussian(x: np.ndarray) -> np.ndarray:
    """f(x) = exp(-x^2/2) — bounded smooth test payoff."""
    return np.exp(-x ** 2 / 2.0)


def payoff_tanh(x: np.ndarray) -> np.ndarray:
    """f(x) = tanh(x) — bounded monotone smooth test payoff."""
    return np.tanh(x)


def payoff_sin_env(x: np.ndarray) -> np.ndarray:
    """f(x) = sin(x) * exp(-x^2/4) — oscillatory bounded payoff."""
    return np.sin(x) * np.exp(-x ** 2 / 4.0)


def payoff_const(x: np.ndarray) -> np.ndarray:
    """f(x) = 0.5 — constant payoff (tests boundedness)."""
    return 0.5 * np.ones_like(x)


PAYOFF_LIBRARY: list[tuple[str, Callable]] = [
    ("f_plus", payoff_f_plus),
    ("gaussian", payoff_gaussian),
    ("tanh", payoff_tanh),
    ("sin_env", payoff_sin_env),
    ("const", payoff_const),
]


# ---------------------------------------------------------------------------
# State and time grids
# ---------------------------------------------------------------------------

def make_x_grid(n_x: int = config.N_X) -> np.ndarray:
    """Uniform spatial grid on [X_MIN, X_MAX]."""
    return np.linspace(config.X_MIN, config.X_MAX, n_x)


def make_t_eval(
    n_t: int = config.N_T_EVAL,
    t_min: float = config.T_EVAL_MIN,
    t_max: float = config.T_EVAL_MAX,
) -> np.ndarray:
    """Evaluation time points."""
    return np.linspace(t_min, t_max, n_t)


# ---------------------------------------------------------------------------
# Path generation for Example 4.1
# ---------------------------------------------------------------------------

def generate_absorbing_paths(
    x0: float,
    t_end: float,
    n_paths: int,
    rng: np.random.Generator,
    alpha: float = config.ALPHA,
    dt: float = config.DT,
) -> dict:
    """Generate paths under the absorbing branch Q^abs for Example 4.1.

    Paper: Section 4.2, p. 30 — absorbing Feller family Q_x^abs.
    """
    from src.simulator import simulate_absorbing_branch
    X_end, alive = simulate_absorbing_branch(
        x0=x0, t_end=t_end, dt=dt, alpha=alpha, n_paths=n_paths, rng=rng
    )
    return {"X_end": X_end, "alive": alive, "branch": "absorbing", "x0": x0}


def generate_zero_sojourn_paths(
    x0: float,
    t_end: float,
    n_paths: int,
    rng: np.random.Generator,
    alpha: float = config.ALPHA,
    dt: float = config.DT,
) -> dict:
    """Generate paths under the zero-sojourn branch Q^0 for Example 4.1.

    Paper: Section 4.2, p. 30 — zero-sojourn Feller family Q_x^0.
    """
    from src.simulator import simulate_zero_sojourn_branch
    X_end, alive = simulate_zero_sojourn_branch(
        x0=x0, t_end=t_end, dt=dt, alpha=alpha, n_paths=n_paths, rng=rng
    )
    return {"X_end": X_end, "alive": alive, "branch": "zero_sojourn", "x0": x0}


# ---------------------------------------------------------------------------
# Path generation for controlled-diffusion model
# ---------------------------------------------------------------------------

def generate_controlled_paths(
    x0: float,
    t_end: float,
    control: dict,
    n_paths: int,
    rng: np.random.Generator,
    dt: float = config.DT,
) -> dict:
    """Generate paths under a single constant control for the controlled-diffusion model."""
    sigma_c = control["sigma"]
    b_c = control["b"]
    c_c = control["c"]

    def b_fn(x: np.ndarray) -> np.ndarray:
        return b_c * np.ones_like(x)

    def sigma_fn(x: np.ndarray) -> np.ndarray:
        return sigma_c * np.ones_like(x)

    killing_fn = None
    if c_c < 0.0:
        def killing_fn(x: np.ndarray, _k=-c_c) -> np.ndarray:  # type: ignore
            return _k * np.ones_like(x)

    X_end, alive = simulate_paths_with_kill(
        x0=x0, t_end=t_end, dt=dt,
        b_fn=b_fn, sigma_fn=sigma_fn, killing_fn=killing_fn,
        n_paths=n_paths, rng=rng,
    )
    return {"X_end": X_end, "alive": alive, "control": control, "x0": x0}


# ---------------------------------------------------------------------------
# Lyapunov objects for structural verification
# ---------------------------------------------------------------------------

def lyapunov_data(x_grid: np.ndarray, alpha: float = config.ALPHA) -> dict:
    """Return phi, grad phi, Hess phi and Lyapunov bound for Examples 4.1 and 4.2.

    phi(x) = 1 + x^2; G_alpha(x, J^2 phi) = a_alpha(x) <= phi(x).

    Paper: Section 4.2 — 'G_alpha in G_Lyap: for phi(x)=1+x^2, G_alpha(x,J^2 phi)=a_alpha(x) <= phi(x)'.
    """
    phi = phi_quadratic(x_grid)
    grad_phi = grad_phi_quadratic(x_grid)
    hess_phi = hess_phi_quadratic(x_grid)
    G_phi_41 = 0.5 * a_alpha(x_grid, alpha) * hess_phi
    return {
        "phi": phi,
        "grad_phi": grad_phi,
        "hess_phi": hess_phi,
        "G_phi_example41": G_phi_41,
        "C_phi_example41": 1.0,  # a_alpha(x) <= 1 = phi(x)/phi(x) * phi(x) effectively C_phi=1
    }
