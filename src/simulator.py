"""
SDE path simulation via Euler-Maruyama and Cox-process killing.

Paper: Section 2.3 (path-space construction, tau_kill), Section 1.3.
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np


def simulate_paths(
    x0: float,
    sigma: float,
    b: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama paths with constant sigma, b.

    Returns (n_paths, n_steps+1) array.
    """
    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for step in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, step + 1] = paths[:, step] + b * dt + sigma * dW
    return paths


def simulate_paths_with_kill(
    x0: float,
    sigma: float,
    b: float,
    kill_rate: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> dict:
    """Euler-Maruyama with Cox-process killing.

    Returns dict with 'paths' (n_paths, n_steps+1) and 'alive' (n_paths, n_steps+1) bool.
    Once killed, alive stays False.
    """
    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1))
    alive = np.ones((n_paths, n_steps + 1), dtype=bool)
    paths[:, 0] = x0
    still_alive = np.ones(n_paths, dtype=bool)
    cum_kill = np.zeros(n_paths)
    kill_threshold = (
        rng.exponential(scale=1.0, size=n_paths) if kill_rate > 0.0
        else np.full(n_paths, np.inf)
    )
    for step in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, step + 1] = paths[:, step] + b * dt + sigma * dW
        if kill_rate > 0.0:
            cum_kill += kill_rate * dt
            newly_killed = still_alive & (cum_kill >= kill_threshold)
            still_alive[newly_killed] = False
        alive[:, step + 1] = still_alive
    return {"paths": paths, "alive": alive}


def _simulate_with_kill_fn(
    x0: float,
    t_end: float,
    dt: float,
    b_fn: Callable[[np.ndarray], np.ndarray],
    sigma_fn: Callable[[np.ndarray], np.ndarray],
    killing_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    n_paths: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Functional-coefficient Euler-Maruyama with Cox killing.

    Returns (X_end, alive) where X_end is (n_paths,) terminal positions
    and alive is (n_paths,) bool.
    """
    n_steps = max(1, int(round(t_end / dt)))
    dt_act = t_end / n_steps
    sqrt_dt = np.sqrt(dt_act)
    X = np.full(n_paths, float(x0))
    alive = np.ones(n_paths, dtype=bool)
    cum_kill = np.zeros(n_paths)
    kill_threshold = (
        rng.exponential(scale=1.0, size=n_paths) if killing_fn is not None
        else np.full(n_paths, np.inf)
    )
    for _ in range(n_steps):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        x_act = X[idx]
        dW = rng.standard_normal(len(idx)) * sqrt_dt
        X[idx] = x_act + b_fn(x_act) * dt_act + sigma_fn(x_act) * dW
        if killing_fn is not None:
            k_act = np.maximum(killing_fn(x_act), 0.0)
            cum_kill[idx] += k_act * dt_act
            alive[idx[cum_kill[idx] >= kill_threshold[idx]]] = False
    return X, alive


def simulate_absorbing_branch(
    x0: float,
    t_end: float,
    dt: float,
    alpha: float,
    n_paths: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Absorbing branch Q^abs for Example 4.1.

    X stays at 0 if x0=0; otherwise Euler-Maruyama with sigma_alpha.
    Paper: Section 4.2, p. 30-31.
    """
    from src.local_spec import sigma_alpha as _sigma_alpha
    if abs(x0) < 1e-14:
        return np.zeros(n_paths), np.ones(n_paths, dtype=bool)
    def b_fn(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    def sigma_fn(x: np.ndarray) -> np.ndarray:
        return _sigma_alpha(x, alpha)
    return _simulate_with_kill_fn(x0, t_end, dt, b_fn, sigma_fn, None, n_paths, rng)


def simulate_zero_sojourn_branch(
    x0: float,
    t_end: float,
    dt: float,
    alpha: float,
    n_paths: int,
    rng: np.random.Generator,
    epsilon_depart: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-sojourn branch Q^0 for Example 4.1.

    Paper: Section 4.2, p. 30-31.
    """
    from src.local_spec import sigma_alpha as _sigma_alpha
    def b_fn(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    def sigma_fn(x: np.ndarray) -> np.ndarray:
        return _sigma_alpha(x, alpha)
    start = epsilon_depart * rng.choice([-1.0, 1.0], size=n_paths) if abs(x0) < 1e-14 else np.full(n_paths, x0)
    n_steps = max(1, int(round(t_end / dt)))
    dt_act = t_end / n_steps
    sqrt_dt = np.sqrt(dt_act)
    X = start.copy()
    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        X = X + sigma_fn(X) * dW
    return X, np.ones(n_paths, dtype=bool)


def simulate_deterministic_flow(
    x0: float | np.ndarray,
    t: float,
    n_paths: int = 1,
) -> np.ndarray:
    """Deterministic flow X_t = x + t for Example 4.2.

    Paper: Section 4.3, p. 31.
    """
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    return x0_arr + t
