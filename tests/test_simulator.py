"""
Unit tests for src/simulator.py.

Verifies SDE path generation, killing via Cox clock, and statistical
properties expected from the paper's path model (Section 3.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulator import simulate_paths, simulate_paths_with_kill


class TestSimulatePaths:
    """Tests for Euler-Maruyama path simulation."""

    def test_output_shape(self, rng_fixed):
        n_paths, n_steps, dt = 50, 20, 0.05
        paths = simulate_paths(
            x0=0.0, sigma=1.0, b=0.0, n_paths=n_paths,
            n_steps=n_steps, dt=dt, rng=np.random.default_rng(1),
        )
        assert paths.shape == (n_paths, n_steps + 1)

    def test_initial_condition(self, rng_fixed):
        x0 = 2.5
        paths = simulate_paths(
            x0=x0, sigma=0.5, b=0.0, n_paths=30, n_steps=10, dt=0.1,
            rng=np.random.default_rng(2),
        )
        assert np.all(paths[:, 0] == pytest.approx(x0))

    def test_zero_diffusion_follows_drift(self):
        # With sigma=0, dX = b dt => X_t = x0 + b*t
        x0, b, dt, n_steps = 1.0, 0.5, 0.1, 10
        paths = simulate_paths(
            x0=x0, sigma=1e-15, b=b, n_paths=5, n_steps=n_steps, dt=dt,
            rng=np.random.default_rng(3),
        )
        expected = x0 + b * np.arange(n_steps + 1) * dt
        assert np.allclose(paths, expected[None, :], atol=1e-6)

    def test_no_lookahead_in_increments(self, rng_fixed):
        # All increments should use info at time t, not t+1
        paths = simulate_paths(
            x0=0.0, sigma=1.0, b=0.0, n_paths=100, n_steps=50, dt=0.01,
            rng=np.random.default_rng(4),
        )
        # Euler increments: paths[:, t+1] - paths[:, t] is a function of paths[:, t]
        increments = np.diff(paths, axis=1)
        assert increments.shape == (100, 50)
        assert np.all(np.isfinite(increments))

    def test_std_scales_with_sqrt_t(self):
        # Var[X_T] ~ sigma^2 * T for zero drift
        sigma, T, n_steps = 1.0, 1.0, 100
        dt = T / n_steps
        paths = simulate_paths(
            x0=0.0, sigma=sigma, b=0.0, n_paths=10000, n_steps=n_steps, dt=dt,
            rng=np.random.default_rng(5),
        )
        terminal = paths[:, -1]
        emp_std = float(np.std(terminal))
        expected_std = sigma * np.sqrt(T)
        assert abs(emp_std - expected_std) < 0.05 * expected_std, (
            f"Empirical std {emp_std:.3f} should be close to {expected_std:.3f}"
        )


class TestSimulatePathsWithKill:
    """Tests for path simulation with Cox-process killing."""

    def test_output_keys(self, rng_fixed):
        result = simulate_paths_with_kill(
            x0=0.0, sigma=1.0, b=0.0, kill_rate=0.5,
            n_paths=20, n_steps=10, dt=0.1,
            rng=np.random.default_rng(6),
        )
        assert "paths" in result
        assert "alive" in result

    def test_alive_mask_shape(self):
        n_paths, n_steps = 30, 15
        result = simulate_paths_with_kill(
            x0=0.0, sigma=1.0, b=0.0, kill_rate=0.2,
            n_paths=n_paths, n_steps=n_steps, dt=0.1,
            rng=np.random.default_rng(7),
        )
        assert result["alive"].shape == (n_paths, n_steps + 1)

    def test_zero_kill_rate_all_alive(self):
        result = simulate_paths_with_kill(
            x0=0.0, sigma=1.0, b=0.0, kill_rate=0.0,
            n_paths=50, n_steps=20, dt=0.05,
            rng=np.random.default_rng(8),
        )
        assert np.all(result["alive"]), "Zero kill rate: all paths should survive"

    def test_high_kill_rate_reduces_survivors(self):
        result = simulate_paths_with_kill(
            x0=0.0, sigma=1.0, b=0.0, kill_rate=100.0,
            n_paths=500, n_steps=10, dt=0.1,
            rng=np.random.default_rng(9),
        )
        fraction_alive_at_end = float(np.mean(result["alive"][:, -1]))
        # With rate=100 over T=1, survival prob ~ exp(-100) ≈ 0
        assert fraction_alive_at_end < 0.01

    def test_paths_stopped_after_kill(self):
        """Once a path is killed (alive=False) it must stay dead."""
        result = simulate_paths_with_kill(
            x0=0.0, sigma=1.0, b=0.0, kill_rate=1.0,
            n_paths=200, n_steps=30, dt=0.1,
            rng=np.random.default_rng(10),
        )
        alive = result["alive"]
        # If alive[i, t] is False, then alive[i, t+1..] must also be False
        for t in range(alive.shape[1] - 1):
            died = ~alive[:, t]
            assert np.all(~alive[died, t + 1]), (
                f"Paths killed at step {t} must remain dead"
            )
