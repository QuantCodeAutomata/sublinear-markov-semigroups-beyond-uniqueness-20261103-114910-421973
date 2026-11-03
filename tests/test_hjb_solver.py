"""
Unit tests for src/hjb_solver.py.

Verifies that the HJB PDE solvers produce stable, consistent, and bounded
output, and that the transport-PDE solver matches the known analytic solution
for Example 4.2 (deterministic flow X_t = x + t).

Paper references: Section 5.2 (Upper Perron characterisation), Section 4.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hjb_solver import solve_hjb_controlled, solve_transport_pde, compare_pde_stochastic


class TestSolveHJBControlled:
    """Tests for the controlled-diffusion HJB solver."""

    @pytest.fixture
    def simple_controls(self):
        return (
            np.array([0.5, 1.0, 1.5]),  # sigma_vals
            np.array([-0.1, 0.0, 0.1]),  # b_vals
            np.array([0.0, 0.0, 0.0]),   # c_vals
        )

    def test_output_contains_u_final(self, x_grid_small, simple_controls):
        sigma_vals, b_vals, c_vals = simple_controls
        result = solve_hjb_controlled(
            f_fn=lambda x: np.exp(-0.5 * x ** 2),
            x_grid=x_grid_small,
            t_max=0.2,
            n_steps=10,
            sigma_vals=sigma_vals,
            b_vals=b_vals,
            c_vals=c_vals,
        )
        assert "u_final" in result
        assert result["u_final"].shape == x_grid_small.shape

    def test_values_bounded_by_payoff(self, x_grid_small, simple_controls):
        sigma_vals, b_vals, c_vals = simple_controls
        payoff = lambda x: np.exp(-0.5 * x ** 2)
        f_init = payoff(x_grid_small)
        result = solve_hjb_controlled(
            f_fn=payoff,
            x_grid=x_grid_small,
            t_max=0.3,
            n_steps=15,
            sigma_vals=sigma_vals,
            b_vals=b_vals,
            c_vals=c_vals,
        )
        u = result["u_final"]
        max_payoff = np.max(f_init)
        # With no killing, u should not exceed max payoff by too much
        assert np.all(u <= max_payoff * 1.1 + 0.05), (
            "PDE solution should not exceed max payoff + small numerical margin"
        )

    def test_constant_payoff_stays_constant(self, x_grid_small, simple_controls):
        sigma_vals, b_vals, c_vals = simple_controls
        c = 0.6
        result = solve_hjb_controlled(
            f_fn=lambda x: c * np.ones_like(x),
            x_grid=x_grid_small,
            t_max=0.2,
            n_steps=10,
            sigma_vals=sigma_vals,
            b_vals=b_vals,
            c_vals=c_vals,
        )
        u = result["u_final"]
        assert np.allclose(u, c, atol=0.02), (
            "HJB solution for constant initial condition should remain constant"
        )

    def test_u_finite_everywhere(self, x_grid_small, simple_controls):
        sigma_vals, b_vals, c_vals = simple_controls
        result = solve_hjb_controlled(
            f_fn=lambda x: np.tanh(x),
            x_grid=x_grid_small,
            t_max=0.2,
            n_steps=10,
            sigma_vals=sigma_vals,
            b_vals=b_vals,
            c_vals=c_vals,
        )
        assert np.all(np.isfinite(result["u_final"]))


class TestSolveTransportPDE:
    """Tests for the transport PDE (deterministic flow) solver."""

    def test_output_shape(self, x_grid_small):
        result = solve_transport_pde(
            f_fn=lambda x: np.exp(-0.5 * x ** 2),
            x_grid=x_grid_small,
            t_max=0.5,
            n_steps=20,
        )
        assert "u_final" in result
        assert result["u_final"].shape == x_grid_small.shape

    def test_matches_analytic_shift_gaussian(self, x_grid_fine):
        """For transport PDE du/dt = du/dx, solution is f(x+t) (advect by +t)."""
        t = 0.4
        f = lambda x: np.exp(-0.5 * x ** 2)
        result = solve_transport_pde(
            f_fn=f,
            x_grid=x_grid_fine,
            t_max=t,
            n_steps=80,
        )
        u_numeric = result["u_final"]
        # On interior points (away from boundary), compare to f(x+t)
        # Use interior subset to avoid boundary effects
        mask = (x_grid_fine > -2.0) & (x_grid_fine < 2.0)
        u_analytic = f(x_grid_fine[mask] + t)
        assert np.allclose(u_numeric[mask], u_analytic, atol=0.05), (
            f"Transport PDE solution should match f(x+t)={f.__name__} shifted by t={t}"
        )

    def test_constant_payoff_unchanged(self, x_grid_small):
        c = 0.8
        result = solve_transport_pde(
            f_fn=lambda x: c * np.ones_like(x),
            x_grid=x_grid_small,
            t_max=0.5,
            n_steps=20,
        )
        assert np.allclose(result["u_final"], c, atol=1e-10)


class TestComparePDEStochastic:
    """Tests for comparison between PDE and stochastic (Monte Carlo) semigroup."""

    def test_returns_discrepancy_metric(self):
        x_pts = np.linspace(-1.0, 1.0, 11)
        sigma_vals = np.array([0.8, 1.2])
        b_vals     = np.array([0.0, 0.0])
        c_vals     = np.array([0.0, 0.0])
        controls = [{"sigma": s, "b": b, "c": c}
                    for s, b, c in zip(sigma_vals.tolist(), b_vals.tolist(), c_vals.tolist())]
        result = compare_pde_stochastic(
            x_pts=x_pts,
            t=0.2,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            controls=controls,
            n_paths=300, dt=0.02, n_pde_steps=10,
            rng=np.random.default_rng(600),
        )
        assert "max_abs_err" in result
        assert "mean_abs_err" in result
        assert result["max_abs_err"] >= 0.0
        assert result["mean_abs_err"] >= 0.0
