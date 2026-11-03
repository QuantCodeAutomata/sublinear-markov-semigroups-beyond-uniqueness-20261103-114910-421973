"""
Unit tests for src/generator.py.

Verifies finite-difference generator estimation and comparison of the
estimated generator against the prescribed local specification G.

Paper references: Section 1.3 (infinitesimal generator), Section 4.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generator import (
    estimate_generator,
    estimate_generator_analytic,
    check_generator_bound,
    compare_generators_to_spec,
)
from src.local_spec import g_example41


class TestEstimateGenerator:
    """Finite-difference generator estimation."""

    @pytest.fixture
    def controls(self):
        return [
            {"sigma": 0.8, "b": 0.0, "c": 0.0},
            {"sigma": 1.2, "b": 0.0, "c": 0.0},
        ]

    def test_output_shape(self, x_grid_small, controls):
        result = estimate_generator(
            x_pts=x_grid_small,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            h_vals=[0.1, 0.05],
            controls=controls,
            n_paths=200, dt=0.01,
            rng=np.random.default_rng(500),
        )
        assert "gen_estimates" in result
        for h in [0.1, 0.05]:
            assert result["gen_estimates"][h].shape == x_grid_small.shape

    def test_generator_finite_values(self, x_grid_small, controls):
        result = estimate_generator(
            x_pts=x_grid_small,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            h_vals=[0.05],
            controls=controls,
            n_paths=200, dt=0.01,
            rng=np.random.default_rng(501),
        )
        gen_vals = result["gen_estimates"][0.05]
        assert np.all(np.isfinite(gen_vals)), "Generator estimates must be finite"

    def test_generator_zero_diffusion_sigma0(self):
        """For sigma=0, b=0, c=0: generator of any f should be ~0."""
        controls = [{"sigma": 1e-8, "b": 0.0, "c": 0.0}]
        x_pts = np.array([0.0, 0.5, 1.0])
        result = estimate_generator(
            x_pts=x_pts,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            h_vals=[0.05],
            controls=controls,
            n_paths=100, dt=0.01,
            rng=np.random.default_rng(502),
        )
        gen_h = result["gen_estimates"][0.05]
        assert np.all(np.abs(gen_h) < 0.5), (
            "Generator with near-zero diffusion should be near zero"
        )


class TestEstimateGeneratorAnalytic:
    """Analytic (sympy-based) generator estimation for smooth functions."""

    def test_brownian_generator(self):
        """For sigma=1, b=0: generator of f(x) = x^2 is 1 = 0.5 * 2 * sigma^2."""
        result = estimate_generator_analytic(
            x_pts=np.array([0.0]),
            sigma=1.0, b=0.0,
            f_expr="x**2",
        )
        # G f(x) = 0.5 sigma^2 * f''(x) = 0.5 * 1 * 2 = 1
        gen = result["gen_values"]
        assert float(gen[0]) == pytest.approx(1.0, abs=1e-6)

    def test_drift_contribution(self):
        """Generator of f(x) = x with drift b: G f = b * f' = b."""
        b_val = 0.5
        result = estimate_generator_analytic(
            x_pts=np.array([0.0, 1.0, 2.0]),
            sigma=0.0, b=b_val,
            f_expr="x",
        )
        gen = result["gen_values"]
        assert np.allclose(gen, b_val, atol=1e-6)


class TestCheckGeneratorBound:
    """Checks the inequality G_T(x, J^2 f) <= G(x, J^2 f)."""

    def test_sub_family_bound_holds(self):
        """Sub-family generator should be bounded above by full-family generator."""
        controls_full = [
            {"sigma": 0.5, "b": -0.1, "c": 0.0},
            {"sigma": 1.0, "b":  0.0, "c": 0.0},
            {"sigma": 1.5, "b":  0.1, "c": 0.0},
        ]
        controls_sub = controls_full[:2]
        x_pts = np.linspace(-1.0, 1.0, 11)
        result = check_generator_bound(
            x_pts=x_pts,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            h=0.05,
            controls_T=controls_sub,
            controls_VG=controls_full,
            n_paths=300, dt=0.01,
            rng=np.random.default_rng(510),
        )
        assert result["bound_satisfied"], (
            f"G_T <= G_VG failed; max excess = {result['max_excess']:.4f}"
        )


class TestCompareGeneratorsToSpec:
    """Compare estimated generator of a controlled semigroup with the spec G."""

    def test_gen_matches_spec_for_simple_control(self):
        """For a single control (sigma, b, c), generator should match G_spec."""
        controls = [{"sigma": 1.0, "b": 0.0, "c": 0.0}]
        x_pts = np.array([0.0, 0.5, 1.0, -0.5])
        alpha = 0.25
        result = compare_generators_to_spec(
            x_pts=x_pts,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            h_vals=[0.05, 0.02],
            controls=controls,
            spec_fn=lambda x, z, p, H: g_example41(x, z, p, H, alpha=alpha),
            n_paths=400, dt=0.01,
            rng=np.random.default_rng(520),
        )
        assert "mean_abs_diff" in result
        assert result["mean_abs_diff"] < 0.5, (
            "Generator estimate should approximately match the spec"
        )
