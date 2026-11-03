"""
Unit tests for src/upper_expectation.py.

Verifies that compute_VG satisfies the key properties of the canonical
semigroup V^G: monotonicity, sublinearity, domination over sub-families,
and that it returns values in the feasible payoff range.

Paper references: Section 3.2 (upper-expectation representation).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.upper_expectation import compute_VG, check_semigroup_property, check_semigroup_sublinearity


class TestComputeVG:
    """Core tests for the upper-expectation semigroup V_t^G f(x)."""

    @pytest.fixture
    def controls(self):
        return [
            {"sigma": 0.5, "b": -0.1, "c": 0.0},
            {"sigma": 1.0, "b":  0.0, "c": 0.0},
            {"sigma": 1.5, "b":  0.1, "c": 0.0},
        ]

    @pytest.fixture
    def x_pts(self):
        return np.linspace(-2.0, 2.0, 21)

    def test_output_shape(self, x_pts, controls):
        payoff = lambda x: np.exp(-0.5 * x ** 2)
        V = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=payoff, controls=controls,
            n_paths=100, dt=0.05, rng=np.random.default_rng(100),
        )
        assert V.shape == x_pts.shape, "V must have same shape as x0_vals"

    def test_values_in_range(self, x_pts, controls):
        # f in [0, 1] => V_t f in [0, 1]
        payoff = lambda x: np.exp(-0.5 * x ** 2)  # in (0, 1]
        V = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=payoff, controls=controls,
            n_paths=200, dt=0.05, rng=np.random.default_rng(101),
        )
        assert np.all(V >= -1e-9), "V_t f >= inf f = 0 when f >= 0"
        assert np.all(V <= 1.0 + 1e-9), "V_t f <= sup f = 1 when f in [0,1]"

    def test_monotonicity(self, x_pts, controls):
        # f1 <= f2 => V_t f1 <= V_t f2
        f1 = lambda x: 0.5 * np.exp(-0.5 * x ** 2)
        f2 = lambda x: np.exp(-0.5 * x ** 2)
        V1 = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=f1, controls=controls,
            n_paths=200, dt=0.05, rng=np.random.default_rng(102),
        )
        V2 = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=f2, controls=controls,
            n_paths=200, dt=0.05, rng=np.random.default_rng(103),
        )
        assert np.all(V1 <= V2 + 0.05), "V monotone: V(f1) <= V(f2) when f1 <= f2"

    def test_constant_payoff(self, x_pts, controls):
        # f = c => V_t f = c (with no killing; c_theta=0)
        c = 0.7
        payoff = lambda x: c * np.ones_like(x)
        V = compute_VG(
            x0_vals=x_pts, t=0.3, payoff_fn=payoff, controls=controls,
            n_paths=100, dt=0.05, rng=np.random.default_rng(104),
        )
        assert np.allclose(V, c, atol=0.02), "V_t(constant) = constant when no killing"

    def test_larger_family_dominates_smaller(self, x_pts):
        # V^G with more controls >= V^G with fewer controls
        all_controls = [
            {"sigma": 0.5, "b": -0.1, "c": 0.0},
            {"sigma": 1.0, "b":  0.0, "c": 0.0},
            {"sigma": 1.5, "b":  0.1, "c": 0.0},
            {"sigma": 2.0, "b":  0.2, "c": 0.0},
        ]
        sub_controls = all_controls[:2]
        payoff = lambda x: np.tanh(x)
        V_full = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=payoff, controls=all_controls,
            n_paths=300, dt=0.05, rng=np.random.default_rng(105),
        )
        V_sub = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=payoff, controls=sub_controls,
            n_paths=300, dt=0.05, rng=np.random.default_rng(106),
        )
        # V_full >= V_sub (with Monte Carlo tolerance)
        assert np.all(V_sub <= V_full + 0.05), (
            "Larger control family must dominate smaller (Theorem 4.2)"
        )

    def test_sublinearity(self, x_pts, controls):
        """V_t(lam*f) = lam * V_t(f) for lam > 0 (positive homogeneity)."""
        lam = 2.0
        payoff = lambda x: np.exp(-0.5 * x ** 2)
        V_f = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=payoff, controls=controls,
            n_paths=200, dt=0.05, rng=np.random.default_rng(107),
        )
        V_lf = compute_VG(
            x0_vals=x_pts, t=0.2, payoff_fn=lambda x: lam * payoff(x), controls=controls,
            n_paths=200, dt=0.05, rng=np.random.default_rng(107),  # same seed
        )
        assert np.allclose(V_lf, lam * V_f, atol=0.05), (
            "V_t(lam*f) = lam * V_t(f) (positive homogeneity)"
        )


class TestSemigroupProperty:
    """Tests for time-consistency V_{s+t} = V_s ∘ V_t."""

    @pytest.fixture
    def controls(self):
        return [
            {"sigma": 0.8, "b": 0.0, "c": 0.0},
            {"sigma": 1.2, "b": 0.0, "c": 0.0},
        ]

    def test_time_consistency_approx(self, controls):
        """V_{s+t} f ≈ V_s(V_t f) within Monte Carlo error (paper Sec 3.2)."""
        x_pts = np.linspace(-1.5, 1.5, 11)
        result = check_semigroup_property(
            x_pts=x_pts,
            s=0.15,
            t=0.10,
            payoff_fn=lambda x: np.exp(-0.5 * x**2),
            controls=controls,
            n_paths=500,
            dt=0.05,
            rng=np.random.default_rng(200),
        )
        # max abs error should be within 3 * Monte Carlo standard error
        assert result["max_abs_err"] < 0.15, (
            f"Semigroup time-consistency error {result['max_abs_err']:.4f} too large"
        )
