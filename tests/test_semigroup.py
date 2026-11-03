"""
Unit tests for src/semigroup.py.

Verifies semigroup-orbit construction, domination checking, and the
C_b-Feller continuity property for the controlled-diffusion example.

Paper references: Section 3.1 (stable structures), Section 5.3 (Feller).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.semigroup import (
    SemigroupType,
    eval_semigroup,
    semigroup_orbit_at_time,
    check_domination,
    check_feller,
)


@pytest.fixture
def controls_fixture():
    return [
        {"sigma": 0.5, "b": -0.1, "c": 0.0},
        {"sigma": 1.0, "b":  0.0, "c": 0.0},
        {"sigma": 1.5, "b":  0.1, "c": 0.0},
    ]


class TestEvalSemigroup:
    """Tests for eval_semigroup dispatching correct branch."""

    def test_controlled_returns_array(self, controls_fixture):
        x = np.linspace(-2.0, 2.0, 11)
        payoff = lambda u: np.exp(-0.5 * u ** 2)
        V = eval_semigroup(
            x0_vals=x, t=0.2, payoff_fn=payoff,
            stype=SemigroupType.CONTROLLED,
            controls=controls_fixture,
            alpha=0.25,
            n_paths=100, dt=0.05,
            rng=np.random.default_rng(300),
        )
        assert V.shape == x.shape
        assert np.all(np.isfinite(V))

    def test_absorbing_branch_zero_at_origin(self):
        """Absorbing branch: X stays at 0 so payoff that vanishes there gives 0."""
        x = np.array([0.0])
        payoff = lambda u: np.abs(u)  # |0| = 0
        V = eval_semigroup(
            x0_vals=x, t=0.5, payoff_fn=payoff,
            stype=SemigroupType.ABSORBING,
            controls=[{"sigma": 1.0, "b": 0.0, "c": 0.0}],
            alpha=0.25,
            n_paths=50, dt=0.05,
            rng=np.random.default_rng(301),
        )
        assert float(V[0]) == pytest.approx(0.0, abs=1e-10)

    def test_deterministic_shift(self):
        """DETERMINISTIC branch: X_t = x + t (Example 4.2)."""
        x = np.array([0.0, 1.0, -1.0])
        t = 0.5
        payoff = lambda u: u ** 2  # f(x+t) = (x+t)^2
        V = eval_semigroup(
            x0_vals=x, t=t, payoff_fn=payoff,
            stype=SemigroupType.DETERMINISTIC,
            controls=[], alpha=0.25,
            n_paths=1, dt=t,
            rng=np.random.default_rng(302),
        )
        expected = (x + t) ** 2
        assert np.allclose(V, expected, atol=1e-10)


class TestSemigroupOrbit:
    """Tests for semigroup_orbit_at_time used in recovery (R^T)."""

    def test_orbit_supermartingale(self, controls_fixture):
        """Y_s = T_{R-s} g(X_s) should be approximately a P-supermartingale."""
        result = semigroup_orbit_at_time(
            x0=0.5,
            R=0.4,
            g=lambda u: np.exp(-0.5 * u ** 2),
            s_vals=np.array([0.0, 0.1, 0.2, 0.3]),
            controls=controls_fixture,
            n_paths=400,
            dt=0.05,
            rng=np.random.default_rng(310),
        )
        # Mean of Y_s decreasing (supermartingale property)
        means = result["orbit_means"]
        # First step could have MC noise; check overall trend
        assert means[-1] <= means[0] + 0.1, (
            "Semigroup orbit mean should not increase (supermartingale)"
        )


class TestCheckDomination:
    """Tests for check_domination: T_sub <= V^G."""

    def test_sub_dominated_by_full(self, controls_fixture):
        x_pts = np.linspace(-1.0, 1.0, 11)
        sub_controls = controls_fixture[:2]
        result = check_domination(
            x_pts=x_pts,
            t=0.2,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            full_controls=controls_fixture,
            sub_controls=sub_controls,
            n_paths=300, dt=0.05,
            rng=np.random.default_rng(320),
        )
        assert result["dominated"], (
            "Subfamily semigroup must be dominated by full family V^G"
        )


class TestCheckFeller:
    """Tests for C_b-Feller: continuity of x -> V_t f(x)."""

    def test_gaussian_payoff_gives_continuous_output(self, controls_fixture):
        x_pts = np.linspace(-2.0, 2.0, 41)
        result = check_feller(
            x_pts=x_pts, t=0.2,
            payoff_fn=lambda x: np.exp(-0.5 * x ** 2),
            controls=controls_fixture,
            n_paths=300, dt=0.05,
            rng=np.random.default_rng(330),
        )
        assert result["is_continuous"], (
            f"V^G should map C_b -> C_b; max jump {result['max_jump']:.4f}"
        )
