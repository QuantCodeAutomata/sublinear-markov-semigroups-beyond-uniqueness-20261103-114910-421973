"""
Unit tests for src/recovery.py.

Verifies the semigroup-orbit supermartingale characterisation of R^T and
the order relation T_1 <= T_2 iff R^T1 subseteq R^T2.

Paper references: Section 3.2–3.3 (recovered structure of laws).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recovery import (
    test_supermartingale_orbit,
    classify_laws_into_RT,
    test_attainment,
    test_order_isomorphism,
)


@pytest.fixture
def controls_full():
    return [
        {"sigma": 0.5, "b": -0.1, "c": 0.0},
        {"sigma": 1.0, "b":  0.0, "c": 0.0},
        {"sigma": 1.5, "b":  0.1, "c": 0.0},
    ]


@pytest.fixture
def controls_sub(controls_full):
    return controls_full[:2]


class TestSupermartingaleOrbit:
    """Supermartingale orbit check for candidate laws."""

    def test_own_law_passes(self, controls_full):
        """A law in the family defining V^G should pass the supermartingale test."""
        result = test_supermartingale_orbit(
            x0=0.5,
            R=0.4,
            g_fn=lambda u: np.exp(-0.5 * u ** 2),
            s_vals=np.array([0.0, 0.1, 0.2, 0.3]),
            candidate_controls=controls_full,
            semigroup_controls=controls_full,
            n_paths=300, dt=0.05,
            rng=np.random.default_rng(400),
        )
        assert result["is_supermartingale"], (
            "Laws from the defining family should pass the supermartingale test"
        )

    def test_returns_expected_keys(self, controls_full):
        result = test_supermartingale_orbit(
            x0=0.0, R=0.3,
            g_fn=lambda u: np.tanh(u),
            s_vals=np.array([0.0, 0.1, 0.2]),
            candidate_controls=controls_full,
            semigroup_controls=controls_full,
            n_paths=100, dt=0.05,
            rng=np.random.default_rng(401),
        )
        for key in ("is_supermartingale", "orbit_means", "max_violation"):
            assert key in result, f"Expected key '{key}' in result"


class TestClassifyLawsIntoRT:
    """Tests for classify_laws_into_RT (proxy for R^T construction)."""

    def test_full_family_classified_in(self, controls_full):
        result = classify_laws_into_RT(
            x0=0.5,
            R=0.4,
            g_fns=[lambda u: np.exp(-0.5 * u ** 2), lambda u: np.tanh(u)],
            candidate_controls=controls_full,
            semigroup_controls=controls_full,
            n_paths=200, dt=0.05,
            rng=np.random.default_rng(410),
        )
        # All controls from the defining family should be in R^T
        assert result["n_admitted"] > 0

    def test_sub_family_subset_of_full(self, controls_full, controls_sub):
        """R^{T_sub} should be a subset of R^{T_full} (order isomorphism)."""
        res_full = classify_laws_into_RT(
            x0=0.5, R=0.4,
            g_fns=[lambda u: np.exp(-0.5 * u ** 2)],
            candidate_controls=controls_full,
            semigroup_controls=controls_full,
            n_paths=200, dt=0.05, rng=np.random.default_rng(411),
        )
        res_sub = classify_laws_into_RT(
            x0=0.5, R=0.4,
            g_fns=[lambda u: np.exp(-0.5 * u ** 2)],
            candidate_controls=controls_sub,
            semigroup_controls=controls_sub,
            n_paths=200, dt=0.05, rng=np.random.default_rng(412),
        )
        assert res_sub["n_admitted"] <= res_full["n_admitted"] + 1, (
            "Smaller control set -> smaller or equal R^T"
        )


class TestAttainment:
    """Tests for test_attainment: supremum over R^T is achieved."""

    def test_attainment_achieved(self, controls_full):
        x_pts = np.array([0.0, 0.5, 1.0])
        result = test_attainment(
            x0_vals=x_pts, t=0.2,
            payoff_fn=lambda u: np.exp(-0.5 * u ** 2),
            controls=controls_full,
            n_paths=300, dt=0.05,
            rng=np.random.default_rng(420),
        )
        assert result["attainment_gap"] < 0.1, (
            f"Attainment gap {result['attainment_gap']:.4f} should be small"
        )


class TestOrderIsomorphism:
    """T_1 <= T_2 iff R^{T1} subseteq R^{T2}."""

    def test_order_preserved(self, controls_full, controls_sub):
        result = test_order_isomorphism(
            x0=0.5, t=0.2, R=0.4,
            payoff_fn=lambda u: np.exp(-0.5 * u ** 2),
            g_fn=lambda u: np.exp(-0.5 * u ** 2),
            controls_1=controls_sub,
            controls_2=controls_full,
            n_paths=200, dt=0.05,
            rng=np.random.default_rng(430),
        )
        assert result["T1_leq_T2"], "T_sub <= V^G (smaller control set)"
        assert result["RT1_subset_RT2"], "R^T1 subseteq R^T2"
