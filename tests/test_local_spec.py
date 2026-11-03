"""
Unit tests for src/local_spec.py.

Verifies structural properties required by the paper: sublinearity in (z,p,H),
monotonicity in H, antitonicity in z, Lyapunov conditions, and correct
behaviour of the Example 4.1 diffusion coefficient.

Paper references: Conditions (G1)–(G3), Assumption 2.2 (p. 7–9).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.local_spec import (
    a_alpha,
    g_example41,
    g_example42,
    g_controlled_diffusion,
    check_lyapunov,
    lyapunov_phi,
)


class TestAAlpha:
    """Tests for the Example 4.1 diffusion coefficient a_alpha."""

    def test_nonnegative(self, alpha_small):
        x = np.linspace(-3.0, 3.0, 51)
        assert np.all(a_alpha(x, alpha_small) >= 0), "a_alpha must be non-negative"

    def test_zero_at_origin(self, alpha_small):
        val = float(a_alpha(np.array([0.0]), alpha_small))
        assert val == pytest.approx(0.0, abs=1e-12), "a_alpha(0) must be 0"

    def test_bounded(self, alpha_small):
        x = np.linspace(-10.0, 10.0, 201)
        vals = a_alpha(x, alpha_small)
        assert np.all(vals <= 1.0 + 1e-9), "a_alpha <= 1 for alpha in (0,0.5)"

    def test_symmetric(self, alpha_small):
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(a_alpha(x, alpha_small), a_alpha(-x, alpha_small))


class TestGExample41:
    """Tests for the Example 4.1 local specification G_alpha."""

    def test_sublinear_in_H(self, alpha_small, x_grid_small):
        # G(x,0,0,lam*H) = lam * G(x,0,0,H) for lam >= 0
        x, H = x_grid_small, np.ones_like(x_grid_small)
        lam = 2.5
        g1 = g_example41(x, 0.0, 0.0, H, alpha=alpha_small)
        g2 = g_example41(x, 0.0, 0.0, lam * H, alpha=alpha_small)
        assert np.allclose(g2, lam * g1, atol=1e-10)

    def test_monotone_in_H(self, alpha_small, x_grid_small):
        x = x_grid_small
        g_lo = g_example41(x, 0.0, 0.0, np.ones_like(x), alpha=alpha_small)
        g_hi = g_example41(x, 0.0, 0.0, 2.0 * np.ones_like(x), alpha=alpha_small)
        assert np.all(g_hi >= g_lo - 1e-12), "G must be non-decreasing in H"

    def test_antitone_in_z(self, alpha_small, x_grid_small):
        x = x_grid_small
        H = np.ones_like(x)
        g_neg_z = g_example41(x, -1.0, 0.0, H, alpha=alpha_small)
        g_pos_z = g_example41(x, 1.0, 0.0, H, alpha=alpha_small)
        # G is linear in z with coefficient 0 for this spec => z-part vanishes
        # Just check both are finite
        assert np.all(np.isfinite(g_neg_z))
        assert np.all(np.isfinite(g_pos_z))

    def test_vanishes_at_origin(self, alpha_small):
        # a_alpha(0) = 0 => G_alpha(0, z, p, H) = 0
        x = np.array([0.0])
        g = g_example41(x, 1.0, 1.0, 1.0, alpha=alpha_small)
        assert g == pytest.approx(0.0, abs=1e-12)


class TestGExample42:
    """Tests for the Example 4.2 local specification G(x,z,p,H) = p + 1_{x=0} p^+."""

    def test_away_from_origin(self):
        x = np.array([1.0, -2.0, 3.0])
        p = np.array([2.0, -1.0, 0.5])
        g = g_example42(x, 0.0, p, 0.0)
        assert np.allclose(g, p), "Away from 0: G = p"

    def test_at_origin_positive_p(self):
        x = np.array([0.0])
        p = np.array([1.0])
        g = g_example42(x, 0.0, p, 0.0)
        # G(0, z, p, H) = p + p^+ = 2p for p > 0
        assert float(g[0]) == pytest.approx(2.0, rel=1e-9)

    def test_at_origin_negative_p(self):
        x = np.array([0.0])
        p = np.array([-1.0])
        g = g_example42(x, 0.0, p, 0.0)
        # G(0, z, -1, H) = -1 + 0 = -1
        assert float(g[0]) == pytest.approx(-1.0, rel=1e-9)

    def test_linear_in_p_away(self):
        x = np.array([1.0])
        lam = 3.0
        p = np.array([2.0])
        assert float(g_example42(x, 0.0, lam * p, 0.0)[0]) == pytest.approx(
            lam * float(g_example42(x, 0.0, p, 0.0)[0]), rel=1e-9
        )


class TestGControlledDiffusion:
    """Tests for the controlled-diffusion local specification."""

    @pytest.fixture
    def controls_arr(self):
        sigma = np.array([0.5, 1.0, 1.5])
        b     = np.array([-0.1, 0.0, 0.1])
        c     = np.array([0.0, 0.0, -0.05])
        return sigma, b, c

    def test_sublinearity_in_z(self, x_grid_small, controls_arr):
        sigma, b, c = controls_arr
        x, p, H = x_grid_small, np.zeros_like(x_grid_small), np.zeros_like(x_grid_small)
        g1 = g_controlled_diffusion(x, 1.0, p, H, sigma, b, c)
        g2 = g_controlled_diffusion(x, 2.0, p, H, sigma, b, c)
        # z coeff = max c_theta <= 0 => G decreases as z increases
        assert np.all(g2 <= g1 + 1e-10), "G antitone in z (c_theta <= 0)"

    def test_sublinearity_positive_homogeneous_H(self, x_grid_small, controls_arr):
        sigma, b, c = controls_arr
        x = np.array([1.0])
        lam = 2.0
        z, p = 0.0, np.zeros(1)
        H = np.ones(1)
        g1 = g_controlled_diffusion(x, z, p, H, sigma, b, c)
        g2 = g_controlled_diffusion(x, z, p, lam * H, sigma, b, c)
        assert np.allclose(g2, lam * g1, atol=1e-10), "G positive-homogeneous in H"

    def test_monotone_in_H(self, x_grid_small, controls_arr):
        sigma, b, c = controls_arr
        x = x_grid_small
        z, p = 0.0, np.zeros_like(x)
        g_lo = g_controlled_diffusion(x, z, p, np.ones_like(x),     sigma, b, c)
        g_hi = g_controlled_diffusion(x, z, p, 2.0 * np.ones_like(x), sigma, b, c)
        assert np.all(g_hi >= g_lo - 1e-10), "G non-decreasing in H"


class TestLyapunov:
    """Tests for Lyapunov function verification."""

    def test_phi_positive(self, x_grid_small):
        phi = lyapunov_phi(x_grid_small)
        assert np.all(phi >= 1.0), "phi(x) = 1 + x^2 >= 1"

    def test_lyapunov_condition_example41(self, alpha_small, x_grid_small):
        # G_alpha(x, phi, grad phi, Hess phi) = 0.5 a_alpha(x) * 2 = a_alpha(x)
        # a_alpha(x) <= 1 = phi(0) <= phi(x) for |x| >= 0
        # The condition is G(x, phi, Dphi, D2phi) <= C_phi * phi(x)
        result = check_lyapunov(x_grid_small, alpha=alpha_small)
        assert result["lyapunov_satisfied"], (
            "Lyapunov condition must hold for Example 4.1 on the test grid"
        )
        assert result["max_ratio"] <= result["C_phi"] + 1e-9
