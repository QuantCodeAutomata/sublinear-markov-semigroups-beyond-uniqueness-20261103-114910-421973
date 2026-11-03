"""
Local specifications G(x, z, p, H) satisfying conditions (G1)-(G3) and the
Lyapunov condition (Assumption 2.2) from the paper.

A local specification is a function G : D x J -> R where J = R x R^d x S(d)
and the jet J_x^2 f = (f(x), grad f(x), Hess f(x)).  The structural conditions
(G1)-(G3) require:
  G1 – sublinearity in (z, p, H): positive homogeneity and subadditivity;
  G2 – monotonicity in H: H1 >= H2 => G(x, z, p, H1) >= G(x, z, p, H2);
  G3 – antitonicity in z: c <= 0 in the support correspondence A_G.

Assumption 2.2 (Lyapunov): there exists phi in C^2(D) with phi >= 1 such that
  G(x, phi(x), grad phi(x), Hess phi(x)) <= C_phi * phi(x) for all x in D.

Paper sections: Definition 2.3, Lemma 2.2, Assumption 2.2 (pp. 9-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A jet is a triple (z, p, H) representing (f(x), grad f(x), Hess f(x)).
# For d=1: z scalar, p scalar, H scalar.
# For d>1: z scalar, p array(d,), H array(d,d).

LocalSpecFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# Signature: G(x, z, p, H) -> scalar or array matching x shape.


# ---------------------------------------------------------------------------
# Concrete local specifications
# ---------------------------------------------------------------------------

def g_linear_diffusion(
    x: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
    a_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Linear diffusion specification G(x, z, p, H) = 0.5 * a(x) * H.

    Used in Example 4.1 (paper p. 30) with a(x) = sigma_alpha(x)^2.
    No killing (c=0), no drift (b=0).  Sublinear (actually linear) in H only.

    Parameters
    ----------
    x, z, p, H : arrays of matching shape (broadcast-compatible)
    a_fn       : callable returning diffusion coefficient a(x) >= 0

    Returns
    -------
    G values, same shape as inputs (broadcast).
    """
    return 0.5 * a_fn(x) * H


def sigma_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Diffusion coefficient for Example 4.1.

    sigma_alpha(x) = |x|^alpha / (1 + |x|^alpha)

    Paper: Section 4.2, p. 30.
    """
    ax = np.abs(x)
    ax_a = ax ** alpha
    return ax_a / (1.0 + ax_a)


def a_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Variance coefficient a_alpha = sigma_alpha^2 for Example 4.1.

    Paper: Section 4.2, p. 30.
    """
    s = sigma_alpha(x, alpha)
    return s * s


def g_example41(
    x: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Local specification for Example 4.1: G_alpha(x, z, p, H) = 0.5*a_alpha(x)*H.

    Paper: Section 4.2, p. 30, Eq. defining G_alpha.
    """
    return 0.5 * a_alpha(x, alpha) * H


def g_example42(
    x: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Local specification for Example 4.2: G(x, z, p, H) = p + 1_{x=0} * p^+.

    Paper: Section 4.3, p. 31.
    G^downarrow = p (the effective specification).
    """
    indicator = (np.abs(x) < 1e-12).astype(float)
    p_plus = np.maximum(p, 0.0)
    return p + indicator * p_plus


def g_effective_42(
    x: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Effective specification G^downarrow for Example 4.2: G^downarrow(x,z,p,H) = p.

    Paper: Section 4.3, p. 31 — G^downarrow is strictly smaller than G at x=0, p>0.
    """
    return p + np.zeros_like(x)


# Aliases used by exp scripts
g_alpha = g_example41
g_example42_downarrow = g_effective_42


def g_controlled_diffusion(
    x: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    H: np.ndarray,
    sigma_vals: np.ndarray,
    b_vals: np.ndarray,
    c_vals: np.ndarray,
) -> np.ndarray:
    """Controlled-diffusion local specification.

    G(x, z, p, H) = sup_{theta in Theta} [0.5*a_theta*H + b_theta*p + c_theta*z]
    where a_theta = sigma_theta^2 >= 0, c_theta <= 0 (killing), b_theta is drift.

    Paper: Section 1.3 semigroup map definition; HJB representation, Lemma 2.2 Eq. (2.4).

    Parameters
    ----------
    x, z, p, H : arrays of matching shape
    sigma_vals  : (N_theta,) diffusion coefficient values
    b_vals      : (N_theta,) drift values
    c_vals      : (N_theta,) <= 0 killing-rate values
    """
    # Broadcast x to (N_theta, *x.shape) for vectorised supremum
    N = len(sigma_vals)
    a_vals = sigma_vals ** 2

    # Ensure jet components match x shape (handles scalar z/p/H inputs)
    x_arr = np.asarray(x)
    x_shape = x_arr.shape
    z_arr = np.broadcast_to(np.asarray(z), x_shape)
    p_arr = np.broadcast_to(np.asarray(p), x_shape)
    H_arr = np.broadcast_to(np.asarray(H), x_shape)

    # Shape: (N_theta, *x_shape)
    zb = np.broadcast_to(z_arr, (N,) + x_shape)
    pb = np.broadcast_to(p_arr, (N,) + x_shape)
    Hb = np.broadcast_to(H_arr, (N,) + x_shape)

    contrib = (
        0.5 * a_vals[:, None] * Hb
        + b_vals[:, None] * pb
        + c_vals[:, None] * zb
    )
    return np.max(contrib, axis=0)


# ---------------------------------------------------------------------------
# Lyapunov verification
# ---------------------------------------------------------------------------

def check_lyapunov(
    G_or_xgrid,
    phi_fn=None,
    grad_phi_fn=None,
    hess_phi_fn=None,
    x_grid=None,
    *,
    alpha: float | None = None,
    c_phi: float = 1.0,
    **ignored_kwargs,
) -> dict:
    """Check Lyapunov condition G(x, phi, grad_phi, Hess_phi) <= c_phi * phi(x).

    Accepts two call conventions:
      Generic: check_lyapunov(G, phi_fn, grad_phi_fn, hess_phi_fn, x_grid, c_phi=...)
      Example 4.1: check_lyapunov(x_grid, alpha=alpha, c_phi=...)

    Paper: Assumption 2.2 (Lyapunov condition, p. 8).

    Returns dict with 'satisfied', 'max_violation', 'lyapunov_satisfied', 'max_ratio', 'C_phi'.
    """
    if callable(G_or_xgrid):
        # Generic signature: (G, phi_fn, grad_phi_fn, hess_phi_fn, x_grid, ...)
        G = G_or_xgrid
        xg = x_grid
        phi_vals = phi_fn(xg)
        grad_vals = grad_phi_fn(xg)
        hess_vals = hess_phi_fn(xg)
        G_phi = G(xg, phi_vals, grad_vals, hess_vals)
        viol = G_phi - c_phi * phi_vals
        max_violation = float(np.max(viol))
        satisfied = max_violation <= 1e-9
        ratio = G_phi / np.maximum(phi_vals, 1e-12)
        max_ratio = float(np.max(ratio))
    else:
        # Example 4.1 signature: (x_grid, alpha=alpha, c_phi=...)
        xg = G_or_xgrid
        if alpha is None:
            raise ValueError("alpha must be provided when x_grid is first arg")
        phi_vals = 1.0 + xg ** 2
        G_phi = a_alpha(xg, alpha)  # G_alpha(x, phi, 2x, 2) = a_alpha(x)
        viol = G_phi - c_phi * phi_vals
        max_violation = float(np.max(viol))
        satisfied = max_violation <= 1e-9
        ratio = G_phi / np.maximum(phi_vals, 1e-12)
        max_ratio = float(np.max(ratio))
    return {
        "satisfied": satisfied,
        "max_violation": max_violation,
        "lyapunov_satisfied": satisfied,
        "max_ratio": max_ratio,
        "C_phi": c_phi,
    }


# ---------------------------------------------------------------------------
# Standard Lyapunov function phi(x) = 1 + x^2 (1D)
# ---------------------------------------------------------------------------

def phi_quadratic(x: np.ndarray) -> np.ndarray:
    """phi(x) = 1 + x^2; used for Example 4.1 and 4.2 Lyapunov checks."""
    return 1.0 + x ** 2


# Alias used by tests and experiments
lyapunov_phi = phi_quadratic


def grad_phi_quadratic(x: np.ndarray) -> np.ndarray:
    """grad phi(x) = 2x."""
    return 2.0 * x


def hess_phi_quadratic(x: np.ndarray) -> np.ndarray:
    """Hess phi(x) = 2 (scalar, 1D)."""
    return 2.0 * np.ones_like(x)


# ---------------------------------------------------------------------------
# Sublinearity and structural checks
# ---------------------------------------------------------------------------

def check_sublinearity(
    g_fn: LocalSpecFn,
    x_test: np.ndarray,
    jets_1: tuple,
    jets_2: tuple,
    lamda: float = 2.0,
    **g_kwargs,
) -> dict:
    """Numerically check positive homogeneity and subadditivity of G in (z, p, H).

    For sublinearity: G(x, lambda*U) = lambda*G(x,U) and
                      G(x, U1+U2) <= G(x,U1) + G(x,U2).

    Paper: condition (G1), p. 9.
    """
    z1, p1, H1 = jets_1
    z2, p2, H2 = jets_2

    # Homogeneity: G(x, lambda*U1) = lambda * G(x, U1)
    G_U1 = g_fn(x_test, z1, p1, H1, **g_kwargs)
    G_lU1 = g_fn(x_test, lamda * z1, lamda * p1, lamda * H1, **g_kwargs)
    hom_err = np.max(np.abs(G_lU1 - lamda * G_U1))

    # Subadditivity: G(x, U1+U2) <= G(x,U1) + G(x,U2)
    G_U2 = g_fn(x_test, z2, p2, H2, **g_kwargs)
    G_sum = g_fn(x_test, z1 + z2, p1 + p2, H1 + H2, **g_kwargs)
    sub_viol = np.max(G_sum - (G_U1 + G_U2))

    return {
        "homogeneity_error": float(hom_err),
        "subadditivity_violation": float(sub_viol),
        "sublinear": float(hom_err) < 1e-10 and float(sub_viol) < 1e-10,
    }


def check_monotonicity_H(
    g_fn: LocalSpecFn,
    x_test: np.ndarray,
    z_test: np.ndarray,
    p_test: np.ndarray,
    H_base: np.ndarray,
    dH: float = 1.0,
    **g_kwargs,
) -> dict:
    """Check monotonicity in H: H1 >= H2 => G(x,z,p,H1) >= G(x,z,p,H2).

    Paper: condition (G2), p. 9.
    """
    G_base = g_fn(x_test, z_test, p_test, H_base, **g_kwargs)
    G_plus = g_fn(x_test, z_test, p_test, H_base + dH, **g_kwargs)
    violation = np.max(G_base - G_plus)  # should be <= 0
    return {
        "max_violation": float(violation),
        "monotone_in_H": float(violation) <= 1e-10,
    }


def check_antitonicity_z(
    g_fn: LocalSpecFn,
    x_test: np.ndarray,
    z_test: np.ndarray,
    p_test: np.ndarray,
    H_test: np.ndarray,
    dz: float = 1.0,
    **g_kwargs,
) -> dict:
    """Check antitonicity in z: G(x, z+dz, p, H) <= G(x, z, p, H) for dz > 0.

    Paper: condition (G3), p. 9 (c_theta <= 0 in support A_G).
    """
    G_base = g_fn(x_test, z_test, p_test, H_test, **g_kwargs)
    G_plus = g_fn(x_test, z_test + dz, p_test, H_test, **g_kwargs)
    violation = np.max(G_plus - G_base)  # should be <= 0
    return {
        "max_violation": float(violation),
        "antitone_in_z": float(violation) <= 1e-10,
    }


# ---------------------------------------------------------------------------
# Convenience aliases matching the paper's notation
# ---------------------------------------------------------------------------

# g_alpha is the local specification from Example 4.1 (Section 4.2)
g_alpha = g_example41

# g_example42_downarrow is the effective specification G^downarrow from Section 4.3
g_example42_downarrow = g_effective_42
