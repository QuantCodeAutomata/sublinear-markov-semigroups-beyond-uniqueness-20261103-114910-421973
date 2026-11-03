"""
exp7: Uniqueness transfer and the C_b-Feller property of V^G.

Tests the paper's claim (Section 5.3) that, for comparison-friendly
specifications G, the canonical semigroup V^G is the unique dominated
semigroup satisfying lower infinitesimal consistency, and that V^G maps
continuous bounded functions to continuous bounded functions (C_b-Feller)
with V_h^G f -> f locally uniformly as h -> 0.

Paper: Section 5.3, "Viscosity uniqueness and the Feller property."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from src import config
from src.local_spec import g_controlled_diffusion
from src.hjb_solver import solve_hjb_controlled
from src.upper_expectation import compute_VG
from src.generator import estimate_generator
from src.semigroup import SemigroupType, eval_semigroup
from data.generate_synthetic import (
    make_x_grid, make_control_grid,
    payoff_f_plus, payoff_gaussian, payoff_tanh,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(full_run: bool = False) -> dict:
    """
    1. Build V^G for a controlled diffusion with continuous coefficients.
    2. C_b-Feller test: verify x -> V_t^G f(x) is continuous for payoffs f in C_b.
    3. Local-uniform convergence: sup_{x in K} |V_h^G f(x) - f(x)| -> 0 as h -> 0.
    4. PDE uniqueness proxy: compare V^G (stochastic) vs HJB solution on continuous f.
    5. Sub-semigroup comparison: dominated T vs V^G.
    """
    n_paths = config.N_PATHS_FULL if full_run else config.N_PATHS
    rng = np.random.default_rng(config.SEED + 700)

    controls = make_control_grid(full_run=full_run)
    x_grid = make_x_grid()
    sigma_vals = np.array([c["sigma"] for c in controls])
    b_vals     = np.array([c["b"]     for c in controls])
    c_vals     = np.array([c["c"]     for c in controls])

    # ------------------------------------------------------------------
    # Part 1: C_b-Feller — verify V_t^G f is continuous for t in {0.25, 0.5}
    # ------------------------------------------------------------------
    print("[exp7] Part 1: C_b-Feller continuity of x -> V_t^G f(x)")

    test_payoffs = {
        "gaussian": payoff_gaussian,
        "tanh":     payoff_tanh,
        "f_plus":   payoff_f_plus,
    }
    t_test = 0.25
    feller_results = {}

    for name, payoff_fn in test_payoffs.items():
        f_vals = payoff_fn(x_grid)
        payoff_interp = lambda x, _f=f_vals, _xg=x_grid: np.interp(x, _xg, _f)

        V = compute_VG(
            x0_vals=x_grid,
            t=t_test,
            payoff_fn=payoff_interp,
            controls=controls,
            n_paths=n_paths,
            dt=config.DT,
            rng=np.random.default_rng(config.SEED + 700),
        )

        # Continuity: max jump in V over neighbouring grid points
        diffs = np.abs(np.diff(V))
        max_jump = float(np.max(diffs))
        # Compare to jumps in original payoff scaled by t (rough bound)
        input_max_jump = float(np.max(np.abs(np.diff(f_vals))))
        is_continuous = max_jump < max(2.0 * input_max_jump, 0.15)
        feller_results[name] = {
            "max_jump_V": max_jump,
            "max_jump_f": input_max_jump,
            "continuous": is_continuous,
        }
        print(f"  [{name}] max Δ_V={max_jump:.4f}, max Δ_f={input_max_jump:.4f}, "
              f"continuous={is_continuous}")

    all_continuous = all(v["continuous"] for v in feller_results.values())
    print(f"  C_b-Feller satisfied on all payoffs: {all_continuous}")

    # ------------------------------------------------------------------
    # Part 2: Local uniform convergence V_h^G f -> f as h -> 0
    # ------------------------------------------------------------------
    print("\n[exp7] Part 2: Local-uniform convergence V_h^G f -> f as h -> 0")

    h_sequence = [0.25, 0.1, 0.05, 0.02, 0.01]
    compact_K = np.linspace(-1.5, 1.5, 61)  # compact set K
    f_gauss = payoff_gaussian
    payoff_gauss_interp = lambda x: np.interp(x, x_grid, f_gauss(x_grid))
    f_K = f_gauss(compact_K)

    luconv_errors = []
    for h in h_sequence:
        V_h = compute_VG(
            x0_vals=compact_K,
            t=h,
            payoff_fn=payoff_gauss_interp,
            controls=controls,
            n_paths=max(n_paths // 2, 200),
            dt=min(config.DT, h / 5),
            rng=np.random.default_rng(config.SEED + 710),
        )
        sup_err = float(np.max(np.abs(V_h - f_K)))
        luconv_errors.append(sup_err)
        print(f"  h={h:.3f}: sup_K|V_h^G f - f| = {sup_err:.5f}")

    # Check convergence (errors should decrease toward 0)
    luconv_decreasing = luconv_errors[-1] < luconv_errors[0]
    print(f"  Errors decreasing as h -> 0: {luconv_decreasing}")
    print(f"  Final error at h={h_sequence[-1]}: {luconv_errors[-1]:.5f}")

    # ------------------------------------------------------------------
    # Part 3: PDE uniqueness proxy — stochastic V^G matches HJB solution
    # ------------------------------------------------------------------
    print("\n[exp7] Part 3: PDE/stochastic agreement (uniqueness proxy)")

    t_pde = config.T_MAX / 2.0
    f_fn_pde = lambda x: payoff_gaussian(np.asarray(x))
    f_init_pde = f_fn_pde(x_grid)

    pde_sol = solve_hjb_controlled(
        f_fn=f_fn_pde,
        x_grid=x_grid,
        t_max=t_pde,
        n_steps=config.N_PDE_STEPS,
        sigma_vals=sigma_vals,
        b_vals=b_vals,
        c_vals=c_vals,
    )
    u_pde = pde_sol["u_final"]

    payoff_pde_interp = lambda x: np.interp(x, x_grid, f_init_pde)
    V_stoch = compute_VG(
        x0_vals=x_grid,
        t=t_pde,
        payoff_fn=payoff_pde_interp,
        controls=controls,
        n_paths=n_paths,
        dt=config.DT,
        rng=np.random.default_rng(config.SEED + 720),
    )

    pde_stoch_err = np.abs(u_pde - V_stoch)
    max_pde_err = float(np.max(pde_stoch_err))
    mean_pde_err = float(np.mean(pde_stoch_err))
    agreement = max_pde_err < 0.15
    print(f"  Max |u_PDE - V_MC| = {max_pde_err:.4f}")
    print(f"  Mean |u_PDE - V_MC| = {mean_pde_err:.4f}")
    print(f"  PDE and stochastic agree (< 0.15): {agreement}")

    # ------------------------------------------------------------------
    # Part 4: Dominated sub-semigroup stays strictly below V^G
    # ------------------------------------------------------------------
    print("\n[exp7] Part 4: Dominated sub-semigroup T <= V^G")

    # Use a subfamily with reduced sigma range (strict subset of control set)
    n_sub = max(2, len(controls) // 3)
    sub_controls = controls[:n_sub]  # strict subset

    f_dom = payoff_f_plus(x_grid)
    payoff_dom_interp = lambda x: np.interp(x, x_grid, f_dom)

    V_full = compute_VG(
        x0_vals=x_grid, t=t_test,
        payoff_fn=payoff_dom_interp, controls=controls,
        n_paths=n_paths, dt=config.DT,
        rng=np.random.default_rng(config.SEED + 730),
    )
    V_sub = compute_VG(
        x0_vals=x_grid, t=t_test,
        payoff_fn=payoff_dom_interp, controls=sub_controls,
        n_paths=n_paths, dt=config.DT,
        rng=np.random.default_rng(config.SEED + 731),
    )

    dominated = bool(np.all(V_sub <= V_full + 0.02))  # small MC tolerance
    mean_gap = float(np.mean(V_full - V_sub))
    strict_below = mean_gap > 0.001
    print(f"  T_sub <= V^G (all points): {dominated}")
    print(f"  Mean gap V^G - T_sub = {mean_gap:.4f} (expect > 0)")
    print(f"  T_sub strictly below V^G: {strict_below}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: C_b-Feller — V_t f vs f
    ax = axes[0]
    f_plot = payoff_gaussian(x_grid)
    V_plot = compute_VG(
        x0_vals=x_grid, t=t_test,
        payoff_fn=lambda x: np.interp(x, x_grid, f_plot),
        controls=controls, n_paths=n_paths, dt=config.DT,
        rng=np.random.default_rng(config.SEED + 740),
    )
    ax.plot(x_grid, f_plot, label="f (payoff)", lw=1.5)
    ax.plot(x_grid, V_plot, label=f"V$_{{t}}^G$ f, t={t_test}", lw=1.5, ls="--")
    ax.set_title("C_b-Feller: V^G maps C_b → C_b")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Local uniform convergence h -> 0
    ax = axes[1]
    ax.plot(h_sequence, luconv_errors, "o-", color="steelblue")
    ax.set_xlabel("h")
    ax.set_ylabel("sup_K |V_h f - f|")
    ax.set_title("Local Uniform Convergence as h → 0")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 3: PDE vs stochastic
    ax = axes[2]
    ax.plot(x_grid, u_pde, label="HJB PDE", lw=1.5)
    ax.plot(x_grid, V_stoch, label="Stochastic V^G", lw=1.5, ls="--")
    ax.set_title(f"PDE vs Stochastic V^G (t={t_pde:.2f})")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "exp7_feller_uniqueness.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[exp7] Plot saved: {plot_path}")

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    results = {
        "experiment": "exp7",
        "feller_continuity": feller_results,
        "all_continuous": all_continuous,
        "luconv_h_sequence": h_sequence,
        "luconv_errors": luconv_errors,
        "luconv_decreasing": luconv_decreasing,
        "luconv_final_error": luconv_errors[-1],
        "pde_max_err": max_pde_err,
        "pde_mean_err": mean_pde_err,
        "pde_stochastic_agree": agreement,
        "dominated_T_sub_leq_VG": dominated,
        "dominated_T_sub_strictly_below": strict_below,
        "mean_gap_VG_minus_T_sub": mean_gap,
    }

    out_path = RESULTS_DIR / "exp7_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[exp7] Results saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp7: Feller property and uniqueness")
    parser.add_argument("--full", action="store_true",
                        help="Use full run configuration (more paths/controls)")
    args = parser.parse_args()
    run(full_run=args.full)
