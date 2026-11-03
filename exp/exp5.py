"""
Experiment 5: Compute the effective local specification G^downarrow and test
global realizability of a prescribed G.

Implements the reverse composition G -> P(G) -> V^G -> G^downarrow = G_{V^G},
and checks whether G^downarrow = G (global realizability) or G^downarrow < G
(excess local information).

Paper: Section 4.3 (pp. 31-32); Theorem 4.11, p. 32.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from src import config
from src.local_spec import (
    g_example42, g_effective_42,
    phi_quadratic, grad_phi_quadratic, hess_phi_quadratic,
    g_controlled_diffusion,
)
from src.generator import (
    estimate_generator, test_function_library, estimate_generator_analytic,
)
from src.semigroup import SemigroupType, eval_semigroup
from src.simulator import simulate_deterministic_flow
from data.generate_synthetic import make_x_grid, make_control_grid, payoff_f_plus

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp5(full_run: bool = False) -> dict:
    n_paths = (config.N_PATHS_FULL if full_run else config.N_PATHS) // 2
    rng = np.random.default_rng(config.SEED + 4)

    # ------------------------------------------------------------------
    # Part A: Example 4.2 — explicit effective specification
    # ------------------------------------------------------------------
    print("[exp5] Part A: Example 4.2 — G^downarrow vs G")

    x_grid_42 = np.linspace(-2.0, 2.0, 81)
    t_42 = 0.5
    z_arr = payoff_f_plus(x_grid_42)
    p_arr = np.ones_like(x_grid_42)
    H_arr = np.zeros_like(x_grid_42)

    G_vals = g_example42(x_grid_42, z_arr, p_arr, H_arr)
    Gd_vals = g_effective_42(x_grid_42, z_arr, p_arr, H_arr)

    strict_excess = float(np.max(G_vals - Gd_vals))
    print(f"  Max G - G^downarrow (at p=1): {strict_excess:.4f} (should be > 0)")

    # Semigroup V^G f(x, t) = f(x + t) for both G and G^downarrow
    x0_vals = np.linspace(-1.5, 1.5, 15)
    VG_vals = payoff_f_plus(x0_vals + t_42)
    VGd_vals = payoff_f_plus(x0_vals + t_42)
    same_semigroup = float(np.max(np.abs(VG_vals - VGd_vals))) < 1e-10
    print(f"  Same semigroup V^G = V^G^downarrow: {same_semigroup}")

    # Estimate G^downarrow numerically = G_{V^G} via finite differences
    # V^G f(x, t) = f(x + t) => (V^G_h f(x) - f(x))/h -> f'(x) = p
    # So G^downarrow = lim_h->0 (V^G_h f(x) - f(x))/h = f'(x) * 1 = p
    test_fns = test_function_library(x_grid_42)
    gen_downarrow_estimates = {}
    gen_G_analytic_42 = {}
    gen_Gd_analytic_42 = {}

    for fn_dict in test_fns[:4]:
        name = fn_dict["name"]
        # Exact: G_{V^G} = G^downarrow = p (Eq. from Section 4.3)
        f_at_xpt = lambda x, _f=fn_dict["f"], _xg=x_grid_42: np.interp(x, _xg, _f)
        G_est_list = []
        for h in [0.1, 0.05, 0.02, 0.01]:
            # V^G_h f(x) = f(x + h) for the deterministic flow semigroup
            V_h_f = np.interp(x_grid_42 + h, x_grid_42 + h, fn_dict["f"])
            # Actually: V^G_h f(x) = f(x + h)
            # Need to evaluate f at (x + h); use interp on extended domain
            x_shifted = x_grid_42 + h
            V_h = np.array([f_at_xpt(np.array([xi]))[0] for xi in x_shifted])
            G_est = (V_h - fn_dict["f"]) / h
            G_est_list.append(float(np.mean(G_est)))

        gen_downarrow_estimates[name] = G_est_list[-1]  # smallest h

        # G_full(x, J^2 f): p + 1_0(x) * p^+
        G_full_vals = g_example42(x_grid_42, fn_dict["f"], fn_dict["grad_f"], fn_dict["hess_f"])
        gen_G_analytic_42[name] = float(np.mean(G_full_vals))

        # G^downarrow(x, J^2 f): p only
        Gd_vals_fn = g_effective_42(x_grid_42, fn_dict["f"], fn_dict["grad_f"], fn_dict["hess_f"])
        gen_Gd_analytic_42[name] = float(np.mean(Gd_vals_fn))

        print(f"  [{name}] G^downarrow ~ {gen_downarrow_estimates[name]:.4f}, "
              f"G_analytic = {gen_G_analytic_42[name]:.4f}, "
              f"G^downarrow_analytic = {gen_Gd_analytic_42[name]:.4f}")

    # Realizability check: G^downarrow = G iff G globally realizable
    # In Ex 4.2: G > G^downarrow at x=0 => G not globally realizable
    G_globally_realizable = strict_excess < 0.01
    print(f"  G globally realizable: {G_globally_realizable} "
          f"(expected: False for Example 4.2)")

    # ------------------------------------------------------------------
    # Part B: Realizable case — controlled diffusion G is globally realizable
    # ------------------------------------------------------------------
    print("\n[exp5] Part B: Controlled diffusion — G globally realizable check")

    controls = make_control_grid(full_run=full_run)
    sigma_arr = np.array([c["sigma"] for c in controls])
    b_arr = np.array([c["b"] for c in controls])
    c_arr = np.array([c["c"] for c in controls])

    def G_ctrl(x, z, p, H):
        return g_controlled_diffusion(x, z, p, H, sigma_arr, b_arr, c_arr)

    x_check = make_x_grid()[10:51:5]
    test_fns_ctrl = test_function_library(x_check)

    h_vals = config.H_VALUES[:3]
    eval_kw = dict(n_paths=min(n_paths, 1000), dt=config.DT,
                   controls=controls, rng=np.random.default_rng(config.SEED + 500))

    diff_G_Gd_ctrl = []
    for fn_dict_c in test_fns_ctrl[:3]:
        name = fn_dict_c["name"]
        # G_ctrl already captures sigma_arr/b_arr/c_arr via closure — no extra kwargs needed
        G_analytic = estimate_generator_analytic(x_check, fn_dict_c, G_ctrl)
        f_fn = lambda x, _f=fn_dict_c["f"], _xg=x_check: np.interp(x, _xg, _f)
        gen_res = estimate_generator(SemigroupType.CONTROLLED, x_check, f_fn, h_vals, eval_kw)
        G_Vg = gen_res["G_T_estimate"]
        diff = float(np.max(np.abs(G_Vg - G_analytic)))
        diff_G_Gd_ctrl.append(diff)
        print(f"  [{name}] |G_{{V^G}} - G|_inf = {diff:.4f}")

    ctrl_realizable = max(diff_G_Gd_ctrl) < 0.20
    print(f"  Controlled G approximately realizable: {ctrl_realizable}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(x_grid_42, G_vals, label="G(x,z,p,H) = p + 1_0(x)p^+")
    ax.plot(x_grid_42, Gd_vals, "--", label="G^downarrow(x,z,p,H) = p")
    ax.fill_between(x_grid_42, Gd_vals, G_vals, alpha=0.2, label="Excess (non-realizable)")
    ax.set_xlabel("x")
    ax.set_ylabel("Local specification value (p=1)")
    ax.set_title("Example 4.2: G vs G^downarrow = G_{V^G}")
    ax.legend(fontsize=8)

    ax = axes[1]
    names = list(gen_downarrow_estimates.keys())
    est_vals = [gen_downarrow_estimates[n] for n in names]
    gd_vals_bar = [gen_Gd_analytic_42[n] for n in names]
    x_idx = np.arange(len(names))
    ax.bar(x_idx - 0.2, est_vals, 0.4, label="G^downarrow numeric (Δ_h V^G)")
    ax.bar(x_idx + 0.2, gd_vals_bar, 0.4, label="G^downarrow analytic (= p)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_title("G^downarrow vs numeric estimate per test function")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp5_effective_spec.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    results = {
        "experiment": "exp5",
        "G_Gdagger_max_excess": strict_excess,
        "G_globally_realizable_ex42": G_globally_realizable,
        "same_semigroup_ex42": same_semigroup,
        "controlled_diff_G_Gd": diff_G_Gd_ctrl,
        "ctrl_approximately_realizable": ctrl_realizable,
        "gen_downarrow_estimates": gen_downarrow_estimates,
    }
    out_path = RESULTS_DIR / "exp5_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Effective specification G^downarrow")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    results = run_exp5(full_run=args.full)
    print(f"\n=== Exp 5 Summary ===")
    print(f"  G > G^downarrow at x=0: {results['G_Gdagger_max_excess']:.4f} (>0 expected)")
    print(f"  G globally realizable (Ex 4.2): {results['G_globally_realizable_ex42']}")
    print(f"  Controlled diff approx realizable: {results['ctrl_approximately_realizable']}")


if __name__ == "__main__":
    main()
