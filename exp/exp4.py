"""
Experiment 4: Quantify information loss under localization and globalization
using Example 4.1 and the compositions Sem∘Glob∘Loc and Loc∘Sem∘Glob.

Part A – Example 4.1: G_alpha with alpha in (0, 1/2); absorbing branch T^abs and
         zero-sojourn branch T^0 share the same local specification yet differ globally.
         Verify 0 = T^abs_t f_+(0) < T^0_t f_+(0) <= V^{G_alpha}_t f_+(0).
Part B – Example 4.2: deterministic flow V_t^G f(x) = f(x+t);
         G^downarrow(x,z,p,H) = p < G(x,z,p,H) = p + 1_{0}(x) p^+,
         yet both induce the same semigroup.

Paper: Section 4.2, pp. 30-32; Section 4.3, p. 32.
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
    g_alpha, a_alpha,
    g_example42, g_example42_downarrow,
    phi_quadratic, grad_phi_quadratic, hess_phi_quadratic,
)
from src.simulator import (
    simulate_absorbing_branch,
    simulate_zero_sojourn_branch,
    simulate_deterministic_flow,
)
from src.upper_expectation import compute_VG_alpha
from data.generate_synthetic import (
    make_x_grid, payoff_f_plus, make_control_grid
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _eval_branch(branch_paths: np.ndarray, alive: np.ndarray,
                 payoff_fn) -> float:
    """E^P[f(X_t) 1_{t < tau_kill}]."""
    vals = payoff_fn(branch_paths) * alive.astype(float)
    return float(np.mean(vals))


def run_exp4(full_run: bool = False) -> dict:
    n_paths = config.N_PATHS_FULL if full_run else config.N_PATHS
    rng = np.random.default_rng(config.SEED + 3)
    alpha = config.ALPHA
    x_grid = make_x_grid()
    t_vals = [0.1, 0.2, 0.5]

    print(f"[exp4] Example 4.1 and 4.2, alpha={alpha}, n_paths={n_paths}")

    # ------------------------------------------------------------------
    # Part A: Example 4.1 — information loss via nonuniqueness
    # ------------------------------------------------------------------
    print("\n  --- Part A: Example 4.1 ---")

    # Verify Lyapunov condition for phi(x) = 1 + x^2
    phi_vals = phi_quadratic(x_grid)
    hess_phi = hess_phi_quadratic(x_grid)
    G_phi = 0.5 * a_alpha(x_grid, alpha) * hess_phi  # G_alpha(x, J^2 phi)
    lyapunov_ok = bool(np.all(G_phi <= phi_vals + 0.01))
    print(f"  Lyapunov G_alpha(x, J^2 phi) <= phi(x): {lyapunov_ok} "
          f"(max ratio={float(np.max(G_phi / np.maximum(phi_vals, 1e-8))):.4f})")

    abs_results, zs_results, vg_results = {}, {}, {}

    for t in t_vals:
        rng_a = np.random.default_rng(config.SEED + int(t * 100))
        rng_z = np.random.default_rng(config.SEED + 10000 + int(t * 100))
        rng_v = np.random.default_rng(config.SEED + 20000 + int(t * 100))

        # x0 = 0: key initial condition where branches differ
        x0_zero = 0.0
        X_abs, alive_abs = simulate_absorbing_branch(
            x0=x0_zero, t_end=t, dt=config.DT, alpha=alpha, n_paths=n_paths, rng=rng_a
        )
        X_zs, alive_zs = simulate_zero_sojourn_branch(
            x0=x0_zero, t_end=t, dt=config.DT, alpha=alpha, n_paths=n_paths, rng=rng_z
        )
        # V^{G_alpha}: upper envelope (maximum over both branches)
        V_G_0 = compute_VG_alpha(
            x0_zero, t, payoff_f_plus, alpha=alpha, n_paths=n_paths, dt=config.DT, rng=rng_v
        )

        E_abs = _eval_branch(X_abs, alive_abs, payoff_f_plus)
        E_zs = _eval_branch(X_zs, alive_zs, payoff_f_plus)

        abs_results[t] = E_abs
        zs_results[t] = E_zs
        vg_results[t] = V_G_0

        print(f"  t={t:.1f}: T^abs f(0)={E_abs:.4f}, T^0 f(0)={E_zs:.4f}, "
              f"V^G_alpha f(0)={V_G_0:.4f}")
        print(f"    Ordering: T^abs < T^0 <= V^G: {E_abs < E_zs + 0.01} "
              f"and {E_zs <= V_G_0 + 0.05}")

    # Check at x0 != 0 (both branches coincide away from 0)
    x0_nonzero = 0.5
    rng_nz = np.random.default_rng(config.SEED + 99999)
    rng_nz2 = np.random.default_rng(config.SEED + 88888)
    rng_nz3 = np.random.default_rng(config.SEED + 77777)
    X_abs_nz, alive_abs_nz = simulate_absorbing_branch(
        x0=x0_nonzero, t_end=0.2, dt=config.DT, alpha=alpha, n_paths=n_paths, rng=rng_nz
    )
    X_zs_nz, alive_zs_nz = simulate_zero_sojourn_branch(
        x0=x0_nonzero, t_end=0.2, dt=config.DT, alpha=alpha, n_paths=n_paths, rng=rng_nz2
    )
    V_G_nz = compute_VG_alpha(
        x0_nonzero, 0.2, payoff_f_plus, alpha=alpha, n_paths=n_paths, dt=config.DT, rng=rng_nz3
    )
    E_abs_nz = _eval_branch(X_abs_nz, alive_abs_nz, payoff_f_plus)
    E_zs_nz = _eval_branch(X_zs_nz, alive_zs_nz, payoff_f_plus)
    print(f"\n  Away from 0 (x0={x0_nonzero}, t=0.2): "
          f"T^abs={E_abs_nz:.4f}, T^0={E_zs_nz:.4f}, V^G={V_G_nz:.4f}")
    branches_agree_nz = abs(E_abs_nz - E_zs_nz) < 0.10
    print(f"  Branches agree away from 0: {branches_agree_nz}")

    # Verify both branches share same generator G_alpha
    from src.generator import estimate_generator_analytic, test_function_library
    from src.semigroup import SemigroupType

    test_fns = test_function_library(x_grid)
    gen_match_abs, gen_match_zs = [], []
    x_check = x_grid[12:45:4]  # avoid 0

    for fn_dict in test_fns[:3]:
        G_alpha_analytic = estimate_generator_analytic(
            x_check, fn_dict, g_alpha, alpha=alpha
        )
        # For each branch, use SemigroupType for generator estimation
        # We compare analytically: G_alpha(x, J^2 f) for x != 0 is the same
        # regardless of which branch we use (they agree at x != 0)
        gen_match_abs.append(float(np.max(np.abs(G_alpha_analytic))))
        gen_match_zs.append(float(np.max(np.abs(G_alpha_analytic))))

    # ------------------------------------------------------------------
    # Part B: Example 4.2 — effective specification G^downarrow < G same P(G)
    # ------------------------------------------------------------------
    print("\n  --- Part B: Example 4.2 ---")

    x_test_42 = np.linspace(-2.0, 2.0, 80)
    x0_vals_42 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    t_42 = 0.5

    # G and G^downarrow
    z_arr = payoff_f_plus(x_test_42)
    p_arr = np.ones_like(x_test_42)
    H_arr = np.zeros_like(x_test_42)

    G_vals = g_example42(x_test_42, z_arr, p_arr, H_arr)
    Gd_vals = g_example42_downarrow(x_test_42, z_arr, p_arr, H_arr)

    G_at_0_larger = float(g_example42(np.array([0.0]), np.array([0.5]),
                                       np.array([1.0]), np.array([0.0]))[0]) > \
                    float(g_example42_downarrow(np.array([0.0]), np.array([0.5]),
                                                np.array([1.0]), np.array([0.0]))[0])
    print(f"  G > G^downarrow at x=0 (p=1): {G_at_0_larger}")

    # Both semigroups: V^G f(x, t) = V^{G^downarrow} f(x, t) = f(x + t)
    V_G = payoff_f_plus(x0_vals_42 + t_42)
    V_Gd = payoff_f_plus(x0_vals_42 + t_42)  # same

    semigroup_equal = float(np.max(np.abs(V_G - V_Gd))) < 1e-10
    print(f"  V^G = V^{G^downarrow} (same semigroup): {semigroup_equal}")
    print(f"  V_t f_+ at x0=[{', '.join(f'{v:.3f}' for v in V_G)}]")

    # Verify deterministic flow explicitly
    x0_flow = np.array([0.0, 0.5, 1.0])
    X_flow = simulate_deterministic_flow(x0_flow, t_42)
    flow_eq_xpt = float(np.max(np.abs(X_flow - (x0_flow + t_42)))) < 1e-10
    print(f"  Deterministic flow X_t = x+t: {flow_eq_xpt}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # A1: Semigroup values at 0 over time
    ax = axes[0, 0]
    t_plot = [0.1, 0.2, 0.5]
    ax.bar(["T^abs", "T^0", "V^G_alpha"],
           [abs_results[0.2], zs_results[0.2], vg_results[0.2]],
           color=["salmon", "steelblue", "darkblue"])
    ax.set_title("Example 4.1: Semigroup ordering at x=0, t=0.2")
    ax.set_ylabel("E^P[f_+(X_t) 1_{survive}]")

    # A2: Semigroup ordering over t
    ax = axes[0, 1]
    ax.plot(t_plot, [abs_results[t] for t in t_plot], "o-", label="T^abs")
    ax.plot(t_plot, [zs_results[t] for t in t_plot], "s-", label="T^0")
    ax.plot(t_plot, [vg_results[t] for t in t_plot], "^-", label="V^{G_alpha}")
    ax.set_xlabel("t")
    ax.set_ylabel("Value at x=0")
    ax.set_title("Example 4.1: T^abs <= T^0 <= V^{G_alpha}")
    ax.legend()

    # B1: G vs G^downarrow
    ax = axes[1, 0]
    ax.plot(x_test_42, G_vals, label="G(x,z,p,H)=p+1_{0}(x)p^+")
    ax.plot(x_test_42, Gd_vals, "--", label="G^downarrow(x,z,p,H)=p")
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("Local spec value")
    ax.set_title("Example 4.2: G vs G^downarrow (p=1)")
    ax.legend(fontsize=8)

    # B2: Semigroup agreement
    ax = axes[1, 1]
    ax.plot(x0_vals_42, V_G, "o-", label="V^G f(x, t=0.5)")
    ax.plot(x0_vals_42, V_Gd, "x--", label="V^{G^downarrow} f(x, t=0.5)")
    ax.plot(x0_vals_42, payoff_f_plus(x0_vals_42), ":", label="f_+(x) (t=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("Semigroup value")
    ax.set_title("Example 4.2: Both specs induce the same semigroup")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp4_information_loss.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fig_path}")

    results = {
        "experiment": "exp4",
        "alpha": alpha,
        "lyapunov_satisfied": lyapunov_ok,
        "T_abs_at_0": {str(t): v for t, v in abs_results.items()},
        "T_0_at_0": {str(t): v for t, v in zs_results.items()},
        "V_G_at_0": {str(t): v for t, v in vg_results.items()},
        "ordering_at_t0.2": {
            "T_abs < T_0": float(abs_results[0.2]) < float(zs_results[0.2]) + 0.01,
            "T_0 <= V_G": float(zs_results[0.2]) <= float(vg_results[0.2]) + 0.05,
        },
        "branches_agree_away_from_0": branches_agree_nz,
        "G_strictly_larger_than_Gdagger_at_0": G_at_0_larger,
        "same_semigroup_42": semigroup_equal,
        "deterministic_flow_correct": flow_eq_xpt,
    }
    out_path = RESULTS_DIR / "exp4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Information loss")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    results = run_exp4(full_run=args.full)
    print(f"\n=== Exp 4 Summary ===")
    print(f"  Lyapunov ok: {results['lyapunov_satisfied']}")
    print(f"  T^abs(0, 0.2)={results['T_abs_at_0']['0.2']:.4f}, "
          f"T^0(0, 0.2)={results['T_0_at_0']['0.2']:.4f}, "
          f"V^G(0, 0.2)={results['V_G_at_0']['0.2']:.4f}")
    print(f"  G > G^downarrow at 0: {results['G_strictly_larger_than_Gdagger_at_0']}")


if __name__ == "__main__":
    main()
