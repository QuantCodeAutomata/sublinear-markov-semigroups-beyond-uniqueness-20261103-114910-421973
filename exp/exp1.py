"""
Experiment 1: Construct and verify the canonical semigroup V^G from a
prescribed local specification G.

Implements V_t^G f(x) = sup_{P in P_x(G)} E^P[f(X_t) 1_{t < tau_kill}]
for a controlled-diffusion specification and verifies:
  (1) Semigroup / time-consistency property.
  (2) Sublinearity and monotonicity.
  (3) Domination over subfamilies.
  (4) Small-time infinitesimal consistency with G.

Paper: Section 1.3 (semigroup map), Section 4.1, Section 5.2.
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
    g_controlled_diffusion,
    check_lyapunov,
    check_sublinearity,
    check_monotonicity_H,
    check_antitonicity_z,
    phi_quadratic,
    grad_phi_quadratic,
    hess_phi_quadratic,
)
from src.upper_expectation import (
    compute_VG,
    check_semigroup_property,
    check_semigroup_sublinearity,
    make_default_control_grid,
)
from src.generator import (
    test_function_library,
    estimate_generator,
    estimate_generator_analytic,
)
from src.semigroup import SemigroupType, eval_semigroup
from data.generate_synthetic import (
    make_x_grid,
    make_t_eval,
    make_control_grid,
    PAYOFF_LIBRARY,
    payoff_f_plus,
    payoff_gaussian,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp1(full_run: bool = False) -> dict:
    """Run experiment 1 and return a results dict."""
    n_paths = config.N_PATHS_FULL if full_run else config.N_PATHS
    n_theta = config.N_THETA_FULL if full_run else config.N_THETA
    rng = np.random.default_rng(config.SEED)

    x_grid = make_x_grid()
    t_eval = make_t_eval()
    controls = make_control_grid(full_run=full_run)

    # Build sigma, b, c arrays for g_controlled_diffusion
    sigma_arr = np.array([c["sigma"] for c in controls])
    b_arr = np.array([c["b"] for c in controls])
    c_arr = np.array([c["c"] for c in controls])

    print(f"[exp1] Controlled-diffusion spec: {len(controls)} controls, "
          f"{n_paths} paths, grid {len(x_grid)} pts.")

    # ------------------------------------------------------------------
    # 1. Structural verification of G
    # ------------------------------------------------------------------
    def G(x, z, p, H):
        return g_controlled_diffusion(x, z, p, H, sigma_arr, b_arr, c_arr)

    x_test = x_grid[10:51:5]
    z_test = np.zeros_like(x_test)
    p_test = np.ones_like(x_test)
    H_test = np.ones_like(x_test)

    # c_phi derived analytically: for phi=1+x^2, sigma_max=1, b_max=0.5,
    # the tight bound requires C >= (1+sqrt(2))/2 ≈ 1.207; use config.CTRL_C_PHI=1.30.
    lyap = check_lyapunov(
        G, phi_quadratic, grad_phi_quadratic, hess_phi_quadratic,
        x_grid, c_phi=config.CTRL_C_PHI,
        sigma_vals=sigma_arr, b_vals=b_arr, c_vals=c_arr,
    )
    sublin = check_sublinearity(
        G, x_test,
        jets_1=(z_test, p_test, H_test),
        jets_2=(0.5 * z_test, -p_test, 0.5 * H_test),
        lamda=2.0,
        sigma_vals=sigma_arr, b_vals=b_arr, c_vals=c_arr,
    )
    mono_H = check_monotonicity_H(
        G, x_test, z_test, p_test, H_test, dH=1.0,
        sigma_vals=sigma_arr, b_vals=b_arr, c_vals=c_arr,
    )
    anti_z = check_antitonicity_z(
        G, x_test, z_test, p_test, H_test, dz=1.0,
        sigma_vals=sigma_arr, b_vals=b_arr, c_vals=c_arr,
    )

    print(f"  Lyapunov satisfied: {lyap['satisfied']} (max viol={lyap['max_violation']:.3e})")
    print(f"  Sublinear: {sublin['sublinear']} (hom_err={sublin['homogeneity_error']:.3e}, "
          f"sub_viol={sublin['subadditivity_violation']:.3e})")
    print(f"  Monotone H: {mono_H['monotone_in_H']} (viol={mono_H['max_violation']:.3e})")
    print(f"  Antitone z: {anti_z['antitone_in_z']} (viol={anti_z['max_violation']:.3e})")

    # ------------------------------------------------------------------
    # 2. Compute V^G f at several times and states
    # ------------------------------------------------------------------
    payoff_fn = payoff_f_plus
    t_single = config.T0
    eval_kwargs = dict(
        n_paths=n_paths, dt=config.DT, controls=controls, rng=rng
    )

    V_vals = {}
    for t in t_eval:
        rng_t = np.random.default_rng(config.SEED + int(t * 1000))
        V_t = compute_VG(
            x_grid, t, payoff_fn, controls, n_paths, config.DT, rng_t
        )
        V_vals[float(t)] = V_t.tolist()

    print(f"  V^G f at t={t_eval[0]:.2f}: min={min(V_vals[t_eval[0]]):.3f}, "
          f"max={max(V_vals[t_eval[0]]):.3f}")

    # ------------------------------------------------------------------
    # 3. Semigroup (time-consistency) check: V_{s+t} f vs V_s(V_t f)
    # ------------------------------------------------------------------
    s = 0.2
    t = 0.3
    rng_sg = np.random.default_rng(config.SEED + 100)
    x_check = x_grid[10:51:5]
    sg_res = check_semigroup_property(
        x_check, s, t, payoff_fn, controls, n_paths, config.DT, rng_sg
    )
    print(f"  Semigroup check (s={s}, t={t}): max_abs_err={sg_res['max_abs_err']:.4f}, "
          f"mean_abs_err={sg_res['mean_abs_err']:.4f}")

    # ------------------------------------------------------------------
    # 4. Sublinearity of V^G
    # ------------------------------------------------------------------
    rng_sub = np.random.default_rng(config.SEED + 200)
    sub_res = check_semigroup_sublinearity(
        x_check, 0.3, payoff_f_plus, controls,
        n_paths // 2, config.DT, rng_sub
    )
    print(f"  V^G sublinear: {sub_res['is_sublinear']} (hom_err={sub_res['homogeneity_err']:.3f}, "
          f"sub_viol={sub_res['subadditivity_violation']:.3f})")

    # ------------------------------------------------------------------
    # 5. Infinitesimal consistency: Delta_h^{V^G} f -> G(., J^2 f) as h->0
    # ------------------------------------------------------------------
    test_fns = test_function_library(x_grid)
    gen_results = {}
    for fn_dict in test_fns[:3]:  # first 3 test functions
        name = fn_dict["name"]
        f_fn = lambda x, _f=fn_dict["f_arr"], _xg=x_grid: np.interp(x, _xg, _f)
        rng_gen = np.random.default_rng(config.SEED + 300)
        eval_kw = dict(
            n_paths=n_paths, dt=config.DT, controls=controls, rng=rng_gen
        )
        gen_res = estimate_generator(
            SemigroupType.CONTROLLED, x_grid[10:51:5], f_fn,
            config.H_VALUES, eval_kw
        )
        G_analytic = estimate_generator_analytic(
            x_grid[10:51:5], fn_dict, G
        )
        max_diff = float(np.max(np.abs(gen_res["G_T_estimate"] - G_analytic[::1])))
        gen_results[name] = {"max_diff_vs_G": max_diff}
        print(f"  Generator [{name}]: max|Delta_h f - G(.,J^2 f)| = {max_diff:.4f}")

    # ------------------------------------------------------------------
    # 6. Domination by subfamilies
    # ------------------------------------------------------------------
    # Use half the controls as a strict subfamily
    subset_controls = controls[:len(controls) // 2]
    rng_dom = np.random.default_rng(config.SEED + 400)
    V_sub = compute_VG(x_check, 0.3, payoff_fn, subset_controls, n_paths // 2, config.DT, rng_dom)
    rng_full = np.random.default_rng(config.SEED + 500)
    V_full = compute_VG(x_check, 0.3, payoff_fn, controls, n_paths // 2, config.DT, rng_full)
    dom_viol = float(np.max(V_sub - V_full))
    print(f"  Domination (T^C <= V^G): max violation = {dom_viol:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: V^G f at multiple times
    ax = axes[0]
    for t_key, v in V_vals.items():
        ax.plot(x_grid, v, label=f"t={t_key:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("V_t^G f(x)")
    ax.set_title("Canonical semigroup V^G f_+(x)")
    ax.legend(fontsize=7)

    # Plot 2: semigroup consistency
    ax = axes[1]
    ax.plot(x_check, sg_res["V_st"], label=f"V_{{s+t}} f (direct)", marker="o")
    ax.plot(x_check, sg_res["V_s_Vt"], label=f"V_s(V_t f) (composed)", marker="x")
    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    ax.set_title(f"Semigroup consistency (s={s}, t={t})")
    ax.legend()

    # Plot 3: domination
    ax = axes[2]
    ax.plot(x_check, V_sub, label="Subfamily T^C", marker="o")
    ax.plot(x_check, V_full, label="V^G (full)", marker="x")
    ax.set_xlabel("x")
    ax.set_ylabel("T_t f(x)")
    ax.set_title("Domination: T^C <= V^G")
    ax.legend()

    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp1_semigroup_VG.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    results = {
        "experiment": "exp1",
        "lyapunov_satisfied": lyap["satisfied"],
        "lyapunov_max_violation": lyap["max_violation"],
        "sublinear": sublin["sublinear"],
        "homogeneity_error": sublin["homogeneity_error"],
        "subadditivity_violation": sublin["subadditivity_violation"],
        "monotone_H": mono_H["monotone_in_H"],
        "antitone_z": anti_z["antitone_in_z"],
        "semigroup_max_error": sg_res["max_abs_err"],
        "semigroup_rel_error": sg_res["mean_abs_err"],
        "sublinear_semigroup": sub_res["is_sublinear"],
        "domination_violation": dom_viol,
        "generator_consistency": gen_results,
        "V_vals_at_t0": {str(k): float(v[len(v) // 2]) for k, v in V_vals.items()},
    }

    out_path = RESULTS_DIR / "exp1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Canonical semigroup V^G")
    parser.add_argument("--full", action="store_true", help="Use paper full configuration")
    args = parser.parse_args()
    results = run_exp1(full_run=args.full)
    print("\n=== Exp 1 Summary ===")
    print(f"  Lyapunov ok: {results['lyapunov_satisfied']}")
    print(f"  Sublinear G: {results['sublinear']}")
    print(f"  Semigroup max error: {results['semigroup_max_error']:.4f}")
    print(f"  Domination violation: {results['domination_violation']:.4f}")


if __name__ == "__main__":
    main()
