"""
Experiment 6: Upper Perron / viscosity characterization.

Compare V^G (stochastic upper-expectation semigroup) with the greatest bounded
upper semicontinuous viscosity subsolution of the HJB Cauchy problem:
    partial_t u - G(x, u, Du, D2u) = 0,  u(0, .) <= f.

Paper: Section 5.2, pp. 35-36; Theorem 5.5 (Upper Perron = V_f^G).
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
from src.local_spec import g_example42, g_controlled_diffusion
from src.hjb_solver import solve_hjb_controlled, solve_transport_pde
from src.upper_expectation import compute_VG, compute_VG_diffusion
from data.generate_synthetic import (
    make_x_grid, make_control_grid, payoff_f_plus, payoff_gaussian,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp6(full_run: bool = False) -> dict:
    n_paths = config.N_PATHS_FULL if full_run else config.N_PATHS
    rng = np.random.default_rng(config.SEED + 5)

    # ------------------------------------------------------------------
    # Part A: Example 4.2 — first-order transport HJB
    # G(x, z, p, H) = p  =>  partial_t u = p = Du  =>  u(t, x) = f(x + t)
    # ------------------------------------------------------------------
    print("[exp6] Part A: Example 4.2 — transport HJB vs deterministic flow")

    x_grid_42 = np.linspace(-3.0, 3.0, 201)
    dx_42 = x_grid_42[1] - x_grid_42[0]
    t_arr_42 = np.array([0.25, 0.5, 1.0])

    # Stochastic side: V^G f(x, t) = f(x + t) (deterministic flow)
    f_fn_42 = lambda x: payoff_gaussian(x)
    f_init_42 = f_fn_42(x_grid_42)
    stochastic_vals_42 = {}
    for t in t_arr_42:
        V_exact = payoff_gaussian(x_grid_42 + t)
        stochastic_vals_42[float(t)] = V_exact.tolist()

    # PDE side: upwind scheme for partial_t u = partial_x u (transport with speed 1)
    pde_vals_42 = {}
    n_steps_42 = max(int(np.ceil(max(t_arr_42) / (0.5 * dx_42))), 400)
    for t in t_arr_42:
        sol = solve_transport_pde(f_fn_42, x_grid_42, t_max=t, n_steps=n_steps_42)
        pde_vals_42[float(t)] = sol["u_final"].tolist()

    # Compare
    errors_42 = {}
    for t in t_arr_42:
        err = float(np.max(np.abs(
            np.array(pde_vals_42[float(t)]) - np.array(stochastic_vals_42[float(t)])
        )))
        errors_42[float(t)] = err
        print(f"  t={t:.2f}: |u_PDE - V^G|_inf = {err:.5f}")

    avg_err_42 = float(np.mean(list(errors_42.values())))
    print(f"  Avg PDE error (Ex 4.2): {avg_err_42:.5f} (expect < 0.03)")

    # ------------------------------------------------------------------
    # Part B: Controlled diffusion — HJB PDE vs Monte Carlo
    # ------------------------------------------------------------------
    print("\n[exp6] Part B: Controlled diffusion HJB vs Monte Carlo V^G")

    controls = make_control_grid(full_run=full_run)
    x_grid_ctrl = make_x_grid()
    t_ctrl = config.T_MAX / 2.0

    sigma_vals = np.array([c["sigma"] for c in controls])
    b_vals = np.array([c["b"] for c in controls])
    c_vals = np.array([c["c"] for c in controls])

    # PDE side: monotone finite-difference HJB
    f_fn_ctrl = lambda x: payoff_f_plus(np.asarray(x))
    f_init_ctrl = f_fn_ctrl(x_grid_ctrl)
    pde_sol_ctrl = solve_hjb_controlled(
        f_fn=f_fn_ctrl,
        x_grid=x_grid_ctrl,
        t_max=t_ctrl,
        n_steps=config.N_PDE_STEPS,
        sigma_vals=sigma_vals,
        b_vals=b_vals,
        c_vals=c_vals,
    )
    u_pde_ctrl = pde_sol_ctrl["u_final"]

    # Stochastic side: compute_VG over same controls
    payoff_fn_ctrl = lambda x: np.interp(x, x_grid_ctrl, f_init_ctrl)
    V_mc = compute_VG(
        x0_vals=x_grid_ctrl,
        t=t_ctrl,
        payoff_fn=payoff_fn_ctrl,
        controls=controls,
        n_paths=n_paths,
        dt=config.DT,
        rng=np.random.default_rng(config.SEED + 600),
    )

    err_ctrl = np.abs(u_pde_ctrl - V_mc)
    max_err_ctrl = float(np.max(err_ctrl))
    mean_err_ctrl = float(np.mean(err_ctrl))
    rel_err_ctrl = float(np.mean(err_ctrl / (np.abs(V_mc) + 1e-8)))
    print(f"  Max |u_PDE - V_MC|: {max_err_ctrl:.4f}")
    print(f"  Mean |u_PDE - V_MC|: {mean_err_ctrl:.4f}")
    print(f"  Relative error (mean): {rel_err_ctrl:.4f}")

    pde_stochastic_consistent = mean_err_ctrl < 0.25
    print(f"  PDE and stochastic consistent: {pde_stochastic_consistent}")

    # Monotonicity check: u_PDE >= any single-control value
    V_single_control = compute_VG_diffusion(
        x0_vals=x_grid_ctrl,
        t=t_ctrl,
        payoff_fn=payoff_fn_ctrl,
        sigma_fn=lambda x: sigma_vals[0] * np.ones_like(x),
        b_fn=lambda x: b_vals[0] * np.ones_like(x),
        killing_fn=lambda x: -c_vals[0] * np.ones_like(x),
        n_paths=n_paths // 2,
        dt=config.DT,
        rng=np.random.default_rng(config.SEED + 700),
    )
    pde_dominates_single = float(np.mean(u_pde_ctrl >= V_single_control - 0.02))
    print(f"  PDE >= single-control fraction: {pde_dominates_single:.3f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for t in t_arr_42:
        ax.plot(x_grid_42, stochastic_vals_42[float(t)],
                label=f"V^G (t={t})", alpha=0.8)
        ax.plot(x_grid_42, pde_vals_42[float(t)], "--",
                label=f"PDE (t={t})", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_title("Ex 4.2: V^G vs PDE")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(x_grid_ctrl, u_pde_ctrl, label="u_PDE")
    ax.plot(x_grid_ctrl, V_mc, "--", label="V^G (MC)")
    ax.set_xlabel("x")
    ax.set_title(f"Controlled diffusion (t={t_ctrl:.2f}): PDE vs MC")
    ax.legend()

    ax = axes[2]
    ax.semilogy(x_grid_ctrl, err_ctrl + 1e-9, label="|u_PDE - V_MC|")
    ax.set_xlabel("x")
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Pointwise error")
    ax.legend()

    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp6_upper_perron.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    results = {
        "experiment": "exp6",
        "ex42_avg_pde_error": avg_err_42,
        "ex42_errors_by_t": errors_42,
        "ctrl_max_err": max_err_ctrl,
        "ctrl_mean_err": mean_err_ctrl,
        "ctrl_rel_err": rel_err_ctrl,
        "ctrl_pde_stochastic_consistent": pde_stochastic_consistent,
        "ctrl_pde_dominates_single_frac": pde_dominates_single,
    }
    out_path = RESULTS_DIR / "exp6_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Upper Perron / viscosity characterization")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    results = run_exp6(full_run=args.full)
    print(f"\n=== Exp 6 Summary ===")
    print(f"  Ex 4.2 avg PDE error: {results['ex42_avg_pde_error']:.5f}")
    print(f"  Controlled: mean PDE-MC error: {results['ctrl_mean_err']:.4f}")
    print(f"  Consistent: {results['ctrl_pde_stochastic_consistent']}")


if __name__ == "__main__":
    main()
