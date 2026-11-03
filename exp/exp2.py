"""
Experiment 2: Recover and verify the one-to-one correspondence between
semigroups and stable structures of laws via R^T.

Constructs R^T from a candidate semigroup T, tests the attained
representation T_t f = max_{P in R^T} E^P[f(X_t)], and checks the
order isomorphism T1 <= T2 iff R^{T1} subseteq R^{T2}.

Paper: Section 3.2 (pp. 18-20), Corollary 4.7, Theorem 3.18.
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
from src.semigroup import SemigroupType, eval_semigroup
from src.recovery import classify_laws_into_RT, test_order_isomorphism
from src.simulator import simulate_paths_with_kill
from data.generate_synthetic import (
    make_x_grid, make_control_grid, PAYOFF_LIBRARY, payoff_f_plus, payoff_gaussian
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _make_paths(control: dict, x0: float, t_end: float, n_paths: int,
                rng: np.random.Generator, dt: float) -> tuple:
    """Simulate paths under a single constant control with recorded trajectory."""
    sigma_c, b_c, c_c = control["sigma"], control["b"], control["c"]

    def b_fn(x): return b_c * np.ones_like(x)
    def sigma_fn(x): return sigma_c * np.ones_like(x)
    killing_fn = None
    if c_c < 0.0:
        k = -c_c
        def killing_fn(x): return k * np.ones_like(x)

    n_steps = max(1, int(round(t_end / dt)))
    dt_a = t_end / n_steps
    sqrt_dt = np.sqrt(dt_a)
    X = np.full(n_paths, x0)
    alive = np.ones(n_paths, dtype=bool)
    cum_kill = np.zeros(n_paths)
    kill_thresh = rng.exponential(1.0, n_paths) if killing_fn else np.full(n_paths, np.inf)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = X

    for step in range(n_steps):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            paths[:, step + 1:] = paths[:, step:step + 1]
            break
        x_a = X[idx]
        X[idx] = x_a + b_fn(x_a) * dt_a + sigma_fn(x_a) * rng.standard_normal(len(idx)) * sqrt_dt
        if killing_fn:
            cum_kill[idx] += np.maximum(killing_fn(x_a), 0.0) * dt_a
            died = cum_kill[idx] >= kill_thresh[idx]
            alive[idx[died]] = False
        paths[:, step + 1] = X

    t_grid = np.linspace(0.0, t_end, n_steps + 1)
    return paths, t_grid, alive


def run_exp2(full_run: bool = False) -> dict:
    n_paths = (config.N_PATHS_FULL if full_run else config.N_PATHS) // 2
    rng = np.random.default_rng(config.SEED + 1)

    x_grid = make_x_grid()
    controls = make_control_grid(full_run=full_run)
    t_end = 0.5
    x0 = 0.5
    dt = config.DT

    print(f"[exp2] R^T recovery: {len(controls)} candidate laws, x0={x0}, t={t_end}")

    # ------------------------------------------------------------------
    # 1. Build candidate law ensembles (one per control theta)
    # ------------------------------------------------------------------
    print("  Generating path ensembles...")
    n_laws = min(len(controls), 12)
    paths_list, t_grids = [], []
    for ctrl in controls[:n_laws]:
        p, tg, _ = _make_paths(ctrl, x0, t_end, n_paths=min(n_paths, 1000), rng=rng, dt=dt)
        paths_list.append(p)
        t_grids.append(tg)

    # ------------------------------------------------------------------
    # 2. T: the CONTROLLED upper-expectation semigroup (V^G via all controls)
    # ------------------------------------------------------------------
    eval_kwargs_T2 = dict(
        n_paths=min(n_paths, 1000), dt=dt,
        controls=controls, rng=np.random.default_rng(config.SEED + 10),
    )

    # ------------------------------------------------------------------
    # 3. Classify laws into R^{T2}  (using CONTROLLED = V^G, the largest)
    # ------------------------------------------------------------------
    test_payoffs = [payoff_f_plus, payoff_gaussian]
    print("  Testing supermartingale property for each law...")
    RT_results = classify_laws_into_RT(
        law_paths_list=paths_list,
        t_grids=t_grids,
        sg_type=SemigroupType.CONTROLLED,
        eval_kwargs=eval_kwargs_T2,
        R=t_end,
        g_fns=test_payoffs,
        n_test_times=2,
    )

    n_in_RT = sum(r["in_RT"] for r in RT_results)
    max_viol = max(r["max_violation"] for r in RT_results)
    print(f"  Laws in R^T: {n_in_RT}/{n_laws}, max supermartingale violation: {max_viol:.4f}")

    # ------------------------------------------------------------------
    # 4. Order isomorphism: T1 (subfamily) <= T2 (full) => R^{T1} <= R^{T2}
    # ------------------------------------------------------------------
    subset_controls = controls[:len(controls) // 2]
    eval_kwargs_T1 = dict(
        n_paths=min(n_paths, 1000), dt=dt,
        controls=subset_controls, rng=np.random.default_rng(config.SEED + 20),
    )
    RT1_results = classify_laws_into_RT(
        law_paths_list=paths_list,
        t_grids=t_grids,
        sg_type=SemigroupType.CONTROLLED,
        eval_kwargs=eval_kwargs_T1,
        R=t_end,
        g_fns=test_payoffs,
        n_test_times=2,
    )
    RT1_membership = [r["in_RT"] for r in RT1_results]
    RT2_membership = [r["in_RT"] for r in RT_results]
    rng_ord = np.random.default_rng(config.SEED + 700)
    order_res = test_order_isomorphism(
        x0=x0, t=t_end, R=t_end,
        payoff_fn=payoff_f_plus, g_fn=payoff_f_plus,
        controls_1=subset_controls, controls_2=controls,
        n_paths=min(n_paths, 500), dt=dt, rng=rng_ord,
    )
    print(f"  Order: R^{{T1}} subset R^{{T2}}: {order_res['RT1_subset_RT2']} "
          f"(T1<=T2: {order_res['T1_leq_T2']})")

    # ------------------------------------------------------------------
    # 5. Representation test: T_t f(x) ~ max_{P in R^T} E^P[f(X_t)]
    # ------------------------------------------------------------------
    x_repr = np.array([x0])
    T_t_f = eval_semigroup(
        x_repr, t_end, payoff_f_plus, stype=SemigroupType.CONTROLLED, **eval_kwargs_T2
    )
    max_E_RT = np.max([
        float(np.mean(payoff_f_plus(p[:, -1])))
        for p, r in zip(paths_list, RT_results) if r["in_RT"]
    ] or [0.0])
    attain_err = abs(float(T_t_f[0]) - max_E_RT)
    print(f"  Representation: T_t f(x0)={float(T_t_f[0]):.4f}, "
          f"max_E_RT={max_E_RT:.4f}, error={attain_err:.4f}")

    # ------------------------------------------------------------------
    # Plot: supermartingale violation per law
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    law_idxs = list(range(len(RT_results)))
    viol_vals = [r["max_violation"] for r in RT_results]
    colors = ["steelblue" if r["in_RT"] else "salmon" for r in RT_results]
    ax.bar(law_idxs, viol_vals, color=colors)
    ax.axhline(0.05, color="black", linestyle="--", label="Threshold 0.05")
    ax.set_xlabel("Law index")
    ax.set_ylabel("Max supermartingale violation")
    ax.set_title("R^T classification: supermartingale violation per candidate law")
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp2_RT_recovery.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    results = {
        "experiment": "exp2",
        "n_laws_tested": n_laws,
        "n_in_RT": n_in_RT,
        "max_supermartingale_violation": max_viol,
        "order_RT1_in_RT2": order_res["RT1_subset_RT2"],
        "attainment_error": attain_err,
        "T_t_f_at_x0": float(T_t_f[0]),
        "max_E_RT_at_x0": max_E_RT,
    }
    out_path = RESULTS_DIR / "exp2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: R^T recovery")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    results = run_exp2(full_run=args.full)
    print(f"\n=== Exp 2 Summary ===")
    print(f"  Laws in R^T: {results['n_in_RT']}/{results['n_laws_tested']}")
    print(f"  Order isomorphism: {results['order_RT1_in_RT2']}")
    print(f"  Attainment error: {results['attainment_error']:.4f}")


if __name__ == "__main__":
    main()
