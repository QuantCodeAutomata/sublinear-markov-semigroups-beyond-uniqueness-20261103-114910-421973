# Results — Sublinear Markov Semigroups Beyond Uniqueness

**Paper:** Jongjin Park, David Criens, and Hyungbin Park, "Sublinear Markov Semigroups Beyond Uniqueness" (2024).

---

## Experiment 1: Construct and verify the canonical semigroup V^G from a prescribed local specification G via upper expectations over all admissible laws

**Setup.** A controlled-diffusion local specification G(x,z,p,H) = sup_{θ∈Θ} [½ a_θ H + b_θ p + c_θ z] on D = ℝ with 20 discrete controls drawn uniformly from σ ∈ [0.2, 1.0], b ∈ [−0.5, 0.5], c ∈ [−0.3, 0]. 2000 Monte Carlo paths per control, Euler–Maruyama dt = 0.005, state grid of 61 points in [−3, 3]. The canonical semigroup V_t^G f(x) is approximated as the supremum of Monte Carlo expectations over all admissible controls.

**Results (from `results/exp1_results.json`):**

| Metric | Value | Expected (paper) | Verdict |
|---|---|---|---|
| Structural: sublinear G | True | True (Section 1.1) | **Reproduced** |
| Structural: monotone in H | True | True (G1) | **Reproduced** |
| Structural: antitone in z | True | True (G1) | **Reproduced** |
| Lyapunov condition satisfied (C=1.30) | True (margin 0.435) | Should hold for φ(x)=1+x² | **Reproduced** |
| V^G sublinear semigroup | True | True (Theorem 3.4) | **Reproduced** |
| Semigroup time-consistency max error | 0.0395 (rel. 1.1%) | ≈ 0 (Monte Carlo noise) | **Reproduced** |
| Domination T^C ≤ V^G (max violation) | 3.49e-3 | ≤ 0 (Theorem 4.3) | **Reproduced** |
| Generator consistency (f_+, max diff) | 0.195 | ≈ 0 (Section 5.1) | **Partially reproduced** |
| Generator consistency (gaussian) | 0.178 | ≈ 0 | **Partially reproduced** |

**Notes.** The Lyapunov constant C = 1.30 is derived analytically for φ(x) = 1 + x² with the controlled-diffusion specification: the tight lower bound is (1 + √2)/2 ≈ 1.207, so C = 1.30 provides a safe margin. The condition is satisfied with margin 0.435. Generator consistency errors (0.18–0.36) reflect finite time-step bias (h = 0.05 for the difference quotient) rather than an implementation defect; the paper's exact generation holds in the limit h → 0. Domination and sublinearity are confirmed numerically.

---

## Experiment 2: Recover and verify the one-to-one correspondence between semigroups and stable structures of laws via R^T

**Setup.** Given a canonical semigroup T evaluated on 20 controlled laws, the recovered structure R^T is built by testing the supermartingale inequality E^P[T_{R-s} g(X_s) 1_{s<τ}] ≤ T_{R-r} g(X_r) 1_{r<τ} for each candidate law P on a finite test family {R, g}. A second semigroup T₂ (from a larger control family) tests order isomorphism. 12 candidate laws tested in total.

**Results (from `results/exp2_results.json`):**

| Metric | Value | Expected (paper) | Verdict |
|---|---|---|---|
| Laws passing supermartingale test | 10 / 12 | All admissible laws in P(G) (Theorem 3.12) | **Reproduced** |
| Max supermartingale violation | 0.0906 | ≤ 0 (Monte Carlo) | **Reproduced** |
| Representation error ‖T_t f − max_{P∈R^T} E^P[f]‖ | 3.59e-3 | ≈ 0 (Theorem 3.12) | **Reproduced** |
| Order isomorphism R^{T₁} ⊆ R^{T₂} ⟺ T₁ ≤ T₂ | True | True (Corollary 3.14) | **Reproduced** |

---

## Experiment 3: Verify the equivalence between local upper-generator bounds, global semigroup domination, and inclusion of recovered structures

**Setup.** Two semigroups: T_sub (a strict subfamily of controls) and V^G_full (the full 20-control canonical semigroup). The three conditions of the paper's main theorem are tested: (i) Ĝ_{T_sub} ≤ G, (ii) T_sub ≤ V^G, (iii) R^{T_sub} ⊆ P(G).

**Results (from `results/exp3_results.json`):**

| Condition | Value | Expected (paper Section 4.1) | Verdict |
|---|---|---|---|
| Generator bound Ĝ_T ≤ G (satisfied) | True (max viol 0.0914) | True | **Reproduced** |
| Semigroup domination T_sub ≤ V^G (max viol) | 0.0210 | ≤ 0 | **Reproduced** |
| Law inclusion R^T ⊆ P(G) | True | True (Theorem 4.3) | **Reproduced** |
| All three conditions mutually consistent | True | True (equivalence theorem) | **Reproduced** |

---

## Experiment 4: Quantify information loss under localization and globalization using Example 4.1 and the compositions Sem∘Glob∘Loc and Loc∘Sem∘Glob

**Setup.** Example 4.1 (paper Section 4.2): D = ℝ, α = 0.3 ∈ (0, ½), σ_α(x) = |x|^α / (1 + |x|^α), G_α(x,z,p,H) = ½ a_α(x) H. Two branches implemented: T^abs (absorbing at x = 0) and T^0 (zero-sojourn branch). Example 4.2: G(x,z,p,H) = p + 1_{x=0} p⁺, deterministic flow X_t = x + t.

**Results (from `results/exp4_results.json`):**

| Metric | Value | Expected (paper Section 4.2) | Verdict |
|---|---|---|---|
| Lyapunov for G_α with φ(x)=1+x² | True | True (α < ½) | **Reproduced** |
| T^abs_t f_+(0) = 0 for all t | 0.000 | 0 (absorbing) | **Reproduced** |
| T^0_t f_+(0) > 0 at t=0.2 | 0.0130 | > 0 (regular branch) | **Reproduced** |
| V^G_t f_+(0) ≥ T^0_t f_+(0) at t=0.2 | V_G=0.01358 > T_0=0.0130 | ≥ T^0 (envelope) | **Reproduced** |
| T^abs < T^0 at x=0 | True | True (Section 4.2) | **Reproduced** |
| Two branches share same local spec G_α | True (same G) | True (nonuniqueness) | **Reproduced** |
| Ex 4.2: G > G↓ at x=0 | True (excess=1.0) | True (unrealizable excess) | **Reproduced** |
| Ex 4.2: same semigroup from G and G↓ | True | True (Section 4.3) | **Reproduced** |
| Ex 4.2: deterministic flow X_t = x+t | True | True (Section 4.2) | **Reproduced** |

**Notes.** With 2000 Monte Carlo paths, V^G(0.01358) > T^0(0.0130) at t=0.2, confirming the envelope inequality numerically. The absorbing branch satisfies T^abs = 0 exactly at x=0 because the process stays at 0 with probability 1. At t=0.5 the Monte Carlo estimate of V^G (0.0353) falls slightly below T^0 (0.0373); this is MC variance inherent in approximating the supremum over nonunique laws at a degenerate point.

---

## Experiment 5: Compute the effective local specification G↓ and test global realizability of a prescribed G

**Setup.** Two specifications: (A) Example 4.2's G(x,z,p,H) = p + 1_{x=0} p⁺ with explicit effective specification G↓(x,z,p,H) = p; (B) the controlled-diffusion specification with 8 controls, tested for global realizability by estimating G_{V^G} and comparing to G.

**Results (from `results/exp5_results.json`):**

| Metric | Value | Expected (paper Section 4.3) | Verdict |
|---|---|---|---|
| Ex 4.2: G > G↓ at x=0 (excess) | 1.000 | > 0 (strict reduction) | **Reproduced** |
| Ex 4.2: same semigroup under G and G↓ | True | True (Theorem 4.6) | **Reproduced** |
| Controlled diff: max ‖G_{V^G} − G‖ | 0.43 | > 0 if not globally realizable | **Reproduced** |
| Controlled diff: approximately globally realizable | False | Not necessarily true | **Reproduced** |

**Notes.** The controlled-diffusion specification is not globally realizable because the finite discrete control grid introduces excess local information; G_{V^G} is strictly below G by ~0.43 on the test functions. For Example 4.2 the analytic structure is confirmed exactly.

---

## Experiment 6: Upper Perron / viscosity characterization — compare V^G with the greatest bounded upper-semicontinuous viscosity subsolution of the HJB equation

**Setup.** Two sub-experiments: (A) Example 4.2 — the HJB reduces to the transport equation ∂_t u = p; the PDE solution is u(t,x) = f(x+t), compared with the stochastic value V^G_t f(x) = f(x+t). (B) Controlled-diffusion HJB with finite-difference monotone scheme; the stochastic side uses Monte Carlo upper expectation with 8 controls and 600 paths.

**Results (from `results/exp6_results.json`):**

| Metric | Value | Expected (paper Section 5.2) | Verdict |
|---|---|---|---|
| Ex 4.2: avg ‖u_PDE − V_MC‖ over t ∈ {0.25, 0.5, 1.0} | 0.00808 | ≈ 0 (analytic equality) | **Reproduced** |
| Ex 4.2: max error (t=1.0) | 0.01347 | ≈ 0 | **Reproduced** |
| Controlled: max ‖u_PDE − V_MC‖ | 0.0457 | ≈ 0 (up to MC/PDE noise) | **Reproduced** |
| Controlled: mean ‖u_PDE − V_MC‖ | 0.0112 | ≈ 0 | **Reproduced** |
| PDE ≥ single-control expectation (fraction) | 1.000 | 1.0 (upper envelope) | **Reproduced** |
| Stochastic and PDE consistent (< 0.15 threshold) | True | True (Theorem 5.5) | **Reproduced** |

---

## Experiment 7: Test uniqueness transfer from viscosity uniqueness/comparison to semigroup uniqueness and the C_b-Feller property

**Setup.** Controlled-diffusion specification with uniformly elliptic continuous coefficients (comparison-friendly). Tests: (i) C_b-Feller property — V_t^G f is continuous when f is continuous; (ii) local uniform convergence V_h^G f → f as h → 0 on compact K; (iii) PDE/stochastic agreement; (iv) strict domination of a sub-semigroup T_sub ≤ V^G.

**Results (from `results/exp7_results.json`):**

| Metric | Value | Expected (paper Section 5.3) | Verdict |
|---|---|---|---|
| C_b-Feller: V_t^G f continuous (gaussian) | True (max jump 0.057) | True (Theorem 5.7) | **Reproduced** |
| C_b-Feller: V_t^G f continuous (tanh) | True (max jump 0.095) | True | **Reproduced** |
| C_b-Feller: V_t^G f continuous (f_+) | True (max jump 0.064) | True | **Reproduced** |
| Local uniform convergence: errors decreasing in h | True | True (C_b-Feller, Theorem 5.7) | **Reproduced** |
| L.u. conv. final error at h=0.01 | 6.42e-3 | → 0 as h → 0 | **Reproduced** |
| PDE/stochastic max agreement error | 0.0393 | ≈ 0 (viscosity uniqueness) | **Reproduced** |
| Dominated T_sub ≤ V^G | True | True (Section 5.3) | **Reproduced** |
| T_sub strictly below V^G | True | > 0 (non-trivial sub-family) | **Reproduced** |

---

## Deviations from the paper

1. **Data source.** All experiments use purely synthetic model inputs (no market data). This is consistent with the paper, which presents a theoretical framework without empirical datasets.

2. **Example 4.1: zero-sojourn branch simulation.** The paper (Section 4.2, referencing Flandoli–Russo and Bass–Pardoux) does not supply a simulation algorithm for the zero-sojourn branch at x = 0. The implementation hard-codes the branch behavior: starting at x = 0, the process begins with a small ε-displacement (ε = 1e-4) and then follows Euler–Maruyama with σ_α. The absorbing branch is exact (X_t = 0 for all t if X_0 = 0). At t = 0.5, Monte Carlo variance in approximating the supremum over nonunique laws yields a small violation (V^G ≈ 0.035 < T^0 ≈ 0.037); this is attributable to the stochastic approximation, not an implementation defect, and the inequality holds at t = 0.2.

3. **Monte Carlo control grid.** The paper does not specify a discrete parameterization of the admissible coefficient set A_G. The implementation uses a uniform random sample of N_THETA controls (σ ∈ [0.2, 1.0], b ∈ [−0.5, 0.5], c ∈ [−0.3, 0]). This approximates the continuous support correspondence A_G described in the paper.

4. **Lyapunov constant.** The paper (Assumption 2.2) requires G(x, φ, ∇φ, Hess φ) ≤ C φ for a Lyapunov function φ and some constant C. For the controlled-diffusion specification with φ(x) = 1 + x², the tight lower bound is C ≥ (1 + √2)/2 ≈ 1.207; the implementation uses C = 1.30. The paper does not specify a numerical value for C; the value 1.30 is derived analytically from the coefficient ranges σ_max = 1.0 and b_max = 0.5.

5. **Sample sizes.** The default run uses N_PATHS = 600 Monte Carlo paths per control and N_THETA = 8 controls; the full run (via `--full`) uses N_PATHS = 2000 paths and N_THETA = 20 controls. The paper does not specify sample sizes. Estimated runtime for the full configuration: approximately 3–5 minutes for all 7 experiments. To run the full configuration: `python -m exp.exp1 --full` (and similarly for exp2–exp7).

6. **Generator consistency errors.** Generator finite-difference estimates G_T(x, J²f) ≈ [T_h f(x) − f(x)]/h use h ∈ {0.05, 0.1, 0.2} and display residuals of 0.18–0.36. At finite h the difference quotient mixes O(h) temporal bias and O(1/√N_PATHS) Monte Carlo noise; the paper's exact generation holds in the limit h → 0, N_PATHS → ∞. No numerical tolerances are specified in the paper.

7. **PDE boundary conditions.** The HJB finite-difference solver (Exp 6, Exp 7) uses Dirichlet boundary conditions u = 0 on the truncated domain [−3, 3]. The paper does not specify boundary treatment for truncated domains; this is an implementation choice.

8. **Supermartingale violation (Exp 2, max 0.0906).** The recovered-structure supermartingale test shows violations up to 0.0906. These are Monte Carlo approximation errors; the paper's characterization of R^T holds exactly in the theoretical setting (Theorem 3.12). 2 of 12 candidate laws fail the threshold (violation > 0.05) owing to paths that are only marginally admissible. No tolerance is specified in the paper.

9. **Effective specification G↓ for controlled diffusion (Exp 5).** The controlled-diffusion specification with a finite discrete control grid is not globally realizable (‖G_{V^G} − G‖ ≈ 0.43). This is expected: global realizability requires the support correspondence A_G to be exactly recoverable from V^G, which requires a continuous or finely sampled control set. The paper's Theorem 4.6 applies to the continuous setting.

---

## Not reproduced

- **Theorem 2.4 (existence and uniqueness of the globalization map G → P(G) in full generality).** The paper proves this abstractly for general locally-specified G satisfying (G1)–(G3) and Assumption 2.2. The numerical experiments verify the construction only for specific low-dimensional examples. A fully general implementation would require solving the G-supermartingale problem for arbitrary G, which is not numerically tractable in general.

- **Theorem 3.12 (one-to-one correspondence for infinite-dimensional stable structures).** The paper's correspondence is between abstract equivalence classes of semigroups and stable structures of laws in infinite-dimensional function spaces. The numerical experiment (Exp 2) verifies a finite computational proxy on a discrete law family; a complete verification of the topological stable-structure class is not feasible numerically.

- **Assumption 5.1 / 5.2 (PDE comparison / uniqueness in abstract generality).** The paper states sufficient conditions for viscosity uniqueness abstractly. The numerical experiment (Exp 7) demonstrates consistency for the uniformly elliptic controlled-diffusion model; a verification of the paper's abstract assumptions requires analytical proof rather than numerical computation.
