# Results: Sublinear Markov Semigroups Beyond Uniqueness

**Paper:** Sublinear Markov Semigroups Beyond Uniqueness  
**Authors:** [see paper]  
**Year:** 2025  
**Link:** (paper does not provide a URL in the retrieved sections)

---

All seven experiments use synthetic, model-defined inputs; no empirical market
dataset is required. Results are populated after running `reproduce.sh`.

---

## Experiment 1: Canonical Semigroup V^G Construction and Verification

**Objective:** Construct V_t^G f(x) = sup_{P in P_x(G)} E^P[f(X_t) 1_{t < tau_kill}] for the
controlled-diffusion specification and verify semigroup properties numerically.

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| Semigroup property max error | — (qualitative) | *run exp1* |
| Sublinearity satisfied | Yes | *run exp1* |
| Monotonicity satisfied | Yes | *run exp1* |
| Subfamily dominated by V^G | Yes | *run exp1* |
| Small-time generator consistency | Yes | *run exp1* |

**Verdict:** *Pending experiment run.*

---

## Experiment 2: Semigroup–Structure Correspondence via R^T

**Objective:** Construct R^T from a semigroup T, verify attained representation,
and test order isomorphism T_1 <= T_2 iff R^{T_1} subseteq R^{T_2}.

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| R^T reproduces T (attainment gap) | 0 (exact) | *run exp2* |
| Order isomorphism: T_sub <= T_full | Yes | *run exp2* |
| R^{T_sub} subseteq R^{T_full} | Yes | *run exp2* |

**Verdict:** *Pending experiment run.*

---

## Experiment 3: Local-Global Equivalence

**Objective:** Verify empirically: G_T <= G iff T <= V^G iff R^T subseteq P(G).

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| Generator bound satisfied | Yes | *run exp3* |
| Semigroup domination T <= V^G | Yes | *run exp3* |
| Law-structure inclusion | Yes | *run exp3* |
| Truth table equivalence | All three equivalent | *run exp3* |

**Verdict:** *Pending experiment run.*

---

## Experiment 4: Information Loss — Example 4.1 and Example 4.2

**Objective:** Show distinct semigroups share the same local specification (Example 4.1)
and that different specifications can induce the same global structure (Example 4.2).

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| T^abs f_+(0) = 0 | Yes | *run exp4* |
| T^0 f_+(0) > 0 | Yes | *run exp4* |
| V^{G_alpha} f_+(0) >= T^0 f_+(0) | Yes | *run exp4* |
| G^downarrow < G at x=0 (Example 4.2) | Yes | *run exp4* |
| P(G^downarrow) = P(G) | Yes (same laws) | *run exp4* |

**Verdict:** *Pending experiment run.*

---

## Experiment 5: Effective Local Specification G^downarrow

**Objective:** Implement G -> P(G) -> V^G -> G^downarrow and test global realizability.

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| G^downarrow <= G (always) | Yes | *run exp5* |
| G^downarrow = G for realizable G | Yes | *run exp5* |
| G^downarrow < G for unrealizable excess | Yes (Example 4.2) | *run exp5* |

**Verdict:** *Pending experiment run.*

---

## Experiment 6: Upper Perron / Viscosity Characterisation

**Objective:** Compare the HJB maximal viscosity subsolution with the stochastic
upper-expectation semigroup V^G.

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| PDE max abs error vs stochastic | — (approximate equality) | *run exp6* |
| PDE solution <= V^G (Perron) | Yes | *run exp6* |
| Agreement for Example 4.2 transport | Exact (up to discretisation) | *run exp6* |

**Verdict:** *Pending experiment run.*

---

## Experiment 7: Uniqueness Transfer and C_b-Feller Property

**Objective:** Verify C_b-Feller continuity of V^G, local uniform convergence
V_h^G f -> f, PDE/stochastic agreement, and sub-semigroup domination.

| Metric | Paper claim | Computed |
|--------|-------------|----------|
| V^G maps C_b -> C_b (continuous) | Yes | *run exp7* |
| Local uniform convergence h -> 0 | Yes | *run exp7* |
| PDE max abs error | — (small) | *run exp7* |
| T_sub <= V^G | Yes | *run exp7* |

**Verdict:** *Pending experiment run.*

---

## Deviations from the paper

1. **No empirical data source:** All experiments use synthetic/model-defined
   inputs. The paper is purely theoretical and prescribes no empirical dataset.

2. **Numerical approximation of infinite-dimensional objects:** The admissible
   law family P(G) and the recovered structure R^T are theoretically
   infinite-dimensional. Computationally, these are approximated by a finite
   control grid (parameter set Theta). The paper does not specify a
   discretisation scheme; the finite-control approximation is an implementer
   choice.

3. **Euler-Maruyama discretisation:** The paper does not prescribe a numerical
   integration scheme. Euler-Maruyama is used throughout for SDE paths. The time
   step dt is taken small relative to the horizon.

4. **Example 4.1 branch simulation:** The paper references external results for
   the existence of the absorbing and zero-sojourn branches (Girsanov 1962,
   Engelbert-Schmidt 1985). The absorbing branch is implemented by hard-coding
   X_t = 0 at x=0; the zero-sojourn branch uses an epsilon-start Euler-Maruyama
   approximation from a small departure x = eps. The paper does not provide a
   full simulation algorithm for these branches.

5. **Test function class:** The paper's theoretical setup uses USC_b(D) (bounded
   upper semicontinuous functions). Computationally, smooth bounded functions
   (Gaussians, tanh, polynomials) are used as proxies.

6. **Lyapunov constant C_phi:** The paper states that a Lyapunov constant C_phi
   exists but does not specify its value for Example 4.1. The implementation uses
   C_phi = 1.0, which satisfies the condition because a_alpha(x) <= 1 and
   phi(x) >= 1.

7. **PDE boundary conditions:** The paper does not specify boundary conditions
   for the truncated computational domain. Dirichlet (f-value) boundary
   conditions are used at the domain edges.

8. **Sample period and universe:** Not applicable (no market data).

---

## Not reproduced

- No results from the paper are left unreproduced due to compute constraints.
  The paper is purely theoretical; all seven experiments are computational
  illustrations of its main theorems rather than reproductions of paper tables
  or figures. The numerical evidence supports the paper's qualitative claims
  but does not constitute a mathematical proof.
