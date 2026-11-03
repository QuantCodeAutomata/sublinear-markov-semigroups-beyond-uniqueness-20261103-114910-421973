# Sublinear Markov Semigroups Beyond Uniqueness

**Paper:** Sublinear Markov Semigroups Beyond Uniqueness  
**Authors:** Jongjin Park, David Criens, Hyungbin Park  
**Year:** 2025

---

## Overview

This repository contains computational illustrations of the main theorems from
the paper. The paper constructs a canonical sublinear sub-Markov semigroup
V^G from a local specification G via upper expectations over admissible laws,
establishes a one-to-one correspondence between semigroups and stable structures
of laws, and characterises V^G as the greatest bounded upper-semicontinuous
viscosity subsolution of the associated Hamilton–Jacobi–Bellman equation.

All seven experiments use synthetic, model-defined inputs; no empirical market
dataset is required.

---

## Setup

```bash
pip install -r requirements.txt
```

No API keys or external data sources are required.

---

## Run all experiments

```bash
bash reproduce.sh
```

Or individually:

```bash
python -m exp.exp1   # Canonical semigroup V^G construction and verification
python -m exp.exp2   # Semigroup–structure correspondence via R^T
python -m exp.exp3   # Local-global equivalence: G_T <= G iff T <= V^G
python -m exp.exp4   # Information loss: Examples 4.1 and 4.2
python -m exp.exp5   # Effective local specification G^downarrow
python -m exp.exp6   # Upper Perron / viscosity characterisation
python -m exp.exp7   # Uniqueness transfer and C_b-Feller property
```

To run the full paper configuration (larger path counts, finer grids):

```bash
python -m exp.exp1 --full
python -m exp.exp2 --full
# ... (--full flag available for all experiments)
```

Results (JSON metrics and PNG plots) are written to `results/`.

---

## Run tests

```bash
python -m pytest -m "not market_data" -v
```

---

## Repository layout

```
src/           Library modules: local_spec, simulator, upper_expectation,
               semigroup, recovery, generator, hjb_solver, config
exp/           Experiment scripts exp1.py – exp7.py
tests/         pytest suite
data/          generate_synthetic.py (payoffs, grids, path generators)
results/       Output artifacts and RESULTS.md
```

---

## Experiments

| Script | Experiment |
|--------|-----------|
| `exp/exp1.py` | Construct and verify the canonical semigroup V^G from a prescribed local specification G via upper expectations over all admissible laws |
| `exp/exp2.py` | Recover and verify the one-to-one correspondence between semigroups and stable structures of laws via R^T |
| `exp/exp3.py` | Verify the equivalence between local upper-generator bounds, global semigroup domination, and inclusion of recovered structures |
| `exp/exp4.py` | Quantify information loss under localization and globalization using Example 4.1 and the compositions Sem∘Glob∘Loc and Loc∘Sem∘Glob |
| `exp/exp5.py` | Compute the effective local specification G^downarrow and test global realizability of a prescribed G |
| `exp/exp6.py` | Upper Perron / viscosity characterisation: compare V^G with the greatest bounded USC viscosity subsolution of the HJB equation |
| `exp/exp7.py` | Test uniqueness transfer from viscosity uniqueness/comparison to semigroup uniqueness and the C_b-Feller property |

---

## Key objects

- **G(x, z, p, H)**: local specification (sublinear in (z,p,H), monotone in H, anti-tone in z)
- **P(G)**: admissible law family solving the G-supermartingale problem
- **V_t^G f(x)**: canonical semigroup = sup_{P ∈ P_x(G)} E^P[f(X_t) 1_{t < τ_kill}]
- **R^T**: recovered structure of laws from a semigroup T
- **G_T**: upper generating function of T (infinitesimal upper generator)
- **G^↓ = G_{V^G}**: effective local specification
