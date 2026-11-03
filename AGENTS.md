# AGENTS.md — Guide for coding agents and contributors

## What this repository implements

Computational illustrations of the main theorems from:

> **Sublinear Markov Semigroups Beyond Uniqueness** (2025)

The paper constructs a canonical sublinear sub-Markov semigroup V^G from a
local specification G via upper expectations over admissible laws, establishes
a one-to-one correspondence between semigroups and stable structures of laws,
and characterises V^G as the greatest bounded USC viscosity subsolution of
the associated HJB equation. All experiments use synthetic, model-defined
inputs; no market data is required.

---

## Repository layout

```
repo/
├── README.md              # Paper metadata, project description, run commands
├── AGENTS.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore
├── reproduce.sh           # Runs tests + all 7 experiments
│
├── src/                   # Flat library modules
│   ├── config.py          # All parameters (module-level UPPER_CASE variables)
│   ├── local_spec.py      # Local specification G, diffusion coefficients, Lyapunov
│   ├── simulator.py       # Euler-Maruyama SDE simulation, Cox-process killing
│   ├── upper_expectation.py  # V^G computation (Monte Carlo upper expectation)
│   ├── semigroup.py       # Semigroup evaluation, orbits, domination, Feller checks
│   ├── recovery.py        # R^T construction, supermartingale tests, order isomorphism
│   ├── generator.py       # Infinitesimal generator estimation (finite differences)
│   └── hjb_solver.py      # HJB PDE solvers (controlled diffusion, transport)
│
├── exp/                   # Experiment scripts (run as python -m exp.<name>)
│   ├── exp1.py            # Canonical semigroup construction and verification
│   ├── exp2.py            # Semigroup–structure correspondence via R^T
│   ├── exp3.py            # Local-global equivalence
│   ├── exp4.py            # Information loss (Examples 4.1, 4.2)
│   ├── exp5.py            # Effective local specification G^downarrow
│   ├── exp6.py            # Upper Perron / viscosity characterisation
│   └── exp7.py            # Uniqueness transfer and C_b-Feller property
│
├── tests/                 # pytest suite
│   ├── conftest.py        # Shared fixtures, marker registration
│   ├── test_local_spec.py
│   ├── test_simulator.py
│   ├── test_upper_expectation.py
│   ├── test_semigroup.py
│   ├── test_recovery.py
│   ├── test_generator.py
│   └── test_hjb_solver.py
│
├── results/               # All output (plots, metrics, RESULTS.md); no subdirs
│   └── RESULTS.md
│
└── data/
    ├── __init__.py
    └── generate_synthetic.py  # Synthetic payoffs, grids, control sets, path generators
```

---

## Setup

```bash
pip install -r requirements.txt
```

No API keys are required (all experiments use synthetic inputs).

---

## Run tests

```bash
# Fast unit tests only (no market data):
python -m pytest -m "not market_data" -v

# Full suite (also no market data in this project):
python -m pytest -v
```

---

## Run experiments

```bash
# Each experiment writes JSON results and PNG plots to results/
python -m exp.exp1   # Canonical semigroup V^G
python -m exp.exp2   # Semigroup-structure correspondence
python -m exp.exp3   # Local-global equivalence
python -m exp.exp4   # Information loss (Examples 4.1 and 4.2)
python -m exp.exp5   # Effective specification G^downarrow
python -m exp.exp6   # Upper Perron / viscosity characterisation
python -m exp.exp7   # Uniqueness transfer and C_b-Feller

# Full paper configuration (more paths, controls, finer grids):
python -m exp.exp1 --full
# ... (--full flag available for all experiments)

# Run everything:
bash reproduce.sh
```

---

## Conventions

- **Parameters only in `src/config.py`**: all experiment parameters are
  module-level UPPER_CASE variables with a comment on their meaning, unit,
  and paper source. Never define parameters inside exp/ scripts.

- **Data only through `data/`**: synthetic inputs are generated only in
  `data/generate_synthetic.py`. No loaders or generators in src/ or exp/.

- **No market data**: this project has no `data/data_loader.py` and no API
  keys. The `.env.example` file is absent.

- **Test rules**: unit tests in `tests/` run on small fixtures or synthetic
  inputs. The `market_data` marker is registered but unused (no live API
  calls). Run the full suite before every push.

- **After a re-run**: experiment commands overwrite files in `results/`. The
  numbers and verdicts in `results/RESULTS.md` must be updated by hand to
  match the new files.

- **Commit messages**: describe content only; no tool or agent trailers.
