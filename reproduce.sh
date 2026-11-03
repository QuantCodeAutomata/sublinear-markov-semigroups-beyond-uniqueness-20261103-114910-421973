#!/usr/bin/env bash
# reproduce.sh — run all experiments for "Sublinear Markov Semigroups Beyond Uniqueness"
# All experiments use synthetic/model-defined inputs; no API keys are required.
set -uo pipefail

echo "=== Unit tests (no market data required) ==="
python -m pytest -m "not market_data" -v --tb=short

echo ""
echo "=== Exp 1: Canonical semigroup V^G construction and verification ==="
python -m exp.exp1

echo ""
echo "=== Exp 2: Semigroup–structure correspondence via R^T ==="
python -m exp.exp2

echo ""
echo "=== Exp 3: Local-global equivalence: G_T <= G iff T <= V^G ==="
python -m exp.exp3

echo ""
echo "=== Exp 4: Information loss — Example 4.1 and Example 4.2 ==="
python -m exp.exp4

echo ""
echo "=== Exp 5: Effective local specification G^downarrow ==="
python -m exp.exp5

echo ""
echo "=== Exp 6: Upper Perron / viscosity characterisation ==="
python -m exp.exp6

echo ""
echo "=== Exp 7: Uniqueness transfer and C_b-Feller property ==="
python -m exp.exp7

echo ""
echo "=== All experiments complete. Results in results/ ==="
