#!/usr/bin/env bash
# Rejoue le projet dans l'ordre des prompts Claude Code (P2 → P16) : notebooks de recherche
# exécutés (sorties enregistrées), scripts, tests, pipeline, rapport. Arrêt au premier échec.
#   make reproduce                       (données réelles Hillstrom, internet requis)
#   UPLIFT_SYNTHETIC=1 make reproduce    (démo hors-ligne : données synthétiques, résultats sans valeur)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
step() { printf '\n\033[1m━━ %s\033[0m\n' "$*"; }

[ "${UPLIFT_SYNTHETIC:-0}" = "1" ] && echo "⚠ DONNÉES SYNTHÉTIQUES : résultats de démonstration uniquement" || { step "P1/P3 · données réelles"; python -m data; }

step "P2  · notebook 01 — EDA & vérification de la randomisation"
python scripts/run_notebooks.py 01_eda.ipynb
step "P3  · tests données (split stratifié, randomisation)"
python -m pytest -q tests/test_data.py
step "P4  · notebook 02 — prototypage des modèles (hypothèses H1-H3)"
python scripts/run_notebooks.py 02_modeling_prototype.ipynb
step "P5  · tests modèles + évaluation (vérité terrain, Qini, uplift vs réponse)"
python -m pytest -q tests/test_models.py tests/test_evaluation.py
step "P6  · pipeline d'entraînement (sélection validation → évaluation test)"
python -m train
step "P7  · notebook 03 — prototypage de l'optimisation"
python scripts/run_notebooks.py 03_optimization_prototype.ipynb
step "P8  · tests optimisation (budget, contraintes, LP ≥ glouton)"
python -m pytest -q tests/test_optimization.py
step "P9  · notebook 04 — validation A/B (puissance, IPW, test apparié)"
python scripts/run_notebooks.py 04_ab_test_prototype.ipynb
step "P10 · tests A/B + optimisation sur le test + rapport"
python -m pytest -q tests/test_ab_test.py
python -m run_optimization
python -m report
step "P11 · tests API"
python -m pytest -q tests/test_api.py
step "P12 · smoke test du dashboard Streamlit"
python scripts/smoke_app.py
step "P14 · suite complète de tests"
python -m pytest -q -m "not slow"
step "P16 · résultats (reports/RESULTS.md)"
cat reports/RESULTS.md
echo; echo "✔ reproduction terminée"
