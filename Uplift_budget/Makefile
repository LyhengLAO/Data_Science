export PYTHONPATH := src

.PHONY: reproduce notebooks install data train optimize report all test test-nb api app docker-train docker-up clean

reproduce:      ## rejoue tout dans l'ordre des prompts (notebooks exécutés + scripts + tests)
	bash scripts/reproduce.sh

notebooks:      ## exécute les 4 notebooks et enregistre leurs sorties
	python scripts/run_notebooks.py

install:        ## dépendances (dev inclus)
	pip install -r requirements-dev.txt

data:           ## télécharge Hillstrom dans data/raw/
	python -m data

train:          ## sélection sur validation + évaluation test + artefacts
	python -m train

optimize:       ## optimisation LP + validation A/B (IPW)
	python -m run_optimization

report:         ## reports/RESULTS.md depuis les JSON
	python -m report

all: train optimize report

test:           ## tests rapides
	pytest -q

test-nb:        ## exécute les notebooks (tests d'intégration)
	UPLIFT_SYNTHETIC=1 pytest -m slow -q

api:
	uvicorn api:app --port 8000 --reload

app:
	streamlit run streamlit_app/app.py

docker-train:
	docker compose --profile train run --rm trainer

docker-up:
	docker compose up --build

clean:
	rm -rf models/* reports/* data/processed/*
