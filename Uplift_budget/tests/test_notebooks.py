"""Les notebooks de recherche servent aussi de tests d'intégration : ils doivent s'exécuter
de bout en bout (sur données synthétiques en CI). `pytest -m slow` pour les lancer."""
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.slow
@pytest.mark.parametrize("nb_path", sorted(NB_DIR.glob("0*.ipynb")), ids=lambda p: p.name)
def test_notebook_runs(nb_path):
    os.environ["UPLIFT_SYNTHETIC"] = "1"
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
