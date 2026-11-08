"""Exécute des notebooks et ENREGISTRE leurs sorties (équivalent de nbconvert --execute --inplace).

python scripts/run_notebooks.py 01_eda.ipynb 02_modeling_prototype.ipynb   (sans argument : tous)
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
names = sys.argv[1:] or [p.name for p in sorted(NB_DIR.glob("0*.ipynb"))]
for name in names:
    path = NB_DIR / name
    print(f" exécution de {name}", flush=True)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, path)
    print(f"  ✔ {name} exécuté, sorties enregistrées ({len(nb.cells)} cellules)")
