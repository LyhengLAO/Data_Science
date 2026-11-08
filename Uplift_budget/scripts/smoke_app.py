"""Smoke test du dashboard Streamlit (AppTest) : aucune exception, y compris après un changement de slider."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"), default_timeout=180).run()
assert not at.exception, [e.value for e in at.exception]
print(f" app chargée : {len(at.tabs)} onglets, {len(at.metric)} métriques")
at.slider[1].set_value(300).run()
assert not at.exception, [e.value for e in at.exception]
print(" simulateur recalculé après changement de budget")
sys.exit(0)
