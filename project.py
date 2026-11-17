"""Backwards-compatible entry point (the original prototype was started with
`streamlit run project.py`). Same app as main.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.app import main  # noqa: E402

main()
