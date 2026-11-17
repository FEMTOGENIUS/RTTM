"""Entry point:  streamlit run main.py   (also the "Main file path" on Streamlit Community Cloud)"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Streamlit Cloud: copy secrets (NEWSAPI_KEY, GNEWS_KEY, ...) into env vars before config is read
try:
    import streamlit as st

    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from dashboard.app import main  # noqa: E402

main()
