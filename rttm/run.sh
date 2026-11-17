#!/usr/bin/env bash
# Linux / macOS quick start
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt || { echo "Full install failed - installing core only"; pip install -r requirements-core.txt; }
python scripts/setup_nlp.py
streamlit run main.py
