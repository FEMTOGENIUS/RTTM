#!/usr/bin/env bash
# Linux / macOS quick start
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-full.txt || { echo "Full install failed - installing light version"; pip install -r requirements.txt; }
python scripts/setup_nlp.py
streamlit run main.py
