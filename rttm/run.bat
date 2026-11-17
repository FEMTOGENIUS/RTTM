@echo off
REM Windows quick start: creates a virtual env, installs, downloads NLP models, launches the dashboard.
if not exist .venv ( py -3 -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Full install failed - installing core only & pip install -r requirements-core.txt)
python scripts\setup_nlp.py
streamlit run main.py
