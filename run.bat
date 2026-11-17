@echo off
REM Windows quick start: creates a virtual env, installs, downloads NLP models, launches the dashboard.
if not exist .venv ( py -3 -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-full.txt || (echo Full install failed - installing light version & pip install -r requirements.txt)
python scripts\setup_nlp.py
streamlit run main.py
