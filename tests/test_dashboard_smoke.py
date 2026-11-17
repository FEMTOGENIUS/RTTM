"""Executes the whole Streamlit app headlessly for every mode / model and
checks that every tab renders without errors."""
import importlib
import io
import json
import sys

import pandas as pd
import pytest

from tests.fakes import make_plotly, make_streamlit


def run_app(monkeypatch, overrides=None, session=None):
    mod, modules = make_streamlit(overrides)
    for name, m in {**modules, **make_plotly()}.items():
        monkeypatch.setitem(sys.modules, name, m)
    for name in ("dashboard.app", "dashboard.charts"):
        sys.modules.pop(name, None)
    app = importlib.import_module("dashboard.app")
    if session:
        mod.session_state.update(session)
    app.main()
    return mod._fake, app


BASE = {"mode": "Static (sample dataset)"}


@pytest.mark.parametrize("model", ["BERTopic", "LDA", "Compare both"])
def test_static_sample_all_models(monkeypatch, model):
    st, _ = run_app(monkeypatch, {**BASE, "model": model, "ntopics": 9})
    assert st.errors == [], st.errors
    assert st.count("tabs") == 1
    assert st.count("plotly_chart") >= 10
    assert st.count("download_button") == 3
    txt = st.texts()
    assert "Topic" in txt and "Pipeline time" in txt
    if model != "LDA":
        assert "Emerging topic" in txt          # cyclone burst alert from BERTopic trends
    if model == "Compare both":
        assert "Adjusted Rand Index" in txt


def test_dark_mode_search_and_options(monkeypatch):
    st, _ = run_app(monkeypatch, {**BASE, "dark": True, "topic_search": "cyclone", "showarts": True,
                                  "btauto": False, "ntopics": 9, "weighting": "tfidf", "eng": False})
    assert st.errors == []
    assert "1 of 9 topics" in st.texts()


def test_live_mode_fetches_once_then_uses_store(monkeypatch):
    from data import fetchers

    calls = {"n": 0}

    def fake_fetch_all(sources, **kw):
        calls["n"] += 1
        return fetchers.FetchReport(articles=fetchers.load_sample(), per_source={"Google News RSS": 370},
                                    seconds=0.1)

    monkeypatch.setattr(fetchers, "fetch_all", fake_fetch_all)
    ov = {"mode": "Live (real-time)", "window": "Last 7 days", "auto": True, "interval": 120}
    st, app = run_app(monkeypatch, ov)
    monkeypatch.setattr(app, "fetch_all", fake_fetch_all)
    st, app = run_app(monkeypatch, ov)  # fresh session: fetch happens
    assert st.errors == [], st.errors
    assert st.session_state["last_fetch"]["fetched"] == 370
    assert st.count("fragment") == 1                # auto-refresh poller is active
    n_calls = calls["n"]
    # same session, no widget change -> no new fetch, data comes from the store
    app.main()
    assert calls["n"] == n_calls


def test_upload_csv(monkeypatch):
    from data.fetchers import load_sample

    df = pd.DataFrame(load_sample())[["title", "description", "content", "source", "url", "published_at"]]
    buf = io.BytesIO(df.to_csv(index=False).encode())
    buf.name = "my_news.csv"
    st, _ = run_app(monkeypatch, {"mode": "Static (upload file)", "upload": buf, "model": "LDA", "window": "All stored"})
    assert st.errors == []
    assert st.count("plotly_chart") >= 8


def test_upload_missing_shows_info(monkeypatch):
    st, _ = run_app(monkeypatch, {"mode": "Static (upload file)"})
    assert st.errors == [] and "Upload a CSV/JSON" in st.texts()


def test_evaluation_form_saves_feedback(monkeypatch):
    import config

    st, _ = run_app(monkeypatch, {**BASE, "model": "BERTopic", "Submit evaluation": True})
    assert st.errors == []
    lines = [json.loads(l) for l in config.FEEDBACK_PATH.read_text().splitlines()]
    assert any(l["type"] == "topic" and l["interpretable"] == "Yes" for l in lines)
    assert any(l["type"] == "usability" for l in lines)
    assert "Saved" in st.texts()
