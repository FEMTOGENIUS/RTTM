"""Executes the whole Streamlit app headlessly (every page, mode and model) and
checks that everything renders without errors and with valid chart columns."""
import html
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
    import dashboard

    for name in ("app", "charts", "news", "theme"):
        sys.modules.pop(f"dashboard.{name}", None)
        if hasattr(dashboard, name):
            delattr(dashboard, name)
    app = importlib.import_module("dashboard.app")
    if session:
        mod.session_state.update(session)
    app.main()
    return mod._fake, app


BASE = {"mode": "Static (sample dataset)"}
INSIGHTS = {"page": "Insights"}


# ------------------------------------------------------------------ reader pages
def test_home_page_newspaper_layout(monkeypatch):
    st, _ = run_app(monkeypatch, BASE)
    assert st.errors == [], st.errors
    txt = st.texts()
    for part in ("News<span>Lens</span>", "TRENDING", "Top stories", "Most covered", "Latest", "class='lead'"):
        assert part in txt, part
    assert "Breaking" in txt                       # cyclone burst shown as a Breaking story
    assert "Full coverage" in txt
    assert "of 13 sources" not in txt and "Sample edition" in txt
    for part in ("NewsLens Console", "Topic model", "Coherence C_V", "Topics found", "<b>Topic 0</b>",
                 "Open topic analytics"):
        assert part in txt, part


@pytest.mark.parametrize("page", ["Politics", "World", "Business", "Tech", "Sport", "Health", "Entertainment",
                                  "Climate", "Law & Justice", "Science & Space", "Startups", "Energy", "Other news"])
def test_section_pages(monkeypatch, page):
    st, _ = run_app(monkeypatch, BASE, {"page": page})
    assert st.errors == [], (page, st.errors)
    assert f">{html.escape(page)}<" in st.texts()


def test_full_coverage_page(monkeypatch):
    st, _ = run_app(monkeypatch, BASE, {"page": "topic:0"})
    assert st.errors == []
    txt = st.texts()
    assert "All coverage" in txt and "Who is covering it" in txt and "Back to Home" in txt


def test_missing_story_page(monkeypatch):
    st, _ = run_app(monkeypatch, BASE, {"page": "topic:999"})
    assert st.errors == [] and "no longer in the current news window" in st.texts()


def test_search_results(monkeypatch):
    st, _ = run_app(monkeypatch, {**BASE, "query": "cyclone"})
    assert st.errors == [] and "Results for" in st.texts()


# ------------------------------------------------------------------ insights (analytics)
@pytest.mark.parametrize("model", ["BERTopic", "LDA", "Compare both"])
def test_insights_all_models(monkeypatch, model):
    st, _ = run_app(monkeypatch, {**BASE, "model": model, "ntopics": 9}, INSIGHTS)
    assert st.errors == [], st.errors
    assert st.count("tabs") == 1
    assert st.count("plotly_chart") >= 10
    assert st.count("download_button") == 3
    txt = st.texts()
    assert "Pipeline time" in txt
    if model != "LDA":
        assert "Trend alerts" in txt
    if model == "Compare both":
        assert "Adjusted Rand Index" in txt


def test_dark_mode_search_and_options(monkeypatch):
    st, _ = run_app(monkeypatch, {**BASE, "dark": True, "topic_search": "cyclone", "showarts": True,
                                  "btauto": False, "ntopics": 9, "weighting": "tfidf", "eng": False}, INSIGHTS)
    assert st.errors == []
    assert "1 of 9 topics" in st.texts()


def test_evaluation_form_saves_feedback(monkeypatch):
    import config

    st, _ = run_app(monkeypatch, {**BASE, "model": "BERTopic", "Submit evaluation": True}, INSIGHTS)
    assert st.errors == []
    lines = [json.loads(l) for l in config.FEEDBACK_PATH.read_text().splitlines()]
    assert any(l["type"] == "topic" and l["interpretable"] == "Yes" for l in lines)
    assert any(l["type"] == "usability" for l in lines)
    assert "Saved" in st.texts()


# ------------------------------------------------------------------ data modes
def test_live_mode_fetches_once_then_uses_store(monkeypatch):
    from data import fetchers

    calls = {"n": 0}

    def fake_fetch_all(sources, **kw):
        calls["n"] += 1
        return fetchers.FetchReport(articles=fetchers.load_sample(),
                                    per_source={"Google News RSS": 370, "BBC World": 0},
                                    errors=["RSS 'BBC World' failed: rate-limited"], seconds=0.1)

    monkeypatch.setattr(fetchers, "fetch_all", fake_fetch_all)
    ov = {"mode": "Live (real-time)", "window": "Last 7 days", "auto": True, "interval": 120}
    st, app = run_app(monkeypatch, ov)
    assert st.errors == [], st.errors
    assert st.session_state["last_fetch"]["fetched"] == 370
    assert st.count("fragment") == 1                # auto-refresh poller is active
    txt = st.texts()
    assert "1 of 2 sources live" in txt and "1 temporarily unavailable" in txt and "<i></i>LIVE" in txt
    n_calls = calls["n"]
    app.main()                                      # same session -> no new fetch
    assert calls["n"] == n_calls


def test_upload_csv(monkeypatch):
    from data.fetchers import load_sample

    df = pd.DataFrame(load_sample())[["title", "description", "content", "source", "url", "published_at"]]
    buf = io.BytesIO(df.to_csv(index=False).encode())
    buf.name = "my_news.csv"
    st, _ = run_app(monkeypatch, {"mode": "Static (upload file)", "upload": buf, "model": "LDA",
                                  "window": "All stored"}, INSIGHTS)
    assert st.errors == []
    assert st.count("plotly_chart") >= 8


def test_upload_missing_shows_info(monkeypatch):
    st, _ = run_app(monkeypatch, {"mode": "Static (upload file)"})
    assert st.errors == [] and "Upload a CSV/JSON" in st.texts()
