"""Streamlit dashboard - Real-Time Topic Modeling for News Articles.

Run:  streamlit run main.py
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import config
from dashboard import charts, news, theme
from data.fetchers import fetch_all, load_records, load_sample
from data.store import ArticleStore
from models import engine
from models.bertopic_model import backend_summary
from models.lda_model import HAS_GENSIM
from utils.logging_setup import get_logger, tail_log
from utils.preprocess import backend_info

log = get_logger("dashboard")

SOURCE_OPTIONS = {
    "google_news": "Google News RSS (no key)",
    "rss": "RSS feeds: BBC, Al Jazeera, CNN, Indian Express, Guardian, NPR (no key)",
    "newsapi": "NewsAPI.org (API key)",
    "gnews": "GNews.io (API key)",
}
LANGS = {"English": "en", "Hindi": "hi", "French": "fr", "German": "de", "Spanish": "es"}
COUNTRIES = {"India": "IN", "United States": "US", "United Kingdom": "GB", "Australia": "AU",
             "Canada": "CA", "Singapore": "SG"}
WINDOWS = {"Last 6 hours": 6, "Last 12 hours": 12, "Last 24 hours": 24, "Last 48 hours": 48,
           "Last 3 days": 72, "Last 7 days": 168, "All stored": None}


# --------------------------------------------------------------------------- version helpers
def _st_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (1, 40)


NEW_WIDTH_API = _st_version() >= (1, 50)


def _stretch() -> dict:
    return {"width": "stretch"} if NEW_WIDTH_API else {"use_container_width": True}


def _image_stretch() -> dict:
    if NEW_WIDTH_API:
        return {"width": "stretch"}
    return {"use_container_width": True} if _st_version() >= (1, 40) else {"use_column_width": True}


def plot(fig, key: str | None = None):
    st.plotly_chart(fig, key=key, **_stretch())


def table(df: pd.DataFrame, **kw):
    st.dataframe(df, hide_index=True, **_stretch(), **kw)


def _toggle(label: str, value: bool, key: str, help: str | None = None) -> bool:
    fn = getattr(st, "toggle", st.checkbox)
    return fn(label, value=value, key=key, help=help)


# --------------------------------------------------------------------------- theme
def apply_theme(dark: bool) -> None:
    """Light / dark mode (Streamlit theme + modern CSS + plotly template)."""
    want = "dark" if dark else "light"
    try:
        from streamlit import config as st_config

        if st_config.get_option("theme.base") != want:
            st_config.set_option("theme.base", want)
    except Exception:
        pass
    st.markdown(f"<style>{theme.css(dark)}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
MODE_LABELS = {"Live (real-time)": "Live", "Static (sample dataset)": "Sample", "Static (upload file)": "Upload"}
SOURCE_SHORT = {"google_news": "Google News", "rss": "Publisher RSS", "newsapi": "NewsAPI", "gnews": "GNews"}


def _segmented(container, label: str, options: list, key: str, default=None, format_func=str, help=None):
    """Segmented control (Streamlit >= 1.40) with a horizontal-radio fallback.
    A de-selected segmented control falls back to the last chosen value."""
    default = default if default is not None else options[0]
    fn = getattr(container, "segmented_control", None)
    if callable(fn):
        try:
            val = fn(label, options, default=default, format_func=format_func, key=key, help=help)
        except TypeError:
            val = None
        last_key = f"__last_{key}"
        if val is None:
            val = st.session_state.get(last_key, default)
        st.session_state[last_key] = val
        return val
    return container.radio(label, options, index=options.index(default), format_func=format_func,
                           key=key, horizontal=True, help=help)


def _section(container, title: str) -> None:
    container.markdown(f"<div class='sb-sec'>{title}</div>", unsafe_allow_html=True)


def settings_panel() -> dict:
    """All controls, shown inside the Settings pop-over in the top bar."""
    s: dict = {}
    sb = st

    _section(sb, "Data source")
    s["mode"] = _segmented(sb, "Mode", list(MODE_LABELS), key="mode", format_func=MODE_LABELS.get,
                           help="Live fetches news continuously. Sample and Upload work offline.")
    if s["mode"].startswith("Live"):
        s["sources"] = sb.multiselect("Sources", list(SOURCE_OPTIONS), default=["google_news", "rss"],
                                      format_func=SOURCE_SHORT.get, key="sources",
                                      help="Google News and Publisher RSS need no API key.")
        c1, c2 = sb.columns(2)
        s["language"] = LANGS[c1.selectbox("Language", list(LANGS), key="lang")]
        s["country"] = COUNTRIES[c2.selectbox("Region", list(COUNTRIES), key="country")]
        s["limit"] = sb.slider("Articles per source", 10, 100, 40, 5, key="limit")
        s["auto_refresh"] = sb.toggle("Stream new articles", value=False, key="auto",
                                      help="Re-fetch automatically and update topics when new articles arrive.") \
            if hasattr(sb, "toggle") else sb.checkbox("Stream new articles", False, key="auto")
        s["interval"] = sb.slider("Refresh every (seconds)", 60, 1800, 300, 30, key="interval",
                                  disabled=not s["auto_refresh"])
        with sb.expander("API keys"):
            s["newsapi_key"] = st.text_input("NewsAPI key(s)", type="password", key="nk",
                                             help="Comma-separated; otherwise read from .env / secrets")
            s["gnews_key"] = st.text_input("GNews key(s)", type="password", key="gk")
            s["full_text"] = st.checkbox("Download full article text (slower)", key="ft")
    else:
        s.update(sources=[], language="en", country="IN", limit=0, newsapi_key="", gnews_key="",
                 full_text=False, auto_refresh=False, interval=300)
        if s["mode"].endswith("upload file)"):
            s["upload"] = sb.file_uploader("CSV or JSON file", type=["csv", "json"], key="upload",
                                           help="Columns: title, description/content, date, source, url")

    _section(sb, "Topic model")
    model_choice = _segmented(sb, "Model", ["BERTopic", "LDA", "Compare both"], key="model",
                              format_func=lambda x: "Compare" if x == "Compare both" else x)
    s["models"] = ["BERTopic", "LDA"] if model_choice == "Compare both" else [model_choice]
    s["n_topics"] = sb.slider("Number of topics", 2, 20, 8, key="ntopics")
    s["bt_auto"] = sb.checkbox("Auto topic count for BERTopic (HDBSCAN)", value=True, key="btauto",
                               disabled="BERTopic" not in s["models"])
    with sb.expander("Advanced settings"):
        s["min_topic_size"] = st.number_input("Min topic size (0 = auto)", 0, 100, 0, key="mts")
        s["reduce_outliers"] = st.checkbox("Re-assign outliers to nearest topic", True, key="ro")
        s["embedding_model"] = st.text_input("Embedding model", config.EMBEDDING_MODEL, key="emb")
        s["lda_passes"] = st.slider("LDA passes", 2, 30, 10, key="passes")
        s["lda_weighting"] = st.radio("LDA input", ["bow", "tfidf"], key="weighting", horizontal=True,
                                      format_func={"bow": "Word counts", "tfidf": "TF-IDF"}.get)
        s["english_only"] = st.checkbox("English only", True, key="eng")

    _section(sb, "Display")
    c1, c2 = sb.columns(2)
    with c1:
        s["dark"] = _toggle("Dark mode", False, "dark")
    with c2:
        s["show_articles"] = _toggle("Show articles", True, "showarts")

    c1, c2 = sb.columns(2)
    s["fetch_now_btn"] = c1.button("Fetch now", disabled=not s["mode"].startswith("Live"), type="primary",
                                   **_btn_stretch())
    s["rerun_models"] = c2.button("Re-run models", **_btn_stretch())
    full = HAS_GENSIM and "installed" in backend_summary()["bertopic"]
    sb.caption(f"{'Full NLP stack' if full else 'Lite NLP back-ends'} · NewsLens v3")
    return s


def _btn_stretch() -> dict:
    if NEW_WIDTH_API:
        return {"width": "stretch"}
    return {"use_container_width": True}


# --------------------------------------------------------------------------- data
def do_fetch(s: dict, store: ArticleStore) -> dict:
    nk = [k.strip() for k in s.get("newsapi_key", "").split(",") if k.strip()] or None
    gk = [k.strip() for k in s.get("gnews_key", "").split(",") if k.strip()] or None
    rep = fetch_all(s["sources"], query=s["query"], language=s["language"], limit=s["limit"],
                    hours=s["window"] or 72, newsapi_keys=nk, gnews_keys=gk, country=s["country"],
                    full_text=s.get("full_text", False))
    for a in rep.articles:
        a["query"] = s["query"]
    new = store.add(rep.articles)
    store.prune()
    info = {"time": datetime.now(timezone.utc), "fetched": len(rep.articles), "new": new,
            "errors": rep.errors, "per_source": rep.per_source, "seconds": rep.seconds}
    st.session_state["last_fetch"] = info
    st.session_state["last_fetch_ts"] = time.time()
    st.session_state["fetched_for"] = (tuple(s["sources"]), s["query"], s["language"], s["country"])
    return info


def _matches_query(a: dict, query: str) -> bool:
    if not query:
        return True
    if a.get("query", None) == query:
        return True
    blob = f"{a.get('title', '')} {a.get('description', '')} {a.get('content', '')}".lower()
    return any(t in blob for t in query.lower().split())


def _in_window(rows: list[dict], hours: float | None) -> list[dict]:
    if not hours:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return [r for r in rows if datetime.fromisoformat(r["published_at"]) >= cutoff]


def load_articles(s: dict, store: ArticleStore) -> list[dict]:
    if s["mode"].startswith("Live"):
        key = (tuple(s["sources"]), s["query"], s["language"], s["country"])
        if not s["sources"]:
            st.warning("Select at least one news source in the sidebar.")
            return []
        if s["fetch_now"] or st.session_state.get("fetched_for") != key:
            with st.spinner("Fetching live news…"):
                do_fetch(s, store)
        rows = [r for r in store.load(hours=s["window"]) if _matches_query(r, s["query"])]
        return rows
    if s["mode"].startswith("Static (sample"):
        rows = load_sample()
    else:
        up = s.get("upload")
        if up is None:
            st.info("Upload a CSV/JSON file in the sidebar, or switch to the sample dataset.")
            return []
        try:
            if up.name.lower().endswith(".csv"):
                recs = pd.read_csv(up).to_dict("records")
            else:
                raw = json.load(io.TextIOWrapper(up, encoding="utf-8"))
                recs = raw.get("articles", raw) if isinstance(raw, dict) else raw
            rows = load_records(recs)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            log.error("upload failed: %s", exc)
            return []
    rows = [r for r in rows if _matches_query(r, s["query"])]
    return _in_window(rows, s["window"])


def run_models(rows: list[dict], s: dict) -> engine.PipelineOutput:
    params = dict(models=tuple(s["models"]), n_topics=s["n_topics"],
                  bertopic_nr_topics="auto" if s["bt_auto"] else s["n_topics"],
                  min_topic_size=int(s["min_topic_size"]) or None, lda_passes=s["lda_passes"],
                  lda_weighting=s["lda_weighting"], embedding_model=s["embedding_model"],
                  reduce_outliers=s["reduce_outliers"], english_only=s["english_only"])
    key = hashlib.sha1((json.dumps(sorted(r["id"] for r in rows)) + json.dumps(params, default=str))
                       .encode()).hexdigest()
    if st.session_state.get("pipeline_key") != key or s["rerun_models"] or "output" not in st.session_state:
        with st.spinner(f"Running {' + '.join(s['models'])} on {len(rows)} articles…"):
            out = engine.run_pipeline(rows, **params)
        st.session_state["output"] = out
        st.session_state["pipeline_key"] = key
    return st.session_state["output"]


# --------------------------------------------------------------------------- views
def insights_header(out: engine.PipelineOutput, s: dict, n_rows: int) -> None:
    live = s["mode"].startswith("Live")
    lf = st.session_state.get("last_fetch")
    main_model = s["models"][0] if s["models"][0] in out.results else next(iter(out.results), "-")
    n_topics = len(out.results[main_model].topic_ids) if main_model in out.results else 0
    st.markdown(theme.page_head("Insights", "#121212"), unsafe_allow_html=True)
    st.caption("How the topic models see the news: quality metrics, trends, maps, model comparison, export "
               "and human evaluation.")
    cov = out.metrics.get(main_model, {}).get("coherence_cv")
    items = [("", "Articles analysed", str(len(out.df)), f"{n_rows} loaded"),
             ("", "Sources", str(out.df["source"].nunique() if not out.df.empty else 0), "publishers"),
             ("", "Topics", str(n_topics), main_model),
             ("", "Pipeline time", f"{out.timings.get('total', 0):.1f}s", "fetch → model → metrics"),
             (("", "New this fetch", str(lf["new"]), f"{lf['seconds']:.1f}s fetch") if (lf and live) else
              ("", "Coherence C_V", f"{cov:.2f}" if cov is not None and cov == cov else "-", "higher is better"))]
    st.markdown(theme.kpis(items), unsafe_allow_html=True)
    for w in out.warnings:
        st.warning(w)
    alerts = [f"[{m}] {a}" for m in out.results for a in out.alerts.get(m, [])]
    if alerts:
        with st.expander(f"Trend alerts ({len(alerts)})"):
            for a in alerts:
                st.markdown(f"- {a}")


def source_status(s: dict) -> str:
    lf = st.session_state.get("last_fetch")
    if not (lf and s["mode"].startswith("Live")):
        return "<span class='dot' style='background:#94A3B8'></span>Sample edition (offline)"
    ok = sum(1 for v in lf["per_source"].values() if v)
    total = len(lf["per_source"])
    bad = total - ok
    msg = f"<span class='dot' style='background:#16A34A'></span>{ok} of {total} sources live"
    if bad:
        msg += (f" · <span class='dot' style='background:#D97706'></span>{bad} temporarily unavailable "
                "(rate-limited or blocked by the publisher; see Insights → System)")
    return msg


def _pick_model(out: engine.PipelineOutput, s: dict, key: str) -> str | None:
    avail = [m for m in s["models"] if m in out.results] + [m for m in out.results if m not in s["models"]]
    if not avail:
        return None
    if len(avail) == 1:
        return avail[0]
    return st.radio("Model", avail, horizontal=True, key=key)


def tab_overview(out, s, dark):
    model = _pick_model(out, s, "ov_model")
    if not model:
        return
    names = out.names(model)
    docs = out.doc_frame(model)
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(charts.category_treemap(docs, dark), key="treemap")
    with c2:
        plot(charts.topic_pie(out.results[model].sizes(), names, dark), key="pie")
    c1, c2 = st.columns([2, 3])
    with c1:
        plot(charts.word_freq_bar(out.word_freq, dark), key="wf")
    with c2:
        st.markdown("#### Live feed")
        lab = out.labels[model]
        feed = docs.sort_values("published_at", ascending=False).head(12)
        cards = []
        for _, r in feed.iterrows():
            L = lab.get(int(r["topic"]), {"icon": "📰", "category": "Outlier"})
            cards.append(theme.article_card(r["title"], r.get("url", ""), r["source"], r["published_at"],
                                            theme.short_cat(L['category']), int(r["topic"])))
        st.markdown("".join(cards), unsafe_allow_html=True)
    plot(charts.sources_bar(out.df, dark), key="sources")


def _choose(label: str, options: list, key: str, format_func=str):
    """Pills (Streamlit >= 1.40) with a selectbox fallback."""
    fn = getattr(st, "pills", None)
    val = None
    if callable(fn):
        try:
            val = fn(label, options, format_func=format_func, key=key, default=options[0])
        except TypeError:
            val = None
    else:
        val = st.selectbox(label, options, format_func=format_func, key=key)
    return val if val in options else options[0]


def tab_topics(out, s, dark):
    model = _pick_model(out, s, "tp_model")
    if not model:
        return
    res, lab = out.results[model], out.labels[model]
    q = st.text_input("Search topics by keyword", key="topic_search",
                      placeholder="e.g. election, vaccine, AI, budget …")
    ids = engine.search_topics(out, model, q) if q else res.topic_ids
    st.caption(f"{len(ids)} of {len(res.topic_ids)} topics · backend: {res.backend}")
    for n in res.notes:
        st.caption(f"ℹ️ {n}")
    if not ids:
        st.info("No topic matches this keyword.")
        return
    sizes = res.sizes()
    total = max(len(out.df), 1)
    trends = out.trends[model].set_index("topic") if model in out.trends and not out.trends[model].empty \
        else pd.DataFrame()
    status = {t: (trends.loc[t, "status"] if t in trends.index else "") for t in ids}
    # ---- overview grid of topic cards
    for row in range(0, len(ids), 4):
        cols = st.columns(4)
        for col, t in zip(cols, ids[row:row + 4]):
            with col:
                st.markdown(theme.topic_card(t, lab[t], sizes.get(t, 0), 100 * sizes.get(t, 0) / total,
                                             res.top_words(t, 6), status[t]), unsafe_allow_html=True)
    st.markdown("&nbsp;")
    # ---- explorer for one topic
    t = _choose("Explore a topic", ids, key=f"pick_{model}",
                format_func=lambda x: f"T{x} · {lab[x]['keywords']}")
    L = lab[t]
    color = theme.topic_color(t)
    words = res.topic_words[t]
    st.markdown(f"<div class='card' style='border-left:6px solid {color}'><h3 style='margin:0'>"
                f"{theme.esc(L['name'])} {theme.status_badge(status[t])}</h3><div class='muted'>"
                f"{sizes.get(t, 0)} articles · {100 * sizes.get(t, 0) / total:.1f}% of coverage · "
                f"label confidence {L['confidence']:.0%}</div><div style='margin-top:8px'>"
                + "".join(theme.chip(f"{w} {sc:.3f}", color) for w, sc in words[:10]) + "</div></div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.keywords_bar(words, dark, color=color), key=f"kw_{model}_{t}")
    with c2:
        img = charts.wordcloud_image(words, dark)
        if img is not None:
            st.image(img, **_image_stretch())
        else:
            plot(charts.wordcloud_plotly(words, dark), key=f"wc_{model}_{t}")
    if s["show_articles"]:
        docs = out.doc_frame(model)
        sub = docs[docs["topic"] == t].sort_values("published_at", ascending=False)
        st.markdown(f"#### Articles in this topic ({len(sub)})")
        st.markdown("".join(theme.article_card(r["title"], r.get("url", ""), r["source"], r["published_at"],
                                               f"T{t}", t, f"· match {r['confidence']:.2f}")
                            for _, r in sub.head(40).iterrows()), unsafe_allow_html=True)
        if len(sub) > 40:
            st.caption(f"… and {len(sub) - 40} more (see the Articles tab)")


def tab_trends(out, s, dark):
    model = _pick_model(out, s, "tr_model")
    if not model:
        return
    names = out.names(model)
    tot = engine.tr.topics_over_time(out.df, out.results[model].assignments, names)
    if tot.empty or tot["time"].nunique() < 2:
        st.info("Not enough time spread yet for a timeline – keep the app running or widen the time window.")
    else:
        c1, c2 = st.columns([1, 1])
        value = c1.radio("Show", ["count", "share"], horizontal=True, key="tl_val")
        stacked = c2.checkbox("Stacked area", key="tl_stack")
        plot(charts.timeline(tot, value, dark, stacked), key="timeline")
        plot(charts.topic_race(tot, dark), key="race")
    trends = out.trends.get(model)
    if trends is not None and not trends.empty:
        plot(charts.growth_bar(trends, dark), key="growth")
        st.markdown("**Trending / fading / emerging topics** (recent window vs. earlier articles; "
                    "burst z-score = latest time bin vs. history)")
        table(trends[["status", "name", "total", "recent", "earlier", "growth", "burst_z"]])


def tab_maps(out, s, dark):
    model = _pick_model(out, s, "map_model")
    if not model:
        return
    res = out.results[model]
    names = out.names(model)
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.intertopic_scatter(engine.intertopic_map(res), names, dark, model), key="itm")
        st.caption("Distance between bubbles = dissimilarity of topics (Jensen-Shannon for LDA, "
                   "cosine between topic embeddings for BERTopic); bubble size = number of articles.")
    with c2:
        if res.doc_coords is not None:
            plot(charts.doc_map(res.doc_coords, res.assignments, out.df["title"].tolist(), names, dark, model),
                 key="docmap")
            st.caption("Each point is an article (2-D projection of the reduced embedding space).")
    if model == "LDA" and HAS_GENSIM:
        with st.expander("pyLDAvis interactive view (gensim LDA)"):
            try:
                import pyLDAvis  # type: ignore
                import pyLDAvis.gensim_models  # type: ignore
                import streamlit.components.v1 as components

                ex = res.extra
                vis = pyLDAvis.gensim_models.prepare(ex["model"], ex["corpus"], ex["dictionary"], sort_topics=False)
                components.html(pyLDAvis.prepared_data_to_html(vis), height=820, scrolling=True)
                st.caption("Note: pyLDAvis numbers topics 1..K in the model's original order.")
            except Exception as exc:
                st.info(f"pyLDAvis not available ({exc}). Install with `pip install pyLDAvis`.")


def tab_compare(out, s, dark):
    if not out.metrics:
        return
    rows = []
    for m, v in out.metrics.items():
        rows.append({"Model": m, "Backend": out.results[m].backend, "Topics": v["n_topics"],
                     "Coherence C_V": round(v["coherence_cv"], 3),
                     "Coherence NPMI": round(v["coherence_npmi"], 3),
                     "Coherence UMass": round(v["coherence_umass"], 3),
                     "Diversity (top-25)": f"{100 * v['diversity']:.1f}%",
                     "Raw outliers": f"{100 * v['outlier_rate_raw']:.1f}%",
                     "Model runtime (s)": round(v["runtime_s"], 2)})
    table(pd.DataFrame(rows))
    hum = feedback_summary()
    if not hum.empty:
        st.markdown("**Human interpretability (from the Evaluate tab)**")
        table(hum)
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.metrics_bar(out.metrics, dark), key="mbar")
    with c2:
        plot(charts.latency_bar({f"{m} model": out.metrics[m]["runtime_s"] for m in out.metrics}, dark,
                                "Model runtime (s)"), key="mlat")
    if out.comparison:
        cmp = out.comparison
        st.markdown(f"**Agreement between models** - Adjusted Rand Index: `{cmp['ari']:.3f}`, "
                    f"Normalised Mutual Information: `{cmp['nmi']:.3f}` (1 = identical clustering)")
        ln, bn = out.names("LDA"), out.names("BERTopic")
        plot(charts.jaccard_heatmap(cmp["jaccard"], [ln[t][:45] for t in cmp["lda_ids"]],
                                    [bn[t][:45] for t in cmp["bertopic_ids"]], dark), key="jac")
    elif len(out.results) < 2:
        st.info("Select **Compare both** in the sidebar to run LDA and BERTopic side by side.")
    st.caption("C_V coherence (Röder et al., 2015) is computed on the same tokenised articles "
               "(gensim CoherenceModel when installed, otherwise an equivalent built-in implementation). "
               "Diversity = unique words among the top-25 words of all topics (Dieng et al., 2020).")


def tab_articles(out, s, dark):
    model = _pick_model(out, s, "art_model")
    if not model:
        return
    d = out.doc_frame(model)
    c1, c2, c3 = st.columns([2, 2, 3])
    srcs = c1.multiselect("Source", sorted(d["source"].unique()), key="f_src")
    tops = c2.multiselect("Topic", sorted(d["topic_name"].unique()), key="f_topic")
    kw = c3.text_input("Keyword filter", key="f_kw")
    if srcs:
        d = d[d["source"].isin(srcs)]
    if tops:
        d = d[d["topic_name"].isin(tops)]
    if kw:
        m = d["title"].str.contains(kw, case=False, na=False) | d["text"].str.contains(kw, case=False, na=False)
        d = d[m]
    if not d.empty:
        dates = d["published_at"].dt.date
        lo, hi = dates.min(), dates.max()
        if lo < hi:
            rng = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), key="f_dates")
            d = d[(dates >= rng[0]) & (dates <= rng[1])]
    view = d[["published_at", "title", "source", "topic_name", "keywords", "confidence", "url"]].sort_values(
        "published_at", ascending=False)
    cfg = {}
    if hasattr(st, "column_config"):
        cfg = {"url": st.column_config.LinkColumn("url"),
               "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0)}
    table(view, column_config=cfg)
    st.caption(f"{len(view)} articles")
    export = d.drop(columns=["clean"], errors="ignore").copy()
    export["published_at"] = export["published_at"].astype(str)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Articles + topics (CSV)", export.to_csv(index=False).encode("utf-8"),
                       f"articles_{model.lower()}.csv", "text/csv")
    c2.download_button("⬇️ Articles + topics (JSON)",
                       export.to_json(orient="records", force_ascii=False, indent=1).encode("utf-8"),
                       f"articles_{model.lower()}.json", "application/json")
    c3.download_button("⬇️ Topics + metrics (JSON)", json.dumps(topics_export(out), indent=1, default=str),
                       "topics_metrics.json", "application/json")


def topics_export(out) -> dict:
    data = {"generated_at": datetime.now(timezone.utc).isoformat(), "n_articles": len(out.df),
            "preprocessing": out.prep_stats, "timings_s": out.timings, "models": {}}
    for m, r in out.results.items():
        sizes = r.sizes()
        data["models"][m] = {
            "backend": r.backend,
            "metrics": {k: v for k, v in out.metrics[m].items()},
            "topics": [{"id": t, **out.labels[m][t], "size": sizes.get(t, 0),
                        "words": [[w, round(sc, 5)] for w, sc in r.topic_words[t][:15]]} for t in r.topic_ids],
        }
    if out.comparison:
        data["comparison"] = {"ari": out.comparison["ari"], "nmi": out.comparison["nmi"],
                              "best_matches": out.comparison["best_matches"]}
    return data


# --------------------------------------------------------------------------- evaluation (human)
def feedback_summary() -> pd.DataFrame:
    p = config.FEEDBACK_PATH
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("type") == "topic"]
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    g = d.groupby("model").agg(ratings=("interpretable", "size"),
                               interpretable=("interpretable", lambda x: (x == "Yes").mean()),
                               related=("articles_related", lambda x: (x == "Yes").mean())).reset_index()
    g["interpretable"] = (100 * g["interpretable"]).round(1).astype(str) + "%"
    g["related"] = (100 * g["related"]).round(1).astype(str) + "%"
    return g.rename(columns={"interpretable": "% topics interpretable", "related": "% articles related"})


def tab_evaluate(out, s, dark):
    st.markdown("Human-interpretability review, as described in the evaluation plan: for each topic answer "
                "*“Can you understand what this topic is about from its keywords?”* and *“Do the articles under "
                "this topic seem related?”*. Ratings are saved to `logs/feedback.jsonl` and aggregated in the "
                "**Compare Models** tab.")
    model = _pick_model(out, s, "ev_model")
    if not model:
        return
    res = out.results[model]
    docs = out.doc_frame(model)
    with st.form(f"eval_{model}"):
        rater = st.text_input("Your name / initials (optional)")
        answers = {}
        for t in res.topic_ids:
            st.markdown(f"**Topic {t}** - {', '.join(res.top_words(t, 10))}")
            ex = docs[docs["topic"] == t]["title"].head(3).tolist()
            st.caption(" | ".join(ex))
            c1, c2 = st.columns(2)
            answers[t] = (c1.radio("Interpretable?", ["Yes", "Partly", "No"], key=f"i_{model}_{t}", horizontal=True),
                          c2.radio("Articles related?", ["Yes", "No"], key=f"r_{model}_{t}", horizontal=True))
        st.markdown("---")
        usability = st.slider("Overall usability of the dashboard (1-5)", 1, 5, 4)
        comment = st.text_area("Comments / suggestions")
        if st.form_submit_button("Submit evaluation"):
            config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            with open(config.FEEDBACK_PATH, "a", encoding="utf-8") as fh:
                for t, (i, r) in answers.items():
                    fh.write(json.dumps({"type": "topic", "ts": ts, "rater": rater, "model": model, "topic": t,
                                         "keywords": res.top_words(t, 10), "interpretable": i,
                                         "articles_related": r}) + "\n")
                fh.write(json.dumps({"type": "usability", "ts": ts, "rater": rater, "score": usability,
                                     "comment": comment}) + "\n")
            st.success(f"Saved {len(answers)} topic ratings. Thank you!")
    hum = feedback_summary()
    if not hum.empty:
        table(hum)


def tab_system(out, s, dark, store: ArticleStore, t_start: float):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Active back-ends**")
        info = {"Preprocessing": backend_info(), "BERTopic": backend_summary(),
                "LDA": "gensim LdaModel" if HAS_GENSIM else "scikit-learn LDA (install gensim for the paper setup)"}
        st.json(info)
        st.markdown("**Preprocessing summary**")
        st.json(out.prep_stats)
    with c2:
        timings = dict(out.timings)
        lf = st.session_state.get("last_fetch")
        if lf and s["mode"].startswith("Live"):
            timings = {"fetch (last)": lf["seconds"], **timings}
        plot(charts.latency_bar(timings, dark, "Latency by stage (s)"), key="sys_lat")
        render_ph = st.empty()
        rows = store.all()
        st.markdown(f"**Article store**: `{store.path}` - {len(rows)} articles")
        if rows:
            ts = sorted(r["published_at"] for r in rows)
            st.caption(f"oldest {ts[0][:16]} · newest {ts[-1][:16]}")
        if lf:
            st.markdown("**Last fetch per source**")
            st.json(lf["per_source"])
        if st.button("🗑️ Clear article store"):
            store.clear()
            st.session_state.pop("fetched_for", None)
            st.success("Store cleared.")
    with st.expander("Application log (logs/rttm.log)"):
        st.code(tail_log(150) or "(empty)", language="log")
    return render_ph


# --------------------------------------------------------------------------- auto refresh
def auto_refresh(s: dict, store: ArticleStore) -> None:
    if not (s["mode"].startswith("Live") and s["auto_refresh"]):
        return
    interval = int(s["interval"])
    fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if fragment is None:
        st.caption("Auto-refresh needs streamlit >= 1.37 (pip install -U streamlit).")
        return

    @fragment(run_every=interval)
    def _poll():
        due = time.time() - st.session_state.get("last_fetch_ts", 0) >= interval * 0.95
        if due:
            info = do_fetch(s, store)
            log.info("auto-refresh: %d new articles", info["new"])
            if info["new"] > 0:
                st.rerun()
        nxt = datetime.fromtimestamp(st.session_state.get("last_fetch_ts", time.time()) + interval)
        st.caption(f"🔄 Streaming: next fetch at {nxt.strftime('%H:%M:%S')}")

    _poll()


# --------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="NewsLens · real-time news, clustered", page_icon="📰", layout="wide",
                       initial_sidebar_state="collapsed")
    t_start = time.perf_counter()
    # ---- top bar: edition line | search | time window | settings | refresh
    top = st.columns([3.0, 1.6, 1.1, 0.9, 0.9], vertical_alignment="center")
    util_ph = top[0].empty()
    with top[3]:            # settings first: the default time window depends on the mode
        with news._popover("Settings"):
            s = settings_panel()
    with top[1]:
        query = st.text_input("Search", value="", key="query", placeholder="Search news, e.g. election",
                              label_visibility="collapsed").strip()
    with top[2]:
        win_label = st.selectbox("Time window", list(WINDOWS), index=3 if s["mode"].startswith("Live") else 5,
                                 key="window", label_visibility="collapsed")
    with top[4]:
        refresh = st.button("Refresh", type="primary", key="refresh_btn", **_btn_stretch())
    s["query"], s["window"] = query, WINDOWS[win_label]
    live = s["mode"].startswith("Live")
    s["fetch_now"] = bool(s.get("fetch_now_btn")) or (refresh and live)
    s["rerun_models"] = bool(s.get("rerun_models")) or (refresh and not live)
    apply_theme(s["dark"])
    store = ArticleStore()
    try:
        rows = load_articles(s, store)
    except Exception as exc:  # never crash the UI
        log.exception("loading articles failed")
        st.error(f"Loading articles failed: {exc}")
        rows = []
    auto_refresh(s, store)
    lf = st.session_state.get("last_fetch")
    updated = lf["time"].astimezone(theme.ZoneInfo(theme.REGION_TZ.get(s["country"], "UTC"))).strftime("%H:%M") \
        if (lf and live) else "now"
    util_ph.markdown(theme.utility_bar(s["country"], live, updated), unsafe_allow_html=True)
    st.markdown(theme.masthead(), unsafe_allow_html=True)
    news.render_nav()
    if not rows:
        st.info("No articles yet. Press **Refresh**, widen the time window, change the search, or choose "
                "**Sample** in Settings to explore offline.")
        st.markdown(theme.footer(source_status(s)), unsafe_allow_html=True)
        return
    try:
        out = run_models(rows, s)
    except Exception as exc:
        log.exception("pipeline failed")
        st.error(f"Topic modeling failed: {exc}")
        return
    if not out.results:
        for w in out.warnings:
            st.warning(w)
        return
    model = s["models"][0] if s["models"][0] in out.results else next(iter(out.results))
    clusters = news.build_clusters(out, model)
    st.markdown(theme.ticker(news.ticker_items(clusters)), unsafe_allow_html=True)
    page = news.current_page()
    dark = s["dark"]
    try:
        if page == "Insights":
            render_insights(out, s, store, t_start, len(rows))
        elif page.startswith("topic:"):
            news.render_topic(out, model, clusters, int(page.split(":", 1)[1]), s["country"], dark)
        elif page in news.SECTIONS:
            news.render_section(out, model, clusters, page, s["country"])
        else:
            if query:
                st.markdown(theme.page_head(f"Results for “{query}”"), unsafe_allow_html=True)
            news.render_home(out, model, clusters, s["country"])
    except Exception as exc:
        log.exception("page %s failed", page)
        st.error(f"This page failed to render: {exc}")
    st.markdown(theme.footer(source_status(s)), unsafe_allow_html=True)


def render_insights(out, s: dict, store: ArticleStore, t_start: float, n_rows: int) -> None:
    insights_header(out, s, n_rows)
    dark = s["dark"]
    tabs = st.tabs(["Overview", "Topics", "Trends", "Topic map", "Compare models", "Articles & export",
                    "Evaluate", "System"])
    views = [tab_overview, tab_topics, tab_trends, tab_maps, tab_compare, tab_articles, tab_evaluate]
    for tab, view in zip(tabs, views):
        with tab:
            try:
                view(out, s, dark)
            except Exception as exc:  # a broken chart must not take the page down
                log.exception("view %s failed", view.__name__)
                st.error(f"This view failed to render: {exc}")
    with tabs[-1]:
        lf = st.session_state.get("last_fetch")
        if lf and lf.get("errors"):
            st.markdown("**Source status (last fetch)**")
            for e in lf["errors"]:
                st.caption(f"• {e}")
            st.caption("429 = the publisher rate-limited this server; 403 = the publisher blocks automated "
                       "readers from cloud servers. Other sources keep working.")
        render_ph = tab_system(out, s, dark, store, t_start)
    render_ph.caption(f"Render time: {time.perf_counter() - t_start:.2f}s (fetch + preprocessing + models + charts)")
