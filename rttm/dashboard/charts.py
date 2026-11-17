"""Plotly figures for the dashboard (all interactive: zoom, hover, legend toggle)."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Pastel + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel2


def template(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"


def _layout(fig: go.Figure, dark: bool, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(template=template(dark), title=title, height=height,
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font=dict(size=11)))
    return fig


def color_map(names: list[str]) -> dict[str, str]:
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


# --------------------------------------------------------------------------- overview
def word_freq_bar(word_freq: list[tuple[str, int]], dark: bool = False, top: int = 20) -> go.Figure:
    d = pd.DataFrame(word_freq[:top], columns=["word", "count"])
    fig = px.bar(d, x="word", y="count", color_discrete_sequence=["#8ecae6"])
    fig.update_layout(xaxis_tickangle=-45, xaxis_title=None, yaxis_title="Frequency")
    return _layout(fig, dark, "Word frequency distribution")


def topic_pie(sizes: dict[int, int], names: dict[int, str], dark: bool = False) -> go.Figure:
    ids = [t for t in sorted(sizes) if t != -1 and sizes[t] > 0]
    labels = [names.get(t, f"Topic {t}") for t in ids]
    fig = px.pie(names=labels, values=[sizes[t] for t in ids], hole=0.35,
                 color=labels, color_discrete_map=color_map(labels))
    fig.update_traces(textposition="inside", textinfo="percent")
    return _layout(fig, dark, "Topic distribution", height=420)


def topic_size_bar(sizes: dict[int, int], names: dict[int, str], dark: bool = False) -> go.Figure:
    ids = [t for t in sorted(sizes) if t != -1]
    labels = [names.get(t, f"Topic {t}") for t in ids]
    d = pd.DataFrame({"topic": labels, "articles": [sizes[t] for t in ids]})
    fig = px.bar(d, x="articles", y="topic", orientation="h", color="topic",
                 color_discrete_map=color_map(labels))
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, "Articles per topic", height=max(300, 34 * len(ids) + 80))


def sources_bar(df: pd.DataFrame, dark: bool = False, top: int = 15) -> go.Figure:
    d = df["source"].value_counts().head(top).rename_axis("source").reset_index(name="articles")
    fig = px.bar(d, x="articles", y="source", orientation="h", color_discrete_sequence=["#ffb703"])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, "Articles by source", height=max(280, 26 * len(d) + 80))


# --------------------------------------------------------------------------- topic detail
def keywords_bar(words: list[tuple[str, float]], dark: bool = False, n: int = 10,
                 color: str = "#219ebc") -> go.Figure:
    d = pd.DataFrame(words[:n], columns=["keyword", "weight"])
    fig = px.bar(d, x="weight", y="keyword", orientation="h", color_discrete_sequence=[color])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="weight")
    return _layout(fig, dark, None, height=300)


def wordcloud_image(words: list[tuple[str, float]], dark: bool = False):
    """PNG word cloud via the `wordcloud` package; returns None if not installed.
    Cached, because Streamlit re-runs the script on every interaction."""
    return _wordcloud_cached(tuple((w, round(float(s), 6)) for w, s in words[:40]), dark)


@lru_cache(maxsize=256)
def _wordcloud_cached(words: tuple, dark: bool):
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        return None
    freqs = {w: max(float(s), 1e-6) for w, s in words if w}
    if not freqs:
        return None
    wc = WordCloud(width=600, height=300, background_color="#0e1117" if dark else "white",
                   colormap="Pastel1" if dark else "viridis", prefer_horizontal=0.9,
                   random_state=1).generate_from_frequencies(freqs)
    return wc.to_image()


def wordcloud_plotly(words: list[tuple[str, float]], dark: bool = False, n: int = 25) -> go.Figure:
    """Dependency-free word cloud: words on a golden-angle spiral, font size ~ weight."""
    words = [(w, s) for w, s in words[:n] if w]
    fig = go.Figure()
    if not words:
        return _layout(fig, dark, None, 300)
    mx = max(s for _, s in words) or 1
    xs, ys, sizes, texts = [], [], [], []
    for i, (w, s) in enumerate(words):
        r = 0.9 * math.sqrt(i)
        a = i * 2.39996
        xs.append(r * math.cos(a) * 1.6)
        ys.append(r * math.sin(a))
        sizes.append(12 + 30 * (s / mx))
        texts.append(w)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=texts, hoverinfo="text",
                             textfont=dict(size=sizes, color=[PALETTE[i % len(PALETTE)] if dark else
                                                             px.colors.qualitative.Dark24[i % 24]
                                                             for i in range(len(texts))])))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, dark, None, 300)


# --------------------------------------------------------------------------- trends
def timeline(tot: pd.DataFrame, value: str = "count", dark: bool = False,
             stacked: bool = False) -> go.Figure:
    names = list(dict.fromkeys(tot.sort_values("topic")["name"]))
    if stacked:
        fig = px.area(tot, x="time", y=value, color="name", color_discrete_map=color_map(names),
                      category_orders={"name": names})
    else:
        fig = px.line(tot, x="time", y=value, color="name", markers=True,
                      color_discrete_map=color_map(names), category_orders={"name": names})
    fig.update_layout(xaxis_title=None, yaxis_title="articles" if value == "count" else "share of articles",
                      legend=dict(orientation="h", y=-0.25))
    return _layout(fig, dark, "Topic evolution over time", height=480)


def growth_bar(trends: pd.DataFrame, dark: bool = False) -> go.Figure:
    colors = {"🆕 Emerging": "#e63946", "📈 Trending": "#f4a261", "➖ Stable": "#a8dadc",
              "📉 Fading": "#457b9d"}
    d = trends.copy()
    d["log2 growth"] = np.log2(d["growth"].clip(lower=1e-3))
    fig = px.bar(d, x="log2 growth", y="name", color="status", orientation="h",
                 color_discrete_map=colors, hover_data=["recent", "earlier", "burst_z"])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, "Momentum (log2 of recent vs earlier share)",
                   height=max(300, 34 * len(d) + 90))


# --------------------------------------------------------------------------- maps
def intertopic_scatter(m: pd.DataFrame, names: dict[int, str], dark: bool = False,
                       model: str = "") -> go.Figure:
    d = m.copy()
    d["name"] = d["topic"].map(lambda t: names.get(int(t), f"Topic {t}"))
    d["label"] = d["topic"].astype(str)
    fig = px.scatter(d, x="x", y="y", size="size", color="name", text="label", size_max=60,
                     hover_name="name", hover_data={"size": True, "x": False, "y": False, "label": False},
                     color_discrete_map=color_map(list(d["name"])))
    fig.update_traces(textposition="middle center", marker=dict(opacity=0.75, line=dict(width=1)))
    fig.update_layout(showlegend=False, xaxis_title="PC1", yaxis_title="PC2")
    return _layout(fig, dark, f"Intertopic distance map ({model})", height=500)


def doc_map(coords: np.ndarray, assignments: np.ndarray, titles: list[str], names: dict[int, str],
            dark: bool = False, model: str = "") -> go.Figure:
    d = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "title": titles,
                      "topic": [names.get(int(t), "Outlier") for t in assignments]})
    fig = px.scatter(d, x="x", y="y", color="topic", hover_name="title",
                     hover_data={"x": False, "y": False}, color_discrete_map=color_map(sorted(set(d["topic"]))))
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(legend=dict(orientation="h", y=-0.2), xaxis_title=None, yaxis_title=None)
    return _layout(fig, dark, f"Document clusters ({model})", height=560)


# --------------------------------------------------------------------------- comparison
def metrics_bar(metrics: dict[str, dict], dark: bool = False) -> go.Figure:
    rows = []
    for model, m in metrics.items():
        for key, label in [("coherence_cv", "Coherence C_V"), ("coherence_npmi", "Coherence NPMI"),
                           ("diversity", "Topic diversity")]:
            v = m.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                rows.append({"model": model, "metric": label, "value": v})
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="metric", y="value", color="model", barmode="group", text_auto=".2f",
                 color_discrete_map={"LDA": "#8ecae6", "BERTopic": "#ffb703"})
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _layout(fig, dark, "Topic quality (higher is better)")


def latency_bar(timings: dict[str, float], dark: bool = False, title: str = "Pipeline latency") -> go.Figure:
    d = pd.DataFrame([(k, v) for k, v in timings.items() if k != "total"], columns=["stage", "seconds"])
    fig = px.bar(d, x="seconds", y="stage", orientation="h", text_auto=".2f",
                 color_discrete_sequence=["#90be6d"])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, title, height=max(260, 34 * len(d) + 90))


def jaccard_heatmap(J: np.ndarray, lda_names: list[str], bt_names: list[str], dark: bool = False) -> go.Figure:
    fig = px.imshow(J, x=bt_names, y=lda_names, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Blues", zmin=0, zmax=1,
                    labels=dict(x="BERTopic topic", y="LDA topic", color="Jaccard"))
    return _layout(fig, dark, "Topic alignment: top-word overlap (LDA vs BERTopic)",
                   height=max(380, 30 * len(lda_names) + 160))
