"""Plotly figures for the dashboard (all interactive: zoom, hover, legend toggle)."""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.theme import CATEGORY_COLORS, PALETTE, topic_color

FONT = "Inter, Segoe UI, system-ui, sans-serif"
SERIF = "Source Serif 4, Georgia, serif"


def template(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"


def _layout(fig: go.Figure, dark: bool, title: str | None = None, height: int = 380) -> go.Figure:
    txt = "#E2E8F0" if dark else "#0F172A"
    grid = "rgba(148,163,184,.15)" if dark else "rgba(15,23,42,.06)"
    fig.update_layout(template=template(dark), height=height, colorway=PALETTE,
                      title=dict(text=title or None, x=0.01, font=dict(size=17, color=txt, family=SERIF)),
                      font=dict(family=FONT, color=txt, size=12),
                      margin=dict(l=12, r=12, t=52 if title else 16, b=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font=dict(family=FONT, size=12), bordercolor="rgba(0,0,0,0)"),
                      legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    return fig


def _tid(name: str) -> int | None:
    m = re.search(r"Topic (\d+)", str(name))
    return int(m.group(1)) if m else None


def color_map(names: list[str]) -> dict[str, str]:
    out = {}
    for i, n in enumerate(names):
        t = _tid(n)
        out[n] = topic_color(t) if t is not None else PALETTE[i % len(PALETTE)]
    return out


# --------------------------------------------------------------------------- overview
def word_freq_bar(word_freq: list[tuple[str, int]], dark: bool = False, top: int = 20) -> go.Figure:
    d = pd.DataFrame(word_freq[:top], columns=["word", "count"])
    d = d.iloc[::-1]
    fig = px.bar(d, x="count", y="word", orientation="h", color="count",
                 color_continuous_scale=["#FECACA", "#C8102E", "#7F1D1D"])
    fig.update_traces(marker_line_width=0, hovertemplate="<b>%{y}</b><br>%{x} mentions<extra></extra>")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="mentions", yaxis_title=None)
    return _layout(fig, dark, "Hot keywords", height=520)


def topic_pie(sizes: dict[int, int], names: dict[int, str], dark: bool = False) -> go.Figure:
    ids = [t for t in sorted(sizes) if t != -1 and sizes[t] > 0]
    labels = [names.get(t, f"Topic {t}") for t in ids]
    fig = px.pie(names=labels, values=[sizes[t] for t in ids], hole=0.35,
                 color=labels, color_discrete_map=color_map(labels))
    fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="rgba(0,0,0,0)")),
                      hovertemplate="<b>%{label}</b><br>%{value} articles · %{percent}<extra></extra>", pull=0.02)
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    return _layout(fig, dark, "Topic distribution", height=520)


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
    fig = px.bar(d, x="articles", y="source", orientation="h", color_discrete_sequence=["#334155"])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, "Articles by source", height=max(280, 26 * len(d) + 80))


# --------------------------------------------------------------------------- topic detail
def keywords_bar(words: list[tuple[str, float]], dark: bool = False, n: int = 10,
                 color: str = "#C8102E") -> go.Figure:
    d = pd.DataFrame(words[:n], columns=["keyword", "weight"])
    fig = px.bar(d, x="weight", y="keyword", orientation="h", color_discrete_sequence=[color])
    fig.update_traces(marker_line_width=0, hovertemplate="<b>%{y}</b><br>weight %{x:.4f}<extra></extra>")
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="importance")
    return _layout(fig, dark, "Top keywords", height=340)


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
    wc = WordCloud(width=700, height=340, background_color=None, mode="RGBA",
                   colormap="cool" if dark else "plasma", prefer_horizontal=0.9,
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
                             textfont=dict(size=sizes, color=[PALETTE[i % len(PALETTE)] for i in range(len(texts))])))
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
    colors = {"🆕 Emerging": "#F43F5E", "📈 Trending": "#F59E0B", "➖ Stable": "#94A3B8",
              "📉 Fading": "#0EA5E9"}
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
                 color_discrete_map={"LDA": "#64748B", "BERTopic": "#C8102E"})
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _layout(fig, dark, "Topic quality (higher is better)")


def latency_bar(timings: dict[str, float], dark: bool = False, title: str = "Pipeline latency") -> go.Figure:
    d = pd.DataFrame([(k, v) for k, v in timings.items() if k != "total"], columns=["stage", "seconds"])
    fig = px.bar(d, x="seconds", y="stage", orientation="h", text_auto=".2f",
                 color_discrete_sequence=["#10B981"])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    return _layout(fig, dark, title, height=max(260, 34 * len(d) + 90))


def jaccard_heatmap(J: np.ndarray, lda_names: list[str], bt_names: list[str], dark: bool = False) -> go.Figure:
    fig = px.imshow(J, x=bt_names, y=lda_names, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#FFF1F2", "#FB7185", "#9F1239"], zmin=0, zmax=1,
                    labels=dict(x="BERTopic topic", y="LDA topic", color="Jaccard"))
    return _layout(fig, dark, "Topic alignment: top-word overlap (LDA vs BERTopic)",
                   height=max(380, 30 * len(lda_names) + 160))


# --------------------------------------------------------------------------- new interactive views
def category_treemap(docs: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Click a category to zoom into its topics (click the header to zoom out)."""
    d = docs[docs["topic"] != -1].groupby(["category", "topic_name", "keywords"]).size().reset_index(name="articles")
    d["label"] = d["topic_name"].str.replace(r" \(.*\)", "", regex=True) + ": " + d["keywords"]
    cmap = {c: CATEGORY_COLORS.get(c, "#64748B") for c in d["category"].unique()}
    cmap["(?)"] = "#6366F1"
    fig = px.treemap(d, path=[px.Constant("All news"), "category", "label"], values="articles",
                     color="category", color_discrete_map=cmap)
    fig.update_traces(root_color="rgba(0,0,0,0)", marker=dict(cornerradius=6),
                      hovertemplate="<b>%{label}</b><br>%{value} articles<extra></extra>",
                      textinfo="label+value")
    return _layout(fig, dark, "News landscape - click a category to drill down", height=520)


def topic_race(tot: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Animated bar race of cumulative articles per topic - press ▶ Play."""
    d = tot.sort_values("time").copy()
    d["cumulative"] = d.groupby("topic")["count"].cumsum()
    d["frame"] = d["time"].dt.strftime("%d %b %H:%M")
    names = list(dict.fromkeys(d.sort_values("topic")["name"]))
    short = {n: (n[:48] + "…") if len(n) > 48 else n for n in names}
    d["topic_label"] = d["name"].map(short)
    fig = px.bar(d, x="cumulative", y="topic_label", color="name", orientation="h", animation_frame="frame",
                 range_x=[0, max(1, d["cumulative"].max()) * 1.1], color_discrete_map=color_map(names),
                 category_orders={"topic_label": [short[n] for n in names]})
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="articles so far")
    return _layout(fig, dark, "Topic race - press ▶ to replay how the news unfolded",
                   height=max(380, 30 * len(names) + 160))
