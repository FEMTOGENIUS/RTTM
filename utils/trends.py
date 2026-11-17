"""Temporal analysis: topics over time, trending / fading / emerging topics, bursts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def auto_freq(ts: pd.Series) -> str:
    if ts.empty:
        return "D"
    span_h = (ts.max() - ts.min()).total_seconds() / 3600
    if span_h <= 12:
        return "30min"
    if span_h <= 72:
        return "3h"
    if span_h <= 24 * 21:
        return "D"
    return "W"


def topics_over_time(df: pd.DataFrame, assignments, names: dict[int, str],
                     freq: str | None = None) -> pd.DataFrame:
    """Long dataframe: time, topic, name, count, share (of all docs in the bin)."""
    if df.empty:
        return pd.DataFrame(columns=["time", "topic", "name", "count", "share"])
    freq = freq or auto_freq(df["published_at"])
    d = pd.DataFrame({"time": df["published_at"].dt.floor(freq), "topic": np.asarray(assignments)})
    d = d[d["topic"] != -1]
    g = d.groupby(["time", "topic"]).size().rename("count").reset_index()
    full = pd.MultiIndex.from_product([sorted(g["time"].unique()), sorted(g["topic"].unique())],
                                      names=["time", "topic"])
    g = g.set_index(["time", "topic"]).reindex(full, fill_value=0).reset_index()
    tot = g.groupby("time")["count"].transform("sum").replace(0, np.nan)
    g["share"] = (g["count"] / tot).fillna(0.0)
    g["name"] = g["topic"].map(lambda t: names.get(int(t), f"Topic {t}"))
    return g


def trend_table(df: pd.DataFrame, assignments, names: dict[int, str],
                recent_hours: float | None = None) -> pd.DataFrame:
    """Classify every topic as Emerging / Trending / Stable / Fading and detect bursts.

    recent window = last 24h (or last 25% of the time span for short spans).
    growth  = share of recent docs / share of earlier docs
    burst_z = z-score of the topic's count in the latest time bin vs earlier bins
    """
    cols = ["topic", "name", "total", "recent", "earlier", "recent_share", "earlier_share",
            "growth", "burst_z", "status"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    a = np.asarray(assignments)
    ts = df["published_at"]
    end, start = ts.max(), ts.min()
    span_h = max((end - start).total_seconds() / 3600, 1e-6)
    recent_hours = recent_hours or (24 if span_h > 48 else span_h * 0.25)
    cutoff = end - pd.Timedelta(hours=recent_hours)
    is_recent = (ts > cutoff).to_numpy()
    n_recent, n_earlier = max(is_recent.sum(), 1), max((~is_recent).sum(), 1)
    tot = topics_over_time(df, a, names)
    rows = []
    for t in sorted(set(a.tolist()) - {-1}):
        m = a == t
        r, e = int((m & is_recent).sum()), int((m & ~is_recent).sum())
        rs, es = r / n_recent, e / n_earlier
        growth = (rs + 1e-3) / (es + 1e-3)
        series = tot[tot["topic"] == t].sort_values("time")["count"].to_numpy()
        burst = 0.0
        if len(series) >= 4:
            hist, last = series[:-1], series[-1]
            sd = hist.std()
            burst = float((last - hist.mean()) / sd) if sd > 0 else (float(last > hist.mean()) * 3)
        if e == 0 and r >= 3:
            status = "🆕 Emerging"
        elif (r + e) >= 3 and r / (r + e) >= 0.75 and growth >= 2:
            status = "🆕 Emerging"
        elif growth >= 1.5:
            status = "📈 Trending"
        elif growth <= 0.5:
            status = "📉 Fading"
        else:
            status = "➖ Stable"
        rows.append(dict(topic=int(t), name=names.get(int(t), f"Topic {t}"), total=r + e, recent=r,
                         earlier=e, recent_share=round(rs, 4), earlier_share=round(es, 4),
                         growth=round(growth, 2), burst_z=round(burst, 2), status=status))
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["growth", "total"], ascending=False).reset_index(drop=True)


def alerts(trends: pd.DataFrame, burst_threshold: float = 2.0, min_recent: int = 5,
           max_alerts: int = 3) -> list[str]:
    """Only strong signals: emerging topics with >= min_recent new articles or clear
    bursts; at most `max_alerts`, strongest first (avoids alert fatigue)."""
    if trends.empty:
        return []
    cand = []
    for _, r in trends.iterrows():
        if r["status"].startswith("🆕") and r["recent"] >= min_recent:
            cand.append((r["recent"] * 2, f"Emerging topic: {r['name']} - {r['recent']} new articles "
                                           f"in the recent window"))
        elif r["burst_z"] >= burst_threshold and r["recent"] >= min_recent:
            cand.append((r["recent"], f"Burst detected: {r['name']} (z = {r['burst_z']})"))
    return [m for _, m in sorted(cand, key=lambda x: -x[0])[:max_alerts]]
