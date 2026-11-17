"""End-to-end pipeline: preprocess -> topic models -> metrics -> labels -> trends.

`run_pipeline()` is used by the dashboard, the FastAPI service, the
scheduler and `evaluate.py`, so every interface produces identical results.

Graceful degradation: if BERTopic fails (missing libraries, too few documents,
all-outlier clustering, ...) the system falls back to LDA and reports why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config
from models.base import TopicResult
from models.bertopic_model import BERTopicModel
from models.lda_model import LDATopicModel
from utils import trends as tr
from utils.labels import topic_label
from utils.logging_setup import get_logger
from utils.metrics import Timer, evaluate_topics, word_frequencies
from utils.preprocess import preprocess_articles

log = get_logger("engine")
MODELS = ("BERTopic", "LDA")


@dataclass
class PipelineOutput:
    df: pd.DataFrame
    prep_stats: dict
    results: dict[str, TopicResult] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    labels: dict[str, dict[int, dict]] = field(default_factory=dict)
    trends: dict[str, pd.DataFrame] = field(default_factory=dict)
    alerts: dict[str, list[str]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    word_freq: list[tuple[str, int]] = field(default_factory=list)
    comparison: dict | None = None

    def names(self, model: str) -> dict[int, str]:
        return {t: f"{d['name']}: {d['keywords']}" for t, d in self.labels.get(model, {}).items()}

    def doc_frame(self, model: str) -> pd.DataFrame:
        """Articles + their dominant topic for a model."""
        r = self.results[model]
        lab = self.labels[model]
        d = self.df.drop(columns=["tokens"], errors="ignore").copy()
        d["topic"] = r.assignments
        d["topic_name"] = [lab.get(int(t), {}).get("name", "Outlier") for t in r.assignments]
        d["category"] = [lab.get(int(t), {}).get("category", "Outlier") for t in r.assignments]
        d["keywords"] = [lab.get(int(t), {}).get("keywords", "") for t in r.assignments]
        d["confidence"] = np.round(r.confidence, 3)
        return d


def run_pipeline(articles: Iterable[dict], *, models: Iterable[str] = ("BERTopic",),
                 n_topics: int = config.DEFAULT_N_TOPICS, bertopic_nr_topics: int | str | None = "auto",
                 min_topic_size: int | None = None, lda_passes: int = 10, lda_weighting: str = "bow",
                 embedding_model: str = config.EMBEDDING_MODEL, reduce_outliers: bool = True,
                 english_only: bool = True, bertopic_backend: str = "auto",
                 lda_backend: str = "auto", random_state: int = config.RANDOM_STATE,
                 language: str = "en") -> PipelineOutput:
    timer = Timer()
    models = [m for m in MODELS if m in set(models)] or ["BERTopic"]
    with timer.stage("preprocess"):
        df, stats = preprocess_articles(articles, english_only=english_only, language=language)
    if language != "en" and embedding_model == config.EMBEDDING_MODEL:
        embedding_model = config.MULTILINGUAL_EMBEDDING_MODEL
    out = PipelineOutput(df=df, prep_stats=stats)
    if len(df) < 3:
        out.warnings.append(f"Only {len(df)} usable articles after preprocessing - "
                            "fetch more news or widen the time window.")
        out.timings = dict(timer.stages)
        return out
    tokens, clean, raw = df["tokens"].tolist(), df["clean"].tolist(), df["text"].tolist()
    with timer.stage("word_frequency"):
        out.word_freq = word_frequencies(tokens, 30)

    if "BERTopic" in models:
        try:
            with timer.stage("bertopic"):
                bt = BERTopicModel(nr_topics=bertopic_nr_topics, min_topic_size=min_topic_size,
                                   embedding_model=embedding_model, reduce_outliers=reduce_outliers,
                                   backend=bertopic_backend, random_state=random_state)
                out.results["BERTopic"] = bt.fit(clean, raw)
        except Exception as exc:
            msg = f"BERTopic failed ({exc}); falling back to LDA."
            log.error(msg)
            out.warnings.append(msg)
            if "LDA" not in models:
                models.append("LDA")
    if "LDA" in models:
        try:
            with timer.stage("lda"):
                out.results["LDA"] = LDATopicModel(n_topics=n_topics, passes=lda_passes,
                                                   weighting=lda_weighting, backend=lda_backend,
                                                   random_state=random_state).fit(tokens)
        except Exception as exc:
            msg = f"LDA failed: {exc}"
            log.error(msg)
            out.warnings.append(msg)

    with timer.stage("metrics_labels_trends"):
        for name, res in out.results.items():
            tw = {t: [w for w, _ in res.topic_words[t]] for t in res.topic_ids}
            m = evaluate_topics(tw, tokens)
            m["outlier_rate_raw"] = res.raw_outliers / len(df)
            m["runtime_s"] = sum(res.timings.values())
            out.metrics[name] = m
            out.labels[name] = {t: topic_label(t, tw[t]) for t in res.topic_ids}
            _apply_section_hints(out.labels[name], df, res.assignments)
            if res.doc_coords is None:
                res.doc_coords = _doc_coords(res)
            names = out.names(name)
            out.trends[name] = tr.trend_table(df, res.assignments, names)
            out.alerts[name] = tr.alerts(out.trends[name])
        if len(out.results) == 2:
            out.comparison = compare_models(out.results["LDA"], out.results["BERTopic"])
    out.timings = dict(timer.stages)
    out.timings["total"] = timer.total
    for w in out.warnings:
        log.warning(w)
    log.info("pipeline done: %d docs, models=%s, %.2fs", len(df), list(out.results), timer.total)
    return out


# --------------------------------------------------------------------------- helpers
def _apply_section_hints(labels: dict[int, dict], df: pd.DataFrame, assignments) -> None:
    """Articles from section feeds (e.g. 'BBC Sport', 'Google News Business') carry a section hint.
    If most hinted articles of a topic share one section, that section becomes the topic's category.
    This makes labels correct in every language, where the English keyword dictionary can't help."""
    if "section" not in df:
        return
    sec = df["section"].fillna("").to_numpy()
    a = np.asarray(assignments)
    for t, lab in labels.items():
        hints = [h for h in sec[a == t] if h]
        n_docs = int((a == t).sum())
        if not hints or len(hints) < max(2, 0.3 * n_docs):
            continue
        top, cnt = pd.Series(hints).value_counts().index[0], pd.Series(hints).value_counts().iloc[0]
        if cnt / len(hints) >= 0.5 and top != lab["category"]:
            lab["keyword_category"] = lab["category"]
            lab["category"] = top
            lab["name"] = f"Topic {t} ({top})"
            from utils.labels import ICONS

            lab["icon"] = ICONS.get(top, "📰")



def _doc_coords(res: TopicResult) -> np.ndarray | None:
    if res.doc_topic is None or res.doc_topic.shape[1] < 2:
        return None
    return PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(np.sqrt(res.doc_topic))


def intertopic_map(res: TopicResult) -> pd.DataFrame:
    """2-D intertopic distance map (like pyLDAvis): Jensen-Shannon distance between
    topic-word distributions (LDA) or cosine distance between topic centroids
    (BERTopic), embedded with classical MDS."""
    ids = [t for t in res.topic_ids if t in res.topic_vectors]
    sizes = res.sizes()
    if len(ids) < 2:
        return pd.DataFrame({"topic": ids, "x": [0.0] * len(ids), "y": [0.0] * len(ids),
                             "size": [sizes.get(t, 0) for t in ids]})
    V = np.vstack([res.topic_vectors[t] for t in ids]).astype(float)
    if res.model == "LDA":
        P = V / V.sum(axis=1, keepdims=True)
        D = np.array([[jensenshannon(P[i], P[j]) for j in range(len(ids))] for i in range(len(ids))])
    else:
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        D = np.clip(1 - Vn @ Vn.T, 0, 2)
    D = np.nan_to_num((D + D.T) / 2)
    # classical MDS (principal coordinates analysis), as used by LDAvis
    n = len(ids)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:2]
    coords = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))
    return pd.DataFrame({"topic": ids, "x": coords[:, 0], "y": coords[:, 1],
                         "size": [sizes.get(t, 0) for t in ids]})


def compare_models(lda: TopicResult, bt: TopicResult, top_n: int = 10) -> dict:
    """Topic alignment (Jaccard of top words) + document-assignment agreement."""
    li, bi = lda.topic_ids, bt.topic_ids
    J = np.zeros((len(li), len(bi)))
    for i, a in enumerate(li):
        sa = {u for w in lda.top_words(a, top_n) for u in w.split()}
        for j, b in enumerate(bi):
            sb = {u for w in bt.top_words(b, top_n) for u in w.split()}
            J[i, j] = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    mask = (lda.assignments != -1) & (bt.assignments != -1)
    ari = adjusted_rand_score(lda.assignments[mask], bt.assignments[mask]) if mask.sum() > 1 else float("nan")
    nmi = normalized_mutual_info_score(lda.assignments[mask], bt.assignments[mask]) if mask.sum() > 1 else float("nan")
    best = [{"lda_topic": a, "bertopic_topic": bi[int(J[i].argmax())], "jaccard": round(float(J[i].max()), 3)}
            for i, a in enumerate(li)] if len(bi) else []
    return {"jaccard": J, "lda_ids": li, "bertopic_ids": bi, "ari": float(ari), "nmi": float(nmi),
            "best_matches": best}


def search_topics(out: PipelineOutput, model: str, query: str) -> list[int]:
    """Topics whose keywords or articles contain any query term."""
    terms = [t.lower() for t in query.split() if t.strip()]
    if not terms or model not in out.results:
        return out.results[model].topic_ids if model in out.results else []
    res = out.results[model]
    hits = []
    texts = (out.df["title"].fillna("") + " " + out.df["clean"].fillna("")).str.lower()
    for t in res.topic_ids:
        kw = " ".join(res.top_words(t, 25)).lower()
        docs = texts[res.assignments == t]
        if any(term in kw for term in terms) or any(docs.str.contains(term, regex=False).any() for term in terms):
            hits.append(t)
    return hits
