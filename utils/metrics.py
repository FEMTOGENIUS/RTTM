"""Topic-model evaluation metrics.

* Coherence C_V   (Röder et al., 2015)  - primary metric promised in the paper/report
* Coherence NPMI  (Bouma 2009 / Aletras & Stevenson 2013)
* Coherence UMass (Mimno et al., 2011)
* Topic diversity (Dieng et al., 2020): % unique words in the top-25 words of all topics
* Timer helper for latency measurements

If gensim is installed its `CoherenceModel` is used for C_V (identical
definition); otherwise the built-in implementation below is used.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from contextlib import contextmanager
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from utils.logging_setup import get_logger

log = get_logger("metrics")
EPS = 1e-12


# --------------------------------------------------------------------------- helpers
def unigram_topics(topics: Sequence[Sequence[str]], top_n: int = 10,
                   vocab: set[str] | None = None) -> list[list[str]]:
    """Split n-gram keywords into unigrams, de-duplicate, keep order, keep top_n.
    Words unknown to the reference corpus are dropped (they carry no evidence)."""
    out = []
    for words in topics:
        seen: list[str] = []
        for w in words:
            for u in str(w).split():
                if u not in seen and (vocab is None or u in vocab):
                    seen.append(u)
        out.append(seen[:top_n])
    return out


def _window_counts(texts: Sequence[Sequence[str]], relevant: set[str], window: int):
    """Boolean sliding-window document frequencies for words & pairs."""
    w_cnt: Counter = Counter()
    p_cnt: Counter = Counter()
    n_windows = 0
    for toks in texts:
        if not toks:
            continue
        if len(toks) <= window:
            spans = [toks]
        else:
            spans = (toks[i:i + window] for i in range(len(toks) - window + 1))
        for span in spans:
            n_windows += 1
            present = sorted(relevant.intersection(span))
            w_cnt.update(present)
            p_cnt.update(combinations(present, 2))
    return w_cnt, p_cnt, max(n_windows, 1)


def _npmi(a: str, b: str, w_cnt, p_cnt, n) -> float:
    if a == b:
        return 1.0 if w_cnt[a] else 0.0
    pa, pb = w_cnt[a] / n, w_cnt[b] / n
    if pa == 0 or pb == 0:
        return 0.0
    pab = p_cnt[(a, b) if a < b else (b, a)] / n
    pmi = math.log((pab + EPS) / (pa * pb))
    return pmi / -math.log(pab + EPS)


# --------------------------------------------------------------------------- coherence
def coherence_cv(topics: Sequence[Sequence[str]], texts: Sequence[Sequence[str]],
                 top_n: int = 10, window: int = 110, use_gensim: bool = True) -> tuple[float, list[float]]:
    vocab = {w for t in texts for w in t}
    topics = [t for t in unigram_topics(topics, top_n, vocab) if len(t) >= 2]
    if not topics:
        return float("nan"), []
    if use_gensim:
        try:
            from gensim.corpora import Dictionary  # type: ignore
            from gensim.models.coherencemodel import CoherenceModel  # type: ignore

            d = Dictionary(texts)
            cm = CoherenceModel(topics=topics, texts=[list(t) for t in texts], dictionary=d,
                                coherence="c_v", topn=top_n, processes=1)
            per = [float(x) for x in cm.get_coherence_per_topic()]
            return float(np.nanmean(per)), per
        except ImportError:
            pass
        except Exception as exc:  # fall back to built-in implementation
            log.warning("gensim c_v failed (%s) - using built-in implementation", exc)
    relevant = {w for t in topics for w in t}
    w_cnt, p_cnt, n = _window_counts(texts, relevant, window)
    per = []
    for t in topics:
        vecs = np.array([[_npmi(wi, wj, w_cnt, p_cnt, n) for wj in t] for wi in t])
        total = vecs.sum(axis=0)
        sims = []
        for v in vecs:
            denom = np.linalg.norm(v) * np.linalg.norm(total)
            sims.append(float(v @ total / denom) if denom > 0 else 0.0)
        per.append(float(np.mean(sims)))
    return float(np.mean(per)), per


def coherence_npmi(topics: Sequence[Sequence[str]], texts: Sequence[Sequence[str]],
                   top_n: int = 10, window: int = 10) -> tuple[float, list[float]]:
    vocab = {w for t in texts for w in t}
    topics = [t for t in unigram_topics(topics, top_n, vocab) if len(t) >= 2]
    if not topics:
        return float("nan"), []
    relevant = {w for t in topics for w in t}
    w_cnt, p_cnt, n = _window_counts(texts, relevant, window)
    per = [float(np.mean([_npmi(a, b, w_cnt, p_cnt, n) for a, b in combinations(t, 2)]))
           for t in topics]
    return float(np.mean(per)), per


def coherence_umass(topics: Sequence[Sequence[str]], texts: Sequence[Sequence[str]],
                    top_n: int = 10) -> tuple[float, list[float]]:
    vocab = {w for t in texts for w in t}
    topics = [t for t in unigram_topics(topics, top_n, vocab) if len(t) >= 2]
    if not topics:
        return float("nan"), []
    docsets = [set(t) for t in texts]
    relevant = {w for t in topics for w in t}
    df: Counter = Counter()
    co: Counter = Counter()
    for s in docsets:
        present = sorted(relevant & s)
        df.update(present)
        co.update(combinations(present, 2))
    per = []
    for t in topics:
        scores = []
        for i in range(1, len(t)):
            for j in range(i):
                wi, wj = t[i], t[j]
                pair = (wi, wj) if wi < wj else (wj, wi)
                if df[wj]:
                    scores.append(math.log((co[pair] + 1) / df[wj]))
        per.append(float(np.mean(scores)) if scores else 0.0)
    return float(np.mean(per)), per


def topic_diversity(topics: Sequence[Sequence[str]], top_k: int = 25) -> float:
    """Proportion of unique words among the top_k words of every topic (0..1)."""
    topics = [list(t)[:top_k] for t in topics if t]
    if not topics:
        return float("nan")
    total = sum(len(t) for t in topics)
    uniq = len({w for t in topics for w in t})
    return uniq / total if total else float("nan")


def evaluate_topics(topic_words: dict[int, list[str]], texts: Sequence[Sequence[str]],
                    top_n: int = 10) -> dict:
    words = [w for _, w in sorted(topic_words.items()) if w]
    cv, cv_per = coherence_cv(words, texts, top_n)
    npmi, _ = coherence_npmi(words, texts, top_n)
    umass, _ = coherence_umass(words, texts, top_n)
    return {
        "coherence_cv": cv,
        "coherence_cv_per_topic": dict(zip(sorted(k for k, w in topic_words.items() if w), cv_per)),
        "coherence_npmi": npmi,
        "coherence_umass": umass,
        "diversity": topic_diversity(words, 25),
        "diversity_top10": topic_diversity(words, 10),
        "n_topics": len(words),
    }


# --------------------------------------------------------------------------- timing
class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0

    @property
    def total(self) -> float:
        return sum(self.stages.values())


def word_frequencies(token_lists: Iterable[Sequence[str]], top: int = 25) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in token_lists:
        c.update(t)
    return c.most_common(top)
