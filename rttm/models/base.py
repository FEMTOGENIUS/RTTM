"""Common result container shared by all topic models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopicResult:
    model: str                                   # "LDA" | "BERTopic"
    backend: str                                 # library actually used
    topic_words: dict[int, list[tuple[str, float]]]  # topic id -> [(word, weight)] (desc)
    assignments: np.ndarray                      # dominant topic per document (-1 = outlier)
    confidence: np.ndarray                       # probability / similarity of dominant topic
    doc_topic: np.ndarray | None = None          # (n_docs, n_topics) distribution (LDA)
    topic_vectors: dict[int, np.ndarray] = field(default_factory=dict)  # for intertopic map
    doc_coords: np.ndarray | None = None         # (n_docs, 2) projection for cluster map
    raw_outliers: int = 0                        # HDBSCAN outliers before reassignment
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)    # model objects (e.g. for pyLDAvis)

    @property
    def topic_ids(self) -> list[int]:
        return sorted(k for k in self.topic_words if k != -1)

    def top_words(self, topic: int, n: int = 10) -> list[str]:
        return [w for w, _ in self.topic_words.get(topic, [])[:n]]

    def sizes(self) -> dict[int, int]:
        ids, counts = np.unique(self.assignments, return_counts=True)
        out = {int(i): int(c) for i, c in zip(ids, counts)}
        return {t: out.get(t, 0) for t in self.topic_ids} | ({-1: out[-1]} if -1 in out else {})


def renumber_by_size(assign: np.ndarray, *maps: dict) -> tuple[np.ndarray, list[dict], dict[int, int]]:
    """Renumber topics 0..K-1 by descending size (BERTopic convention); -1 kept."""
    ids, counts = np.unique(assign[assign >= 0], return_counts=True)
    order = [int(i) for i, _ in sorted(zip(ids, counts), key=lambda x: (-x[1], x[0]))]
    mapping = {old: new for new, old in enumerate(order)}
    mapping[-1] = -1
    new_assign = np.array([mapping.get(int(a), -1) for a in assign])
    new_maps = [{mapping[k]: v for k, v in m.items() if k in mapping} for m in maps]
    return new_assign, new_maps, mapping
