"""BERTopic (Grootendorst, 2022): embeddings -> UMAP -> HDBSCAN -> c-TF-IDF.

Back-ends (chosen automatically, reported in the UI):

1. "bertopic"  - the real `bertopic` library with SentenceTransformer
                 `all-MiniLM-L6-v2` embeddings, UMAP and HDBSCAN. (Paper setup.)
2. "lite"      - same four-stage pipeline re-implemented with whatever is
                 installed: SentenceTransformer embeddings if available,
                 otherwise LSA (TF-IDF + truncated SVD) embeddings; UMAP if
                 available, otherwise PCA; HDBSCAN from `hdbscan` or
                 scikit-learn >= 1.3; our own c-TF-IDF (models/ctfidf.py).

Real-time optimisation: sentence embeddings are cached per article, so on each
refresh only *new* articles are embedded.
"""
from __future__ import annotations

import hashlib
import time
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import normalize

import config
from models import ctfidf
from models.base import TopicResult, renumber_by_size
from utils.logging_setup import get_logger

log = get_logger("bertopic")


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAS_BERTOPIC = _has("bertopic")
HAS_ST = _has("sentence_transformers")
HAS_UMAP = _has("umap")
HAS_HDBSCAN_LIB = _has("hdbscan")


def backend_summary() -> dict[str, str]:
    return {
        "bertopic": "installed" if HAS_BERTOPIC else "missing -> lite pipeline",
        "embeddings": f"SentenceTransformer ({config.EMBEDDING_MODEL})" if HAS_ST else "LSA (TF-IDF + SVD) fallback",
        "dim_reduction": "UMAP" if HAS_UMAP else "PCA fallback",
        "clustering": "HDBSCAN (hdbscan)" if HAS_HDBSCAN_LIB else "HDBSCAN (scikit-learn)",
    }


def make_hdbscan(min_cluster_size: int, min_samples: int, prediction_data: bool = False):
    if HAS_HDBSCAN_LIB:
        from hdbscan import HDBSCAN  # type: ignore

        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                       cluster_selection_method="eom", prediction_data=prediction_data)
    from sklearn.cluster import HDBSCAN  # type: ignore

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                   cluster_selection_method="eom", copy=True)


# --------------------------------------------------------------------------- embeddings
@lru_cache(maxsize=2)
def load_sentence_model(name: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer  # type: ignore

    log.info("loading sentence-transformer %s", name)
    return SentenceTransformer(name)


_EMB_CACHE: dict[str, np.ndarray] = {}


def embed_texts(texts: list[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Transformer embeddings with a per-text cache (only new texts are encoded)."""
    model = load_sentence_model(model_name)
    keys = [hashlib.sha1(f"{model_name}|{t}".encode()).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in _EMB_CACHE]
    if missing:
        vecs = model.encode([texts[i] for i in missing], show_progress_bar=False,
                            batch_size=64, normalize_embeddings=True)
        for i, v in zip(missing, vecs):
            _EMB_CACHE[keys[i]] = np.asarray(v, dtype=np.float32)
    log.info("embeddings: %d cached, %d new", len(texts) - len(missing), len(missing))
    return np.vstack([_EMB_CACHE[k] for k in keys])


def lsa_embeddings(clean_docs: list[str], dims: int = 100) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(clean_docs)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if n >= 40 else 1, max_df=0.9,
                          sublinear_tf=True)
    try:
        X = vec.fit_transform(clean_docs)
    except ValueError:
        X = TfidfVectorizer().fit_transform(clean_docs)
    k = max(2, min(dims, X.shape[1] - 1, n - 1))
    Z = TruncatedSVD(n_components=k, random_state=config.RANDOM_STATE).fit_transform(X)
    return normalize(Z)


# --------------------------------------------------------------------------- model
class BERTopicModel:
    def __init__(self, nr_topics: int | str | None = "auto", min_topic_size: int | None = None,
                 embedding_model: str = config.EMBEDDING_MODEL, reduce_outliers: bool = True,
                 backend: str = "auto", top_n_words: int = 25,
                 random_state: int = config.RANDOM_STATE):
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.embedding_model = embedding_model
        self.reduce_outliers = reduce_outliers
        self.backend = backend
        self.top_n_words = top_n_words
        self.random_state = random_state
        self.model_ = None

    def _params(self, n: int) -> dict:
        mcs = self.min_topic_size or max(3, min(15, n // 30))
        return dict(n_neighbors=max(2, min(15, n - 1)), n_components=max(2, min(5, n - 2)),
                    min_cluster_size=int(mcs), min_samples=max(1, int(mcs) // 2))

    def fit(self, clean_docs: list[str], raw_texts: list[str]) -> TopicResult:
        n = len(clean_docs)
        if n < 10:
            raise ValueError(f"BERTopic needs at least 10 documents (got {n})")
        use_full = self.backend in ("auto", "bertopic") and HAS_BERTOPIC and HAS_ST
        if self.backend == "bertopic" and not use_full:
            raise RuntimeError("bertopic / sentence-transformers not installed")
        res = self._fit_full(clean_docs, raw_texts) if use_full else self._fit_lite(clean_docs, raw_texts)
        if not res.topic_ids:
            raise RuntimeError("BERTopic found no clusters (all documents are outliers)")
        return res

    # ---------------------------------------------------------------- full library
    def _fit_full(self, clean_docs, raw_texts) -> TopicResult:
        from bertopic import BERTopic  # type: ignore
        from bertopic.vectorizers import ClassTfidfTransformer  # type: ignore
        from umap import UMAP  # type: ignore

        n = len(clean_docs)
        p = self._params(n)
        timings = {}
        t0 = time.perf_counter()
        emb = embed_texts(raw_texts, self.embedding_model)
        timings["embeddings"] = time.perf_counter() - t0
        umap_model = UMAP(n_neighbors=p["n_neighbors"], n_components=p["n_components"], min_dist=0.0,
                          metric="cosine", random_state=self.random_state)
        hdbscan_model = make_hdbscan(p["min_cluster_size"], p["min_samples"], prediction_data=True)
        vectorizer = ctfidf.make_vectorizer(n)   # docs are already stop-word free & lemmatised
        nr = None if self.nr_topics in (None, "auto") else int(self.nr_topics)
        model = BERTopic(embedding_model=load_sentence_model(self.embedding_model), umap_model=umap_model,
                         hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                         ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                         nr_topics=nr, top_n_words=self.top_n_words, calculate_probabilities=False,
                         verbose=False)
        t0 = time.perf_counter()
        topics, _ = model.fit_transform(clean_docs, embeddings=emb)
        topics = np.asarray(topics)
        raw_out = int((topics == -1).sum())
        notes = []
        if self.reduce_outliers and raw_out and (topics != -1).any():
            try:
                new = model.reduce_outliers(clean_docs, topics.tolist(), strategy="embeddings",
                                            embeddings=emb)
                model.update_topics(clean_docs, topics=new, vectorizer_model=vectorizer,
                                    top_n_words=self.top_n_words)
                topics = np.asarray(new)
                notes.append(f"{raw_out} HDBSCAN outliers re-assigned to nearest topic (embeddings strategy)")
            except Exception as exc:
                notes.append(f"outlier reduction skipped: {exc}")
        timings["umap_hdbscan_ctfidf"] = time.perf_counter() - t0
        self.model_ = model
        words = {int(t): [(w, float(s)) for w, s in (model.get_topic(int(t)) or []) if w]
                 for t in set(topics.tolist()) if t != -1}
        reduced = getattr(model.umap_model, "embedding_", None)
        return self._finish(topics, words, emb, reduced, raw_out, timings, notes,
                            backend=f"BERTopic + SentenceTransformer ({self.embedding_model}) + UMAP + HDBSCAN",
                            params=p)

    # ---------------------------------------------------------------- lite pipeline
    def _fit_lite(self, clean_docs, raw_texts) -> TopicResult:
        n = len(clean_docs)
        p = self._params(n)
        timings, notes = {}, []
        t0 = time.perf_counter()
        if HAS_ST:
            emb = embed_texts(raw_texts, self.embedding_model)
            emb_name = f"SentenceTransformer ({self.embedding_model})"
        else:
            emb = lsa_embeddings(clean_docs)
            emb_name = "LSA embeddings (TF-IDF+SVD)"
        timings["embeddings"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if HAS_UMAP:
            from umap import UMAP  # type: ignore

            reduced = UMAP(n_neighbors=p["n_neighbors"], n_components=p["n_components"], min_dist=0.0,
                           metric="cosine", random_state=self.random_state).fit_transform(emb)
            red_name = "UMAP"
        else:
            from sklearn.decomposition import PCA

            k = max(2, min(10, emb.shape[1] - 1, n - 2))
            reduced = normalize(PCA(n_components=k, random_state=self.random_state).fit_transform(emb))
            red_name = "PCA"
        timings["dim_reduction"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        labels = make_hdbscan(p["min_cluster_size"], p["min_samples"]).fit_predict(reduced)
        labels = np.asarray(labels)
        raw_out = int((labels == -1).sum())
        nr = None if self.nr_topics in (None, "auto") else int(self.nr_topics)
        if nr and len(set(labels.tolist()) - {-1}) > nr:
            labels = ctfidf.merge_to_n_topics(clean_docs, labels, nr)
            notes.append(f"topics merged to nr_topics={nr} via c-TF-IDF similarity")
        if self.reduce_outliers and raw_out and (labels != -1).any():
            cents = {c: normalize(emb[labels == c].mean(axis=0, keepdims=True))[0]
                     for c in set(labels.tolist()) if c != -1}
            ids = list(cents)
            C = np.vstack([cents[c] for c in ids])
            for i in np.where(labels == -1)[0]:
                labels[i] = ids[int(np.argmax(C @ emb[i]))]
            notes.append(f"{raw_out} HDBSCAN outliers re-assigned to nearest topic centroid")
        W, classes, vocab = ctfidf.class_tfidf(clean_docs, labels)
        words = ctfidf.top_words(W, classes, vocab, self.top_n_words)
        timings["clustering_ctfidf"] = time.perf_counter() - t0
        backend = f"BERTopic-lite: {emb_name} + {red_name} + HDBSCAN + c-TF-IDF"
        if not HAS_BERTOPIC:
            notes.append("`bertopic` package not installed - using the built-in lite pipeline")
        return self._finish(labels, words, emb, reduced, raw_out, timings, notes, backend, p)

    # ---------------------------------------------------------------- common
    def _finish(self, labels, words, emb, reduced, raw_out, timings, notes, backend, params):
        labels, (words,), _ = renumber_by_size(np.asarray(labels), words)
        cents = {}
        for c in set(labels.tolist()) - {-1}:
            cents[c] = normalize(emb[labels == c].mean(axis=0, keepdims=True))[0]
        conf = np.array([float(emb[i] @ cents[l]) if l != -1 else 0.0 for i, l in enumerate(labels)])
        coords = None
        try:
            from sklearn.decomposition import PCA

            base = reduced if reduced is not None and reduced.shape[1] >= 2 else emb
            coords = PCA(n_components=2, random_state=self.random_state).fit_transform(base)
        except Exception as exc:
            log.warning("doc projection failed: %s", exc)
        return TopicResult(model="BERTopic", backend=backend, topic_words=words, assignments=labels,
                           confidence=conf, topic_vectors=cents, doc_coords=coords,
                           raw_outliers=raw_out, timings=timings, notes=notes,
                           params={**params, "nr_topics": self.nr_topics,
                                   "reduce_outliers": self.reduce_outliers})
