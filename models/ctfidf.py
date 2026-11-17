"""Class-based TF-IDF (Grootendorst, 2022) and c-TF-IDF based topic merging.

    W_{t,c} = tf_{t,c} * log(1 + A / f_t)

tf_{t,c}: frequency of term t in class c (L1-normalised per class)
A       : average number of words per class
f_t     : frequency of term t across all classes
Used by the BERTopic-lite engine (and identical to BERTopic's own default).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def make_vectorizer(n_docs: int = 0, ngram_range=(1, 2)) -> CountVectorizer:
    # NOTE: c-TF-IDF fits the vectorizer on ONE joined document per topic, so
    # min_df must be 1 - otherwise every topic-specific word is discarded.
    return CountVectorizer(ngram_range=ngram_range, min_df=1,
                           token_pattern=r"(?u)\b[a-z][a-z]+\b")


def class_tfidf(docs: list[str], labels: np.ndarray, vectorizer: CountVectorizer | None = None,
                reduce_frequent_words: bool = True):
    """Return (ctfidf matrix [n_classes x V], class ids, vocabulary array)."""
    classes = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    joined = [" ".join(d for d, l in zip(docs, labels) if l == c) for c in classes]
    vectorizer = vectorizer or make_vectorizer(len(docs))
    try:
        X = vectorizer.fit_transform(joined)
    except ValueError:  # vocabulary empty with min_df=2 -> relax
        vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b[a-z][a-z]+\b")
        X = vectorizer.fit_transform(joined)
    X = sparse.csr_matrix(X, dtype=np.float64)
    tf = normalize(X, norm="l1", axis=1)
    if reduce_frequent_words:
        tf.data = np.sqrt(tf.data)
    f_t = np.asarray(X.sum(axis=0)).ravel()
    A = X.sum() / max(X.shape[0], 1)
    idf = np.log(1 + A / np.maximum(f_t, 1))
    W = tf.multiply(idf).tocsr()
    vocab = np.array(vectorizer.get_feature_names_out())
    # suppress accidental n-grams that occur only once inside a topic
    is_ngram = np.array([" " in v for v in vocab])
    if is_ngram.any():
        W = W.multiply(sparse.csr_matrix(~((X.toarray() < 2) & is_ngram[None, :]))).tocsr()
    return W, classes, vocab


def top_words(W, classes: list[int], vocab: np.ndarray, n: int = 25) -> dict[int, list[tuple[str, float]]]:
    out = {}
    for row, c in enumerate(classes):
        vals = W.getrow(row).toarray().ravel()
        idx = np.argsort(-vals)[:n]
        words = [(str(vocab[i]), float(vals[i])) for i in idx if vals[i] > 0]
        # drop unigrams already covered by a stronger bigram to reduce redundancy
        kept: list[tuple[str, float]] = []
        for w, s in words:
            if " " not in w and any(w in k.split() for k, _ in kept if " " in k):
                continue
            kept.append((w, s))
        out[c] = kept
    return out


def merge_to_n_topics(docs: list[str], labels: np.ndarray, n_topics: int,
                      vectorizer: CountVectorizer | None = None) -> np.ndarray:
    """Iteratively merge the two most similar topics (cosine on c-TF-IDF) until n_topics remain."""
    labels = labels.copy()
    while True:
        classes = sorted(c for c in set(labels.tolist()) if c != -1)
        if len(classes) <= n_topics:
            return labels
        W, classes, _ = class_tfidf(docs, labels, vectorizer)
        Wn = normalize(W)
        sim = (Wn @ Wn.T).toarray()
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        a, b = classes[i], classes[j]
        size_a, size_b = (labels == a).sum(), (labels == b).sum()
        small, big = (a, b) if size_a < size_b else (b, a)
        labels[labels == small] = big
