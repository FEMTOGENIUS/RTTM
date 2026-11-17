"""Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003).

Primary implementation: gensim `LdaModel` (as stated in the paper/report).
Fallback: scikit-learn `LatentDirichletAllocation` (identical model family),
used automatically when gensim is not installed.

`weighting="bow"` (default, the statistically correct input for LDA) or
`weighting="tfidf"` (the TF-IDF document-term matrix variant described in the
paper) can be selected.
"""
from __future__ import annotations

import time

import numpy as np

import config
from models.base import TopicResult
from utils.logging_setup import get_logger

log = get_logger("lda")

try:  # optional
    from gensim import corpora, models as gmodels  # type: ignore

    HAS_GENSIM = True
except Exception:  # pragma: no cover
    HAS_GENSIM = False


class LDATopicModel:
    def __init__(self, n_topics: int = config.DEFAULT_N_TOPICS, passes: int = 10,
                 weighting: str = "bow", random_state: int = config.RANDOM_STATE,
                 backend: str = "auto", top_n_words: int = 25):
        self.n_topics = int(n_topics)
        self.passes = int(passes)
        self.weighting = weighting
        self.random_state = random_state
        self.backend = backend
        self.top_n_words = top_n_words
        self.model_ = None
        self.dictionary_ = None

    # ---------------------------------------------------------------- public
    def fit(self, token_lists: list[list[str]]) -> TopicResult:
        n = len(token_lists)
        if n < 3:
            raise ValueError(f"LDA needs at least 3 documents (got {n})")
        k = max(2, min(self.n_topics, n - 1))
        use_gensim = HAS_GENSIM and self.backend in ("auto", "gensim")
        t0 = time.perf_counter()
        if use_gensim:
            words, doc_topic, topic_term_vecs, backend = self._fit_gensim(token_lists, k)
        else:
            words, doc_topic, topic_term_vecs, backend = self._fit_sklearn(token_lists, k)
        fit_s = time.perf_counter() - t0
        assign = doc_topic.argmax(axis=1)
        conf = doc_topic.max(axis=1)
        # keep every LDA topic (even empty ones) but order them by size
        all_ids = list(range(k))
        counts = {t: int((assign == t).sum()) for t in all_ids}
        order = sorted(all_ids, key=lambda t: (-counts[t], t))
        mapping = {old: new for new, old in enumerate(order)}
        assign = np.array([mapping[a] for a in assign])
        doc_topic = doc_topic[:, order]
        words = {mapping[t]: w for t, w in words.items()}
        vecs = {mapping[t]: v for t, v in topic_term_vecs.items()}
        notes = []
        if k != self.n_topics:
            notes.append(f"n_topics reduced to {k} because only {n} documents are available")
        return TopicResult(model="LDA", backend=backend, topic_words=words, assignments=assign,
                           confidence=conf, doc_topic=doc_topic, topic_vectors=vecs,
                           timings={"lda_fit": fit_s}, notes=notes,
                           extra={"model": self.model_, "dictionary": self.dictionary_,
                                  "corpus": getattr(self, "corpus_", None), "order": order},
                           params={"n_topics": k, "passes": self.passes, "weighting": self.weighting})

    # ---------------------------------------------------------------- gensim
    def _fit_gensim(self, token_lists, k):
        n = len(token_lists)
        d = corpora.Dictionary(token_lists)
        no_below = 2 if n >= 40 else 1
        d.filter_extremes(no_below=no_below, no_above=0.5 if n >= 40 else 0.95, keep_n=10000)
        if len(d) < k:
            d = corpora.Dictionary(token_lists)
        corpus = [d.doc2bow(t) for t in token_lists]
        self.corpus_ = corpus
        if self.weighting == "tfidf":
            tfidf = gmodels.TfidfModel(corpus)
            corpus = [tfidf[c] for c in corpus]
        lda = gmodels.LdaModel(corpus=corpus, id2word=d, num_topics=k, passes=self.passes,
                               iterations=100, alpha="auto", eta="auto",
                               random_state=self.random_state, eval_every=None, chunksize=2000)
        self.model_, self.dictionary_ = lda, d
        words = {t: [(w, float(p)) for w, p in lda.show_topic(t, topn=self.top_n_words)] for t in range(k)}
        doc_topic = np.zeros((n, k))
        for i, bow in enumerate(corpus):
            for t, p in lda.get_document_topics(bow, minimum_probability=0.0):
                doc_topic[i, t] = p
        # empty documents (all tokens filtered): uniform
        doc_topic[doc_topic.sum(axis=1) == 0] = 1.0 / k
        topic_term = lda.get_topics()
        return words, doc_topic, {t: topic_term[t] for t in range(k)}, "gensim LdaModel"

    # ---------------------------------------------------------------- sklearn
    def _fit_sklearn(self, token_lists, k):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        n = len(token_lists)
        Vec = TfidfVectorizer if self.weighting == "tfidf" else CountVectorizer
        kwargs = dict(analyzer=lambda x: x, min_df=2 if n >= 40 else 1, max_df=0.5 if n >= 40 else 1.0)
        try:
            vec = Vec(**kwargs)
            X = vec.fit_transform(token_lists)
        except ValueError:
            vec = Vec(analyzer=lambda x: x)
            X = vec.fit_transform(token_lists)
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch",
                                        max_iter=max(10, self.passes * 2),
                                        random_state=self.random_state)
        doc_topic = lda.fit_transform(X)
        self.model_, self.dictionary_ = lda, vec
        vocab = np.array(vec.get_feature_names_out())
        comp = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        words = {}
        for t in range(k):
            idx = np.argsort(-comp[t])[: self.top_n_words]
            words[t] = [(str(vocab[i]), float(comp[t, i])) for i in idx]
        return words, doc_topic, {t: comp[t] for t in range(k)}, "scikit-learn LDA (gensim not installed)"
