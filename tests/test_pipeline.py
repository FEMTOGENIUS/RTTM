"""Preprocessing, store, metrics, models, engine, trends."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from data.fetchers import load_sample, normalize_article
from data.store import ArticleStore
from models import engine
from models.bertopic_model import BERTopicModel
from models.ctfidf import class_tfidf, top_words
from models.lda_model import LDATopicModel
from utils import metrics as M
from utils import preprocess as P
from utils.labels import categorize
from utils.trends import trend_table


# ------------------------------------------------------------------ preprocessing
def test_clean_text_removes_noise():
    t = P.clean_text("<b>BREAKING:</b> Visit https://x.com/a or mail a@b.com — 2026's AI boom!!")
    assert "http" not in t and "@" not in t and "2026" not in t and "<b>" not in t
    assert t.startswith("breaking")


def test_tokenize_lemmatises_and_removes_stopwords():
    toks = P.tokenize("The voters said the elections and cyclones were reported by Reuters today")
    assert "the" not in toks and "said" not in toks and "reuters" not in toks and "today" not in toks
    assert "voter" in toks and "election" in toks and "cyclone" in toks


def test_ai_survives_tokenisation():
    assert "ai" in P.tokenize("AI chips power new AI models")


def test_language_detection():
    assert P.detect_language("Stocks rally after the central bank holds interest rates") == "en"
    assert P.detect_language("El banco central mantiene las tasas de interés") != "en"
    assert P.detect_language("चुनाव आयोग ने मतदान की तारीखों की घोषणा की") != "en"


def test_preprocess_filters_non_english_duplicates_and_short():
    rows = [normalize_article(title="Cyclone hits coast, thousands evacuated", description="Heavy rain and wind"),
            normalize_article(title="Cyclone hits coast, thousands evacuated!", description="dup headline"),
            normalize_article(title="El banco central mantiene las tasas", description="Los inversores esperan"),
            normalize_article(title="Hi", description="")]
    df, st = P.preprocess_articles(rows)
    assert len(df) == 1 and st["duplicates"] == 1 and st["non_english"] == 1 and st["too_short"] == 1
    assert {"tokens", "clean", "text"} <= set(df.columns)


# ------------------------------------------------------------------ store
def test_store_dedupes_and_prunes(tmp_path):
    s = ArticleStore(tmp_path / "a.jsonl")
    old = normalize_article(title="Old story", url="u0",
                            published_at=datetime.now(timezone.utc) - timedelta(days=40))
    a = normalize_article(title="Fresh story", url="u1")
    assert s.add([a, old]) == 2
    assert s.add([normalize_article(title="Fresh story", url="u9"),
                  normalize_article(title="Other", url="u1")]) == 0   # same headline / same URL
    assert len(s.load(hours=24)) == 1
    assert s.prune(max_age_days=14) == 1 and len(s.all()) == 1


# ------------------------------------------------------------------ metrics
def test_coherence_prefers_coherent_topics():
    texts = [["cricket", "match", "wicket", "bowler"]] * 20 + [["bank", "inflation", "rate", "market"]] * 20
    good = [["cricket", "match", "wicket"], ["bank", "inflation", "rate"]]
    bad = [["cricket", "bank", "rate"], ["wicket", "inflation", "match"]]
    assert M.coherence_cv(good, texts, use_gensim=False)[0] > M.coherence_cv(bad, texts, use_gensim=False)[0]
    assert M.coherence_npmi(good, texts)[0] > M.coherence_npmi(bad, texts)[0]
    assert M.coherence_umass(good, texts)[0] > M.coherence_umass(bad, texts)[0]


def test_diversity_formula():
    assert M.topic_diversity([["a", "b"], ["c", "d"]], 2) == 1.0
    assert M.topic_diversity([["a", "b"], ["a", "b"]], 2) == 0.5


def test_unigram_split_of_bigrams():
    assert M.unigram_topics([["climate change", "climate", "flood"]], 10) == [["climate", "change", "flood"]]


# ------------------------------------------------------------------ models
@pytest.fixture(scope="module")
def sample_df():
    df, _ = P.preprocess_articles(load_sample())
    return df


def test_lda_recovers_sample_topics(sample_df):
    r = LDATopicModel(n_topics=9, backend="sklearn").fit(sample_df["tokens"].tolist())
    assert r.model == "LDA" and len(r.topic_ids) == 9
    assert r.doc_topic.shape == (len(sample_df), 9)
    assert np.allclose(r.doc_topic.sum(axis=1), 1, atol=1e-6)
    assert adjusted_rand_score(sample_df["gold_topic"], r.assignments) > 0.5


def test_bertopic_lite_recovers_sample_topics(sample_df):
    r = BERTopicModel(nr_topics=9, backend="auto").fit(sample_df["clean"].tolist(), sample_df["text"].tolist())
    assert len(r.topic_ids) == 9
    assert adjusted_rand_score(sample_df["gold_topic"], r.assignments) > 0.8
    assert r.doc_coords.shape == (len(sample_df), 2)
    words = {w for t in r.topic_ids for w in r.top_words(t, 10)}
    assert {"cyclone", "election"} & words


def test_bertopic_too_few_docs():
    with pytest.raises(ValueError):
        BERTopicModel().fit(["a b c"] * 5, ["a b c"] * 5)


def test_ctfidf_keeps_topic_specific_words():
    docs = ["wicket bowler cricket", "wicket cricket match", "bank rate inflation", "bank inflation market"]
    W, classes, vocab = class_tfidf(docs, np.array([0, 0, 1, 1]))
    tw = top_words(W, classes, vocab, 5)
    assert tw[0][0][0] in ("wicket", "cricket") and tw[1][0][0] in ("bank", "inflation")


def test_category_labels():
    assert categorize(["cyclone", "flood", "rain"])[0] == "Climate & Disaster"
    assert categorize(["court", "bail", "convict"])[0] == "Law & Justice"
    assert categorize(["xyz", "qwe"])[0] == "General"


# ------------------------------------------------------------------ engine
def test_pipeline_both_models_and_comparison():
    out = engine.run_pipeline(load_sample(), models=["BERTopic", "LDA"], n_topics=9)
    assert set(out.results) == {"BERTopic", "LDA"} and not out.warnings
    for m in out.results:
        met = out.metrics[m]
        assert 0 < met["coherence_cv"] <= 1 and 0 < met["diversity"] <= 1
        assert len(out.labels[m]) == len(out.results[m].topic_ids)
    assert out.comparison["jaccard"].shape == (len(out.results["LDA"].topic_ids),
                                               len(out.results["BERTopic"].topic_ids))
    assert out.word_freq and out.timings["total"] > 0
    m = engine.intertopic_map(out.results["LDA"])
    assert len(m) == 9 and {"x", "y", "size"} <= set(m.columns)
    assert engine.search_topics(out, "BERTopic", "cyclone")
    d = out.doc_frame("BERTopic")
    assert {"topic", "topic_name", "confidence"} <= set(d.columns)


def test_bertopic_failure_falls_back_to_lda(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("simulated BERTopic crash")

    monkeypatch.setattr(BERTopicModel, "fit", boom)
    out = engine.run_pipeline(load_sample(), models=["BERTopic"], n_topics=5)
    assert "LDA" in out.results and "BERTopic" not in out.results
    assert any("falling back to LDA" in w for w in out.warnings)


def test_pipeline_with_too_few_articles():
    out = engine.run_pipeline([normalize_article(title="Only one article here about markets")])
    assert not out.results and out.warnings


def test_emerging_and_fading_detection():
    out = engine.run_pipeline(load_sample(), models=["BERTopic"], bertopic_nr_topics=9)
    tr = out.trends["BERTopic"]
    status = {out.labels["BERTopic"][t]["category"]: s for t, s in zip(tr["topic"], tr["status"])}
    assert status["Climate & Disaster"].endswith("Emerging")      # cyclone burst in the last 36h
    assert status["Entertainment"].endswith("Fading")             # film festival 4-7 days ago
    assert any("Emerging" in a for a in out.alerts["BERTopic"])


def test_trend_table_empty():
    import pandas as pd

    assert trend_table(pd.DataFrame(), [], {}).empty
