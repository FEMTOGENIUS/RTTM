"""Full paper stack (gensim LDA, BERTopic + all-MiniLM-L6-v2 + UMAP + HDBSCAN).

These tests are SKIPPED automatically when the heavy libraries are not
installed. After `pip install -r requirements.txt` run:

    pytest tests/test_full_stack.py -v

to verify the exact configuration described in the paper on your machine
(first run downloads the ~90 MB sentence-transformer model).
"""
import pytest
from sklearn.metrics import adjusted_rand_score

from data.fetchers import load_sample
from utils.preprocess import preprocess_articles


@pytest.fixture(scope="module")
def df():
    d, _ = preprocess_articles(load_sample())
    return d


def test_gensim_lda_and_gensim_coherence(df):
    pytest.importorskip("gensim")
    from models.lda_model import LDATopicModel
    from utils.metrics import coherence_cv

    r = LDATopicModel(n_topics=9, backend="gensim").fit(df["tokens"].tolist())
    assert r.backend == "gensim LdaModel" and len(r.topic_ids) == 9
    assert adjusted_rand_score(df["gold_topic"], r.assignments) > 0.4
    topics = [r.top_words(t, 10) for t in r.topic_ids]
    g, _ = coherence_cv(topics, df["tokens"].tolist(), use_gensim=True)
    b, _ = coherence_cv(topics, df["tokens"].tolist(), use_gensim=False)
    print(f"C_V gensim={g:.3f} built-in={b:.3f}")
    assert abs(g - b) < 0.2          # built-in implementation agrees with gensim


def test_real_bertopic_with_transformer_embeddings(df):
    pytest.importorskip("bertopic")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("umap")
    from models.bertopic_model import BERTopicModel

    r = BERTopicModel(nr_topics=9, backend="bertopic").fit(df["clean"].tolist(), df["text"].tolist())
    assert r.backend.startswith("BERTopic + SentenceTransformer")
    assert 3 <= len(r.topic_ids) <= 9
    ari = adjusted_rand_score(df["gold_topic"], r.assignments)
    print("BERTopic (full) ARI vs synthetic ground truth:", round(ari, 3))
    assert ari > 0.5


def test_full_pipeline_end_to_end():
    pytest.importorskip("bertopic")
    pytest.importorskip("gensim")
    from models.engine import run_pipeline

    out = run_pipeline(load_sample(), models=["BERTopic", "LDA"], n_topics=9)
    assert not out.warnings, out.warnings
    assert out.metrics["BERTopic"]["coherence_cv"] > 0 and out.metrics["LDA"]["coherence_cv"] > 0
