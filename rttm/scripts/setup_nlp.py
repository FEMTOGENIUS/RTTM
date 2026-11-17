"""One-time setup + health check ("doctor").

    python scripts/setup_nlp.py            # download models, then show which back-ends are active
    python scripts/setup_nlp.py --check    # only show the back-end report

Downloads: NLTK stopwords/wordnet, spaCy en_core_web_sm, and the
sentence-transformer all-MiniLM-L6-v2 (so the first dashboard run is fast).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def download() -> None:
    try:
        import nltk

        for pkg in ("stopwords", "wordnet", "omw-1.4"):
            nltk.download(pkg, quiet=True)
        print("✔ NLTK data")
    except Exception as exc:
        print("✘ NLTK:", exc)
    try:
        import spacy

        try:
            spacy.load(config.SPACY_MODEL)
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", config.SPACY_MODEL], check=False)
        print("✔ spaCy model", config.SPACY_MODEL)
    except ImportError:
        print("- spaCy not installed (NLTK / rule-based lemmatiser will be used)")
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(config.EMBEDDING_MODEL)
        print("✔ sentence-transformer", config.EMBEDDING_MODEL)
    except ImportError:
        print("- sentence-transformers not installed (LSA embeddings will be used)")
    except Exception as exc:
        print("✘ sentence-transformer download failed:", exc)


def check() -> None:
    from models.bertopic_model import backend_summary
    from models.lda_model import HAS_GENSIM
    from utils.preprocess import _NLTK_STOP, backend_info

    print("\nActive back-ends")
    print("  preprocessing :", backend_info())
    print("  NLTK stopwords:", "yes" if _NLTK_STOP else "no")
    print("  LDA           :", "gensim" if HAS_GENSIM else "scikit-learn (install gensim for the paper setup)")
    for k, v in backend_summary().items():
        print(f"  {k:14s}: {v}")
    print("  NewsAPI keys  :", len(config.NEWSAPI_KEYS), "| GNews keys:", len(config.GNEWS_KEYS))
    full = HAS_GENSIM and "installed" in backend_summary()["bertopic"]
    print("\n=> " + ("FULL paper stack active." if full else
                     "Running with lite fall-backs (everything works; install requirements.txt for the full stack)."))


if __name__ == "__main__":
    if "--check" not in sys.argv:
        download()
    if "--no-check" not in sys.argv:
        check()
