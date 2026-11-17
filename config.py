"""Central configuration for the Real-Time Topic Modeling (RTTM) system.

Values can be overridden through environment variables or a `.env` file
(see `.env.example`). API keys are never hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = Path(os.getenv("RTTM_STORE_DIR", DATA_DIR / "store"))
STORE_PATH = STORE_DIR / "articles.jsonl"
SAMPLE_PATH = DATA_DIR / "sample_articles.json"
LOG_DIR = Path(os.getenv("RTTM_LOG_DIR", BASE_DIR / "logs"))
RESULTS_DIR = BASE_DIR / "results"
FEEDBACK_PATH = LOG_DIR / "feedback.jsonl"


def _split_keys(*names: str) -> list[str]:
    keys: list[str] = []
    for name in names:
        raw = os.getenv(name, "")
        keys += [k.strip() for k in raw.split(",") if k.strip()]
    # de-duplicate, keep order
    return list(dict.fromkeys(keys))


# Multiple keys may be given comma-separated -> automatic key rotation on rate limits.
NEWSAPI_KEYS = _split_keys("NEWSAPI_KEYS", "NEWSAPI_KEY")
GNEWS_KEYS = _split_keys("GNEWS_KEYS", "GNEWS_KEY", "GNEWS_API_KEY")

# Public RSS feeds (no key required). Edit freely.
RSS_FEEDS: dict[str, str] = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "BBC Health": "https://feeds.bbci.co.uk/news/health/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "Indian Express": "https://indianexpress.com/feed/",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
}

# Fetch / scheduler settings
FETCH_INTERVAL_MIN = float(os.getenv("RTTM_FETCH_INTERVAL_MIN", "10"))
HTTP_TIMEOUT = float(os.getenv("RTTM_HTTP_TIMEOUT", "15"))
USER_AGENT = os.getenv(
    "RTTM_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36",
)
STORE_MAX_AGE_DAYS = int(os.getenv("RTTM_STORE_MAX_AGE_DAYS", "14"))
STORE_MAX_ITEMS = int(os.getenv("RTTM_STORE_MAX_ITEMS", "20000"))

# Modeling defaults
EMBEDDING_MODEL = os.getenv("RTTM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# used automatically for Hindi / French / German / Spanish editions
MULTILINGUAL_EMBEDDING_MODEL = os.getenv("RTTM_ML_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
SPACY_MODEL = os.getenv("RTTM_SPACY_MODEL", "en_core_web_sm")
RANDOM_STATE = 42
DEFAULT_N_TOPICS = 8
TOP_N_WORDS = 10
