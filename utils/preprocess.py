"""Text cleaning & normalisation pipeline.

Steps (as specified in the paper/report):
  1. HTML / URL / e-mail / special-character / digit removal
  2. lower-casing
  3. tokenisation
  4. stop-word removal (sklearn + NLTK + custom news stop-words)
  5. lemmatisation (spaCy -> NLTK WordNet -> rule-based fallback)
  6. language detection (langdetect -> heuristic fallback), English-only filter
  7. filtering of short / redundant (duplicate) articles

Whichever NLP back-end is installed is used automatically; the active
back-end is reported in `backend_info()` and shown in the dashboard.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config
from utils.logging_setup import get_logger

log = get_logger("preprocess")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
TRUNC_RE = re.compile(r"\[\+\d+ chars\]")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WS_RE = re.compile(r"\s+")

# Domain-specific stop words for news text (boiler-plate, reporting verbs, time words)
NEWS_STOPWORDS = {
    "said", "says", "say", "told", "according", "reported", "report", "reports", "reuters",
    "bbc", "cnn", "ap", "afp", "pti", "ani", "news", "live", "update", "updates", "breaking",
    "video", "watch", "photo", "photos", "image", "images", "read", "click", "subscribe",
    "newsletter", "today", "yesterday", "tomorrow", "week", "weeks", "month", "months", "year",
    "years", "day", "days", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "mr", "mrs", "ms", "also", "would", "could",
    "one", "two", "three", "new", "get", "make", "like", "just", "time", "first", "last", "latest",
    "chars", "copyright", "rights", "reserved", "via", "amp", "nbsp", "quot", "com", "www",
    "http", "https", "html", "pic", "twitter", "facebook", "instagram", "youtube", "know",
    "people", "want", "need", "thing", "way", "use", "going", "come", "take", "look",
}
KEEP_SHORT = {"ai", "uk", "eu", "un", "ev"}



# --------------------------------------------------------------------------- lemmatiser
@lru_cache(maxsize=1)
def _spacy_nlp():
    try:
        import spacy  # type: ignore

        return spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
    except Exception as exc:
        log.info("spaCy unavailable (%s) - trying NLTK", exc)
        return None


def _ensure_nltk_data() -> None:
    """Download NLTK stop-words / WordNet once if NLTK is installed but its data is
    missing (e.g. on Streamlit Cloud). Disable with RTTM_NO_DOWNLOAD=1."""
    import os

    if os.getenv("RTTM_NO_DOWNLOAD"):
        return
    try:
        import nltk  # type: ignore
    except Exception:
        return
    for res, pkg in (("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                     ("corpora/omw-1.4", "omw-1.4")):
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk_data()

_NLTK_STOP: set[str] = set()
try:  # optional NLTK stop-word list
    from nltk.corpus import stopwords as _sw  # type: ignore

    _NLTK_STOP = set(_sw.words("english"))
except Exception:
    pass

STOPWORDS: frozenset[str] = frozenset(set(ENGLISH_STOP_WORDS) | _NLTK_STOP | NEWS_STOPWORDS)


@lru_cache(maxsize=1)
def _wordnet():
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore

        wnl = WordNetLemmatizer()
        wnl.lemmatize("tests")  # raises LookupError if corpus missing
        return wnl
    except Exception as exc:
        log.info("NLTK WordNet unavailable (%s) - using rule-based lemmatiser", exc)
        return None


_IRREGULAR = {"children": "child", "men": "man", "women": "woman", "people": "people",
              "mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose",
              "was": "be", "were": "be", "is": "be", "are": "be", "has": "have", "had": "have",
              "won": "win", "wins": "win", "ran": "run", "rose": "rise", "fell": "fall",
              "falls": "fall", "led": "lead", "hit": "hit", "took": "take", "data": "data",
              "news": "news", "series": "series", "species": "species", "crises": "crisis",
              "analyses": "analysis", "indices": "index", "media": "media"}
_NO_STRIP = ("ss", "us", "is", "os")


def rule_lemma(w: str) -> str:
    """Light-weight English lemmatiser (plural nouns / common verb forms)."""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if w.endswith("eas"):                      # areas, ideas, seas
        return w[:-1]
    if w.endswith("s") and not w.endswith(_NO_STRIP + ("as",)):  # keeps thomas, texas, gas
        return w[:-1]
    return w


def backend_info() -> dict[str, str]:
    lemma = "spaCy" if _spacy_nlp() else ("NLTK WordNet" if _wordnet() else "rule-based")
    return {"lemmatizer": lemma, "language_detection": "langdetect" if _langdetect() else "heuristic",
            "stopwords": f"sklearn{' + NLTK' if _NLTK_STOP else ''} + {len(NEWS_STOPWORDS)} news-specific"}


# --------------------------------------------------------------------------- language
@lru_cache(maxsize=1)
def _langdetect():
    try:
        from langdetect import DetectorFactory, detect  # type: ignore

        DetectorFactory.seed = 0
        return detect
    except Exception:
        return None


_EN_HINTS = {"the", "and", "of", "to", "in", "for", "on", "with", "as", "is", "at", "by", "after",
             "from", "a", "an", "be", "has", "over", "new", "its", "are", "said", "says",
             "will", "was", "were", "more", "than", "this", "that", "into", "near"}
# frequent function words of es / fr / de / it / pt (latin-script non-English)
_FOREIGN_HINTS = {"el", "la", "los", "las", "del", "y", "que", "por", "con", "una", "para",
                  "le", "les", "des", "et", "du", "pour", "dans", "est", "une", "sur", "au",
                  "der", "die", "das", "und", "mit", "ist", "den", "von", "zu", "ein", "eine", "nicht",
                  "il", "di", "che", "per", "della", "sono", "não", "em", "uma", "os", "ao"}


def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"
    det = _langdetect()
    if det is not None and len(text) > 20:
        try:
            return det(text)
        except Exception:
            pass
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    if ascii_ratio < 0.85:
        return "non-en"
    words = re.findall(r"[a-zA-Z']+", text.lower())
    en_hits = sum(w in _EN_HINTS for w in words)
    foreign_hits = sum(w in _FOREIGN_HINTS for w in words)
    return "en" if words and en_hits >= foreign_hits else "non-en"


# --------------------------------------------------------------------------- cleaning
def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = TRUNC_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = NON_ALPHA_RE.sub(" ", text)          # punctuation, digits, special chars
    return WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """clean -> tokens -> lemmas -> stop-word filtered tokens."""
    cleaned = clean_text(text)
    nlp = _spacy_nlp()
    if nlp is not None:
        lemmas = [t.lemma_.lower() for t in nlp(cleaned) if not t.is_space]
    else:
        toks = cleaned.split()
        wnl = _wordnet()
        if wnl is not None:
            lemmas = [wnl.lemmatize(wnl.lemmatize(t), pos="v") if t not in _IRREGULAR else _IRREGULAR[t]
                      for t in toks]
        else:
            lemmas = [rule_lemma(t) for t in toks]
    return [w for w in lemmas
            if (len(w) > 2 or w in KEEP_SHORT) and w not in STOPWORDS and w.isalpha()]


def build_text(a: dict) -> str:
    parts = [a.get("title", ""), a.get("description", "")]
    content = a.get("content", "") or ""
    desc = a.get("description", "") or ""
    if content and content[:80] != desc[:80]:
        parts.append(content)
    elif content and len(content) > len(desc) + 40:
        parts.append(content[len(desc):])
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())


def preprocess_articles(articles: Iterable[dict], *, english_only: bool = True,
                        min_tokens: int = 3) -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, stats). Dataframe columns:
    id, title, description, content, source, url, published_at, text (raw, for
    embeddings), tokens (list), clean (joined lemmas), language.
    """
    rows = list(articles)
    stats = {"input": len(rows), "non_english": 0, "too_short": 0, "duplicates": 0, "errors": 0}
    out = []
    seen_clean: set[str] = set()
    seen_ids: set[str] = set()
    for a in rows:
        try:
            if a.get("id") and a["id"] in seen_ids:   # same headline hash
                stats["duplicates"] += 1
                continue
            seen_ids.add(a.get("id"))
            text = build_text(a)
            lang = detect_language(f"{a.get('title', '')}. {a.get('description', '')}")
            if english_only and lang not in ("en", "unknown"):
                stats["non_english"] += 1
                continue
            toks = tokenize(text)
            if len(toks) < min_tokens:
                stats["too_short"] += 1
                continue
            clean = " ".join(toks)
            if clean in seen_clean:          # redundant article (same normalised text)
                stats["duplicates"] += 1
                continue
            seen_clean.add(clean)
            out.append({**a, "text": text, "tokens": toks, "clean": clean, "language": lang})
        except Exception as exc:  # never let one bad record stop the pipeline
            stats["errors"] += 1
            log.warning("preprocess error on %s: %s", a.get("id"), exc)
    df = pd.DataFrame(out)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["published_at"] = df["published_at"].fillna(pd.Timestamp.now(tz="UTC"))
        df = df.sort_values("published_at").reset_index(drop=True)
    stats["output"] = len(df)
    log.info("preprocess: %s", stats)
    return df, stats
