"""Real-time news ingestion.

Sources
-------
* NewsAPI.org      (/v2/everything, /v2/top-headlines)  – needs NEWSAPI_KEY(S)
* GNews.io         (/api/v4/search, /api/v4/top-headlines) – needs GNEWS_KEY(S)
* Google News RSS  (keyword search, no key)
* Any RSS / Atom feed (BBC, Al Jazeera, CNN, Indian Express, ... ; no key)
* Offline sample dataset / uploaded CSV-JSON (static mode)

Every source returns *normalised* article dicts (see `normalize_article`).
Errors are logged and returned, never raised to the UI, so one failing source
never breaks the pipeline (graceful degradation).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus

import requests

import config
from utils.logging_setup import get_logger

log = get_logger("fetchers")

try:  # optional, faster + more tolerant RSS parsing
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None


class FetchError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers
_TAG_RE = re.compile(r"<[^>]+>")
_TRUNC_RE = re.compile(r"\s*\[\+\d+ chars\]\s*$")  # NewsAPI truncation marker
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    text = _TRUNC_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_title(title: str) -> str:
    t = strip_html(title).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


def article_id(title: str, url: str = "") -> str:
    """Stable hash used for de-duplication (headline hash; URL as fallback)."""
    key = normalize_title(title) or (url or "").strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_date(value: Any) -> str:
    """Return an ISO-8601 UTC string; falls back to *now* when unparsable."""
    if value is None or value == "":
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if isinstance(value, time.struct_time):
        return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    s = str(value).strip()
    for parser in (
        lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")),
        parsedate_to_datetime,
    ):
        try:
            dt = parser(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return datetime.now(timezone.utc).isoformat()


_IMG_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)


def first_image(*htmls: str) -> str:
    for h in htmls:
        m = _IMG_RE.search(h or "")
        if m and m.group(1).startswith("http"):
            return m.group(1)
    return ""


def normalize_article(*, title: str, description: str = "", content: str = "",
                      source: str = "", url: str = "", published_at: Any = None,
                      language: str = "en", origin: str = "", image: str = "") -> dict:
    image = image or first_image(description, content)
    title = strip_html(title)
    return {
        "id": article_id(title, url),
        "title": title,
        "description": strip_html(description),
        "content": strip_html(content),
        "source": (source or "Unknown").strip(),
        "url": (url or "").strip(),
        "published_at": parse_date(published_at),
        "language": language or "en",
        "origin": origin,
        "image": image if str(image).startswith("http") else "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT,
                      "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                      "Accept-Language": "en-IN,en;q=0.9"})
    return s


def _get_with_retry(session, url: str, retries: int = 2, **kw):
    """GET with a short back-off on 429 (rate limited) and 5xx responses."""
    r = None
    for attempt in range(retries + 1):
        r = session.get(url, timeout=config.HTTP_TIMEOUT, **kw)
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
            return r
        wait = min(8.0, 1.5 * (attempt + 1))
        log.info("HTTP %s from %s - retrying in %.1fs", r.status_code, url[:60], wait)
        time.sleep(wait)
    return r


# --------------------------------------------------------------------------- NewsAPI
NEWSAPI_BASE = "https://newsapi.org/v2"


def fetch_newsapi(query: str = "", *, language: str = "en", page_size: int = 50,
                  from_date: datetime | None = None, to_date: datetime | None = None,
                  sort_by: str = "publishedAt", keys: list[str] | None = None,
                  country: str = "", session: requests.Session | None = None) -> list[dict]:
    keys = list(keys if keys is not None else config.NEWSAPI_KEYS)
    if not keys:
        raise FetchError("NewsAPI: no API key configured (set NEWSAPI_KEY in .env)")
    session = session or _session()
    page_size = max(1, min(int(page_size), 100))
    if query:
        endpoint = f"{NEWSAPI_BASE}/everything"
        params: dict[str, Any] = {"q": query, "language": language, "pageSize": page_size,
                                  "sortBy": sort_by}
        if from_date:
            params["from"] = from_date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        if to_date:
            params["to"] = to_date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    else:
        endpoint = f"{NEWSAPI_BASE}/top-headlines"
        params = {"pageSize": page_size, "language": language}
        if country:
            params["country"] = country
    last_err = ""
    for i, key in enumerate(keys):  # key rotation
        try:
            r = session.get(endpoint, params=params, headers={"X-Api-Key": key},
                            timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            raise FetchError(f"NewsAPI network error: {exc}") from exc
        if r.status_code in (401, 426, 429):
            last_err = f"key #{i + 1} rejected ({r.status_code}): {r.text[:160]}"
            log.warning("NewsAPI %s - rotating key", last_err)
            continue
        if r.status_code != 200:
            raise FetchError(f"NewsAPI HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if data.get("status") != "ok":
            raise FetchError(f"NewsAPI error: {data.get('message')}")
        out = []
        for a in data.get("articles", []):
            if not a.get("title") or a.get("title") == "[Removed]":
                continue
            out.append(normalize_article(
                title=a.get("title", ""), description=a.get("description") or "",
                content=a.get("content") or "", source=(a.get("source") or {}).get("name", ""),
                url=a.get("url", ""), published_at=a.get("publishedAt"),
                language=language, origin="newsapi", image=a.get("urlToImage") or ""))
        log.info("NewsAPI: %d articles (query=%r)", len(out), query)
        return out
    raise FetchError(f"NewsAPI: all keys exhausted - {last_err}")


# --------------------------------------------------------------------------- GNews
GNEWS_BASE = "https://gnews.io/api/v4"


def fetch_gnews(query: str = "", *, language: str = "en", max_results: int = 50,
                from_date: datetime | None = None, to_date: datetime | None = None,
                keys: list[str] | None = None, country: str = "",
                session: requests.Session | None = None) -> list[dict]:
    keys = list(keys if keys is not None else config.GNEWS_KEYS)
    if not keys:
        raise FetchError("GNews: no API key configured (set GNEWS_KEY in .env)")
    session = session or _session()
    endpoint = f"{GNEWS_BASE}/search" if query else f"{GNEWS_BASE}/top-headlines"
    params: dict[str, Any] = {"lang": language, "max": max(1, min(int(max_results), 100))}
    if query:
        params["q"] = query
    if country:
        params["country"] = country
    if from_date:
        params["from"] = from_date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if to_date:
        params["to"] = to_date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_err = ""
    for i, key in enumerate(keys):
        try:
            r = session.get(endpoint, params={**params, "apikey": key}, timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            raise FetchError(f"GNews network error: {exc}") from exc
        if r.status_code in (401, 403, 429):
            last_err = f"key #{i + 1} rejected ({r.status_code}): {r.text[:160]}"
            log.warning("GNews %s - rotating key", last_err)
            continue
        if r.status_code != 200:
            raise FetchError(f"GNews HTTP {r.status_code}: {r.text[:200]}")
        out = []
        for a in r.json().get("articles", []):
            out.append(normalize_article(
                title=a.get("title", ""), description=a.get("description") or "",
                content=a.get("content") or "", source=(a.get("source") or {}).get("name", ""),
                url=a.get("url", ""), published_at=a.get("publishedAt"),
                language=language, origin="gnews", image=a.get("image") or ""))
        log.info("GNews: %d articles (query=%r)", len(out), query)
        return out
    raise FetchError(f"GNews: all keys exhausted - {last_err}")


# --------------------------------------------------------------------------- RSS / Atom
_ATOM = "{http://www.w3.org/2005/Atom}"
_MEDIA_NS = ("{http://purl.org/rss/1.0/modules/content/}encoded",)
_MRSS = "{http://search.yahoo.com/mrss/}"


def _xml_image(it: ET.Element) -> str:
    for tag in (f"{_MRSS}thumbnail", f"{_MRSS}content", f"{_MRSS}group/{_MRSS}content"):
        el = it.find(tag)
        if el is not None and el.get("url", "").startswith("http"):
            return el.get("url")
    enc = it.find("enclosure")
    if enc is not None and enc.get("type", "").startswith("image"):
        return enc.get("url", "")
    return ""


def _txt(el: ET.Element | None) -> str:
    return "" if el is None else ("".join(el.itertext()) or "").strip()


def parse_feed_xml(xml_text: str | bytes, source_name: str = "RSS") -> list[dict]:
    """Dependency-free RSS 2.0 / Atom parser (used when feedparser is absent)."""
    if isinstance(xml_text, str):
        xml_text = xml_text.encode("utf-8")
    root = ET.fromstring(xml_text)
    items: list[dict] = []
    channel_title = _txt(root.find("channel/title")) or _txt(root.find(f"{_ATOM}title"))
    for it in root.iter("item"):
        src_el = it.find("source")
        content = ""
        for tag in _MEDIA_NS:
            content = _txt(it.find(tag)) or content
        items.append(dict(title=_txt(it.find("title")), description=_txt(it.find("description")),
                          content=content, url=_txt(it.find("link")),
                          published_at=_txt(it.find("pubDate")) or _txt(it.find("{http://purl.org/dc/elements/1.1/}date")),
                          source=_txt(src_el) or source_name or channel_title, image=_xml_image(it)))
    for it in root.iter(f"{_ATOM}entry"):
        link_el = it.find(f"{_ATOM}link")
        items.append(dict(title=_txt(it.find(f"{_ATOM}title")),
                          description=_txt(it.find(f"{_ATOM}summary")),
                          content=_txt(it.find(f"{_ATOM}content")),
                          url=link_el.get("href", "") if link_el is not None else "",
                          published_at=_txt(it.find(f"{_ATOM}published")) or _txt(it.find(f"{_ATOM}updated")),
                          source=source_name or channel_title, image=_xml_image(it)))
    return items


_ANCHOR_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", re.S | re.I)


def flatten_link_list(desc: str) -> str:
    """Google News puts up to 5 related headlines (other publishers) in the
    description as <ol><li><a>headline</a> <font>publisher</font></li>...</ol>.
    Turn that into plain text so the extra headlines enrich the document."""
    if not desc or not ("<ol" in desc or ("<font" in desc and "<a " in desc)):
        return desc
    heads = [strip_html(a) for a in _ANCHOR_RE.findall(desc)]
    heads = [h for h in heads if h and not h.lower().startswith("see more headlines")]
    return ". ".join(dict.fromkeys(heads))


def fetch_rss(feeds: dict[str, str], *, query: str = "", limit_per_feed: int = 50,
              session: requests.Session | None = None) -> tuple[list[dict], list[str]]:
    """Fetch several feeds; returns (articles, errors). Optional keyword filter."""
    session = session or _session()
    out: list[dict] = []
    errors: list[str] = []
    terms = [t.lower() for t in re.findall(r"\w+", query or "")]
    for name, url in feeds.items():
        try:
            r = _get_with_retry(session, url)
            if r.status_code == 429:
                raise FetchError("rate-limited by the site (HTTP 429) - try again in a few minutes")
            if r.status_code == 403:
                raise FetchError("the site blocks automated readers from this server (HTTP 403)")
            r.raise_for_status()
            if feedparser is not None:
                parsed = feedparser.parse(r.content)
                raw = [dict(title=e.get("title", ""),
                            description=e.get("summary", "") or e.get("description", ""),
                            content=(e.get("content") or [{}])[0].get("value", "") if e.get("content") else "",
                            url=e.get("link", ""),
                            published_at=e.get("published_parsed") or e.get("updated_parsed") or e.get("published"),
                            source=(e.get("source") or {}).get("title") or name,
                            image=((e.get("media_thumbnail") or [{}])[0].get("url")
                                   or (e.get("media_content") or [{}])[0].get("url")
                                   or next((l.get("href") for l in e.get("links", [])
                                            if str(l.get("type", "")).startswith("image")), "") or ""))
                       for e in parsed.entries]
            else:
                raw = parse_feed_xml(r.content, name)
            n = 0
            for a in raw:
                if not a.get("title"):
                    continue
                a["description"] = flatten_link_list(a.get("description", ""))
                a.setdefault("image", "")
                blob = f"{a.get('title', '')} {a.get('description', '')}".lower()
                if terms and not any(t in blob for t in terms):
                    continue
                art = normalize_article(origin="rss", language="en", **a)
                art["feed"] = name
                out.append(art)
                n += 1
                if n >= limit_per_feed:
                    break
            log.info("RSS %s: %d articles", name, n)
        except Exception as exc:  # one bad feed must not break the others
            msg = f"RSS '{name}' failed: {exc}"
            log.warning(msg)
            errors.append(msg)
    return out, errors


def google_news_rss_url(query: str = "", language: str = "en", country: str = "IN") -> str:
    hl = f"{language}-{country}" if language == "en" else language
    ceid = f"{country}:{language}"
    if query:
        return (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:2d"
                f"&hl={hl}&gl={country}&ceid={ceid}")
    return f"https://news.google.com/rss?hl={hl}&gl={country}&ceid={ceid}"


def fetch_google_news(query: str = "", *, language: str = "en", country: str = "IN",
                      limit: int = 60, session: requests.Session | None = None) -> list[dict]:
    url = google_news_rss_url(query, language, country)
    arts, errs = fetch_rss({"Google News": url}, limit_per_feed=limit, session=session)
    if errs and not arts:
        raise FetchError(errs[0])
    for a in arts:  # Google titles look like "Headline - Publisher"
        if " - " in a["title"]:
            head, pub = a["title"].rsplit(" - ", 1)
            if len(pub) < 60:
                a["title"], a["source"] = head.strip(), pub.strip()
                a["id"] = article_id(a["title"], a["url"])
        # drop the related-headline that merely repeats the title; keep the others
        t_norm = normalize_title(a["title"])
        related = [h for h in a["description"].split(". ")
                   if h and not normalize_title(h).startswith(t_norm[:40])]
        a["description"] = ". ".join(related)
        a["origin"] = "google_news"
    return arts


# --------------------------------------------------------------------------- full text
def extract_full_text(articles: list[dict], max_articles: int = 30) -> int:
    """Optional: download full article bodies with newspaper3k/4k. Returns #enriched."""
    try:
        from newspaper import Article  # type: ignore
    except Exception:
        log.warning("newspaper3k not installed - skipping full-text extraction")
        return 0
    n = 0
    for a in articles[:max_articles]:
        if not a.get("url") or len(a.get("content", "")) > 600:
            continue
        try:
            art = Article(a["url"])
            art.download()
            art.parse()
            if art.text and len(art.text) > len(a.get("content", "")):
                a["content"] = art.text[:5000]
                n += 1
        except Exception as exc:
            log.info("full-text failed for %s: %s", a.get("url"), exc)
    return n


# --------------------------------------------------------------------------- static data
def load_sample(shift_to_now: bool = True) -> list[dict]:
    """Load the bundled synthetic demo dataset.

    Timestamps are shifted so the newest article is 'now', which keeps the
    built-in burst / fading patterns meaningful whenever the demo is run.
    The synthetic ground-truth label is kept in `gold_topic` (for evaluation).
    """
    with open(config.SAMPLE_PATH, encoding="utf-8") as fh:
        rows = json.load(fh)
    out = [normalize_article(origin="sample", **{k: r.get(k, "") for k in (
        "title", "description", "content", "source", "url", "published_at", "language")})
        for r in rows]
    for a, r in zip(out, rows):
        a["gold_topic"] = r.get("_topic", "")
    if shift_to_now and out:
        newest = max(datetime.fromisoformat(a["published_at"]) for a in out)
        delta = datetime.now(timezone.utc) - newest
        for a in out:
            a["published_at"] = (datetime.fromisoformat(a["published_at"]) + delta).isoformat()
    return out


def load_records(records: Iterable[dict], origin: str = "upload") -> list[dict]:
    """Normalise arbitrary user rows (CSV/JSON upload). Accepts common column names."""
    aliases = {
        "title": ("title", "headline", "Title"),
        "description": ("description", "summary", "abstract", "Description"),
        "content": ("content", "text", "body", "Content"),
        "source": ("source", "publisher", "Source"),
        "url": ("url", "link", "URL"),
        "published_at": ("published_at", "publishedAt", "date", "timestamp", "Date"),
    }
    out = []
    for r in records:
        kw = {}
        for k, names in aliases.items():
            kw[k] = next((r[n] for n in names if n in r and r[n] == r[n] and r[n] is not None), "")
        if isinstance(kw["source"], dict):
            kw["source"] = kw["source"].get("name", "")
        if not kw["title"] and kw["content"]:
            kw["title"] = str(kw["content"])[:90]
        if kw["title"]:
            out.append(normalize_article(origin=origin, **kw))
    return out


# --------------------------------------------------------------------------- orchestration
@dataclass
class FetchReport:
    articles: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def fetch_all(sources: list[str], *, query: str = "", language: str = "en",
              limit: int = 50, hours: int | None = 48, rss_feeds: dict[str, str] | None = None,
              newsapi_keys: list[str] | None = None, gnews_keys: list[str] | None = None,
              country: str = "IN", full_text: bool = False) -> FetchReport:
    """Fetch from every selected source. Never raises; errors are collected."""
    t0 = time.perf_counter()
    rep = FetchReport()
    from_date = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    session = _session()
    jobs: dict[str, Callable[[], list[dict]]] = {}
    if "google_news" in sources:
        jobs["Google News RSS"] = lambda: fetch_google_news(query, language=language, country=country,
                                                            limit=limit, session=session)
    if "rss" in sources:
        def _rss():
            feeds = rss_feeds or config.RSS_FEEDS
            arts, errs = fetch_rss(feeds, query=query, limit_per_feed=max(5, limit // 2), session=session)
            rep.errors.extend(errs)
            for name in feeds:   # per-feed status (so the UI can say "9 of 13 sources working")
                rep.per_source[name] = sum(1 for a in arts if a.get("feed") == name)
            return arts
        jobs["RSS feeds"] = _rss
    if "newsapi" in sources:
        jobs["NewsAPI"] = lambda: fetch_newsapi(query, language=language, page_size=limit,
                                                from_date=from_date, keys=newsapi_keys, session=session)
    if "gnews" in sources:
        jobs["GNews"] = lambda: fetch_gnews(query, language=language, max_results=limit,
                                            from_date=from_date, keys=gnews_keys, session=session)
    if "sample" in sources:
        jobs["Sample dataset"] = load_sample
    for name, job in jobs.items():
        try:
            arts = job()
            if name != "RSS feeds":
                rep.per_source[name] = len(arts)
            rep.articles.extend(arts)
        except Exception as exc:
            if name != "RSS feeds":
                rep.per_source[name] = 0
            rep.errors.append(f"{name}: {exc}")
            log.error("%s failed: %s", name, exc)
    # de-duplicate within the batch
    seen: set[str] = set()
    uniq = []
    for a in rep.articles:
        if a["id"] in seen:
            continue
        seen.add(a["id"])
        uniq.append(a)
    rep.articles = uniq
    if full_text and rep.articles:
        extract_full_text(rep.articles)
    rep.seconds = time.perf_counter() - t0
    log.info("fetch_all: %d unique articles from %s in %.2fs", len(uniq), list(jobs), rep.seconds)
    return rep
