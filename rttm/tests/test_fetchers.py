from datetime import datetime, timezone

import pytest

from data import fetchers as F
from tests.conftest import FakeResp, FakeSession


def test_strip_html_and_newsapi_truncation():
    assert F.strip_html("<p>Hello &amp; <b>world</b></p> [+1234 chars]") == "Hello & world"


def test_article_id_is_headline_hash():
    a = F.article_id("Markets RALLY!", "u1")
    b = F.article_id("markets rally", "u2")
    assert a == b and len(a) == 16


@pytest.mark.parametrize("raw", ["Tue, 17 Nov 2026 03:36:20 GMT", "2026-11-17T03:36:20Z",
                                 "2026-11-17T09:06:20+05:30"])
def test_parse_date_variants(raw):
    assert F.parse_date(raw).startswith("2026-11-17T03:36:20")


def test_parse_date_garbage_falls_back_to_now():
    ts = datetime.fromisoformat(F.parse_date("not a date"))
    assert abs((datetime.now(timezone.utc) - ts).total_seconds()) < 5


def test_parse_rss_and_atom(fixture_bytes):
    items = F.parse_feed_xml(fixture_bytes("bbc_style.xml"), "Demo")
    assert [i["title"] for i in items][:2] == ["Ceasefire talks resume in Geneva",
                                              "Vaccine drive expands to rural districts"]
    assert items[1]["published_at"] == "2026-11-17T06:30:00Z"
    atom = F.parse_feed_xml(fixture_bytes("atom.xml"), "Atom")
    assert atom[0]["url"] == "https://example.com/space/1"
    assert "cyclone" in atom[0]["description"]


def test_fetch_rss_skips_untitled_filters_query_and_isolates_errors(fixture_bytes):
    sess = FakeSession({"good": FakeResp(200, content=fixture_bytes("bbc_style.xml")),
                        "bad": FakeResp(500, text="boom")})
    arts, errs = F.fetch_rss({"Good": "https://good/feed", "Bad": "https://bad/feed"}, session=sess)
    assert len(arts) == 2 and all(a["origin"] == "rss" for a in arts)
    assert arts[1]["description"] == "Health workers will visit villages door to door."
    assert len(errs) == 1 and "Bad" in errs[0]
    arts, _ = F.fetch_rss({"Good": "https://good/feed"}, query="vaccine", session=sess)
    assert [a["title"] for a in arts] == ["Vaccine drive expands to rural districts"]


def test_google_news_title_split_and_related_headlines(fixture_bytes):
    sess = FakeSession({"news.google.com": FakeResp(200, content=fixture_bytes("google_news_sample.xml"))})
    arts = F.fetch_google_news("cyclone", session=sess)
    assert "q=cyclone" in sess.calls[0]["url"]
    a = arts[0]
    assert a["title"] == "Port city braces for Cyclone Zephyr as evacuation begins"
    assert a["source"] == "Example Times"
    # related headlines from other publishers are kept, the duplicate of the title is removed
    assert "red alert" in a["description"] and "Heavy rain" in a["description"]
    assert "Port city braces" not in a["description"]
    # single-link item: "See more headlines" boiler-plate removed, nothing left
    assert arts[1]["description"] == ""
    assert arts[0]["published_at"].startswith("2026-11-17T03:36:20")


def test_newsapi_normalisation_and_key_rotation():
    payload = {"status": "ok", "articles": [
        {"title": "AI rules agreed", "description": "EU deal", "content": "Long text… [+2000 chars]",
         "source": {"name": "Demo"}, "url": "https://x/1", "publishedAt": "2026-11-17T10:00:00Z"},
        {"title": "[Removed]", "url": "https://x/2"}]}

    def route(params, headers):
        return FakeResp(429, text="rate limited") if headers["X-Api-Key"] == "k1" else FakeResp(200, payload)

    sess = FakeSession({"newsapi.org": route})
    arts = F.fetch_newsapi("ai", keys=["k1", "k2"], session=sess)
    assert len(sess.calls) == 2                         # rotated to the second key
    assert len(arts) == 1 and arts[0]["content"] == "Long text…"
    assert sess.calls[0]["url"].endswith("/everything")


def test_newsapi_without_key_raises():
    with pytest.raises(F.FetchError):
        F.fetch_newsapi("x", keys=[])


def test_gnews_normalisation():
    payload = {"articles": [{"title": "Rupee gains", "description": "d", "content": "c",
                             "url": "https://g/1", "publishedAt": "2026-11-17T01:00:00Z",
                             "source": {"name": "GSrc"}}]}
    sess = FakeSession({"gnews.io": FakeResp(200, payload)})
    arts = F.fetch_gnews("rupee", keys=["k"], session=sess)
    assert arts[0]["source"] == "GSrc" and arts[0]["origin"] == "gnews"
    assert sess.calls[0]["params"]["q"] == "rupee"


def test_fetch_all_never_raises_and_dedupes(monkeypatch):
    monkeypatch.setattr(F, "fetch_google_news", lambda *a, **k: [F.normalize_article(title="Same story", url="1"),
                                                                 F.normalize_article(title="Same  story!", url="2")])
    rep = F.fetch_all(["google_news", "newsapi"], newsapi_keys=[])
    assert len(rep.articles) == 1
    assert rep.per_source["NewsAPI"] == 0 and any("NewsAPI" in e for e in rep.errors)


def test_load_sample_shifts_to_now_and_keeps_gold():
    rows = F.load_sample()
    newest = max(datetime.fromisoformat(r["published_at"]) for r in rows)
    assert abs((datetime.now(timezone.utc) - newest).total_seconds()) < 60
    assert all("gold_topic" in r for r in rows)


def test_load_records_accepts_common_column_names():
    rows = F.load_records([{"headline": "Budget announced", "summary": "Tax cuts", "date": "2026-11-01",
                            "link": "https://b/1", "publisher": "P"},
                           {"text": "Body only article about elections and voters"}])
    assert rows[0]["title"] == "Budget announced" and rows[0]["source"] == "P"
    assert rows[1]["title"].startswith("Body only")
