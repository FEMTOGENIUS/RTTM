"""Source registry, multilingual preprocessing, edition tagging, RDF feeds."""
from datetime import datetime, timedelta, timezone

import pytest

from data import fetchers as F
from data import sources as S
from models.engine import run_pipeline
from utils.preprocess import preprocess_articles, tokenize

SECTION_CATS = [S.POL, S.WORLD, S.BIZ, S.TECH, S.SPORT, S.HEALTH, S.ENT, S.SCI, S.CLIM, S.LAW, S.START, S.ENERGY]


@pytest.mark.parametrize("lang,region", [("en", "IN"), ("en", "US"), ("en", "GB"), ("en", "AU"), ("en", "CA"),
                                         ("en", "SG"), ("hi", "IN"), ("fr", "FR"), ("de", "DE"), ("es", "ES"),
                                         ("es", "US"), ("fr", "IN")])
def test_every_section_has_feeds_in_every_edition(lang, region):
    feeds = S.build_feeds(["google_sections", "publishers"], lang, region)
    covered = {f["section"] for f in feeds if f["section"]}
    assert set(SECTION_CATS) <= covered, set(SECTION_CATS) - covered
    hl, gl, ceid = S.google_edition(lang, region)
    assert all(f"hl={hl}" in f["url"] for f in feeds if f["kind"] == "google")
    assert hl.startswith(lang)


def test_regions_and_languages_change_the_sources():
    names = lambda l, r: {f["name"] for f in S.build_feeds(["publishers"], l, r)}  # noqa: E731
    assert "The Hindu National" in names("en", "IN") and "The Hindu National" not in names("en", "US")
    assert "CBC World" in names("en", "CA") and "Le Monde à la une" in names("fr", "FR")
    assert "BBC Hindi" in names("hi", "IN") and "Tagesschau" in names("de", "DE") and "El País" in names("es", "ES")
    assert [f["name"] for f in S.build_feeds(["google_news"], "en", "IN")] == ["Google News"]
    q = S.build_feeds(["google_sections"], "fr", "FR", query="Macron")
    assert all("Macron" in f["url"] for f in q)


def test_publisher_subset():
    feeds = S.build_feeds(["publishers"], "en", "IN", publishers=["The Hindu Sport"])
    assert [f["name"] for f in feeds] == ["The Hindu Sport"]


def test_fetch_all_tags_edition_section_and_counts(monkeypatch):
    def fake_rss(feeds, query="", limit_per_feed=50, session=None):
        (name, url), = feeds.items()
        if "bad" in name.lower() or "hindustan" in name.lower():
            return [], [f"RSS '{name}' failed: HTTP 403"]
        return [F.normalize_article(title=f"{name} story {i}", url=f"{url}#{i}", origin="rss") for i in range(3)], []

    monkeypatch.setattr(F, "fetch_rss", fake_rss)
    rep = F.fetch_all(["google_sections", "publishers"], language="en", country="IN", limit=10)
    assert rep.articles and all(a["edition"] == "en-IN" for a in rep.articles)
    sections = {a["section"] for a in rep.articles}
    assert {S.SPORT, S.LAW, S.CLIM, S.ENERGY} <= sections
    assert any(v == 0 for v in rep.per_source.values()) and any("403" in e for e in rep.errors)
    assert len(rep.per_source) > 20


def test_rdf_rss1_feed_parsing():
    xml = b"""<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel><title>DW</title></channel>
      <item><title>Germany passes budget</title><link>https://dw.example/1</link>
        <description>Parliament approved the budget.</description><dc:date>2026-11-17T08:00:00Z</dc:date></item>
    </rdf:RDF>"""
    items = F.parse_feed_xml(xml, "DW")
    assert items[0]["title"] == "Germany passes budget" and items[0]["published_at"] == "2026-11-17T08:00:00Z"


def test_multilingual_tokenisation():
    assert "तारीखों" in tokenize("चुनाव आयोग ने मतदान की तारीखों की घोषणा की", "hi")
    assert "की" not in tokenize("चुनाव आयोग ने मतदान की तारीखों की घोषणा की", "hi")
    assert tokenize("Le gouvernement annonce un plan", "fr") == ["gouvernement", "annonce", "plan"]
    assert "schließt" in tokenize("Die Börse schließt mit Verlusten", "de")


def _corpus(lang):
    base = {
        "fr": [("Le Parlement vote le budget de l'État", S.POL), ("Inflation et taux de la Banque centrale", S.BIZ),
               ("Le PSG gagne le match de football", S.SPORT), ("Tempête et inondations dans le sud", S.CLIM)],
        "hi": [("संसद में बजट पर चुनाव बहस", S.POL), ("शेयर बाजार में सेंसेक्स की तेजी", S.BIZ),
               ("क्रिकेट मैच में भारत की जीत", S.SPORT), ("बाढ़ और भारी बारिश से नुकसान", S.CLIM)],
    }[lang]
    extra = {"fr": "lundi mardi matin soir ville région pays national public officiel nouveau grand petit "
                   "premier dernier semaine",
             "hi": "सोमवार मंगलवार सुबह शाम शहर राज्य देश राष्ट्रीय सरकारी नया बड़ा छोटा पहला अंतिम सप्ताह"}[lang].split()
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(60):
        title, sec = base[i % 4]
        w1, w2 = extra[(i // 4) % len(extra)], extra[(i // 4 + 5) % len(extra)]
        a = F.normalize_article(title=f"{title} {w1} {w2}", description=f"{title} {w2}", url=f"u{i}",
                                published_at=now - timedelta(hours=i))
        a["section"], a["edition"] = sec, f"{lang}-X"
        rows.append(a)
    return rows


@pytest.mark.parametrize("lang", ["fr", "hi"])
def test_non_english_pipeline_and_section_labels(lang):
    df, stats = preprocess_articles(_corpus(lang), language=lang)
    assert len(df) >= 40 and stats["non_english"] == 0
    out = run_pipeline(_corpus(lang), models=["BERTopic", "LDA"], n_topics=4, language=lang)
    assert set(out.results) == {"BERTopic", "LDA"} and not out.warnings
    cats = {lab["category"] for lab in out.labels["BERTopic"].values()}
    assert cats & {S.POL, S.BIZ, S.SPORT, S.CLIM}          # labels come from feed sections, not English keywords
