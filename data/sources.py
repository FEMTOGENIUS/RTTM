"""News source registry: which feeds to read for every edition (language x region)
and every section, so that each section page always has its own coverage.

Every feed carries a `section` hint (one of the topic categories). Articles keep
that hint, which (a) guarantees section pages have stories and (b) gives topic
labels that work in any language.

Feeds that are down, rate-limited or blocked are skipped automatically and
reported in Topic Analytics -> System; the rest keep working.
"""
from __future__ import annotations

from urllib.parse import quote_plus

# section hint values == topic categories used everywhere in the app
POL, WORLD, BIZ, TECH, SPORT, HEALTH, ENT = ("Politics & Elections", "Geopolitics & Conflict", "Business & Markets",
                                            "Technology", "Sports", "Health", "Entertainment")
SCI, CLIM, LAW, START, ENERGY = ("Science & Space", "Climate & Disaster", "Law & Justice",
                                 "Startups & Funding", "Energy")

LANGUAGES = {"English": "en", "Hindi": "hi", "French": "fr", "German": "de", "Spanish": "es"}
REGIONS = {"India": "IN", "United States": "US", "United Kingdom": "GB", "Australia": "AU", "Canada": "CA",
           "Singapore": "SG", "France": "FR", "Germany": "DE", "Spain": "ES"}

# Google News editions (hl, gl, ceid). Unsupported combinations fall back to the language default.
_GN = {("en", "IN"): ("en-IN", "IN", "IN:en"), ("en", "US"): ("en-US", "US", "US:en"),
       ("en", "GB"): ("en-GB", "GB", "GB:en"), ("en", "AU"): ("en-AU", "AU", "AU:en"),
       ("en", "CA"): ("en-CA", "CA", "CA:en"), ("en", "SG"): ("en-SG", "SG", "SG:en"),
       ("hi", "IN"): ("hi", "IN", "IN:hi"), ("fr", "FR"): ("fr", "FR", "FR:fr"), ("fr", "CA"): ("fr-CA", "CA", "CA:fr"),
       ("de", "DE"): ("de", "DE", "DE:de"), ("es", "ES"): ("es", "ES", "ES:es"), ("es", "US"): ("es-419", "US", "US:es-419")}
_GN_DEFAULT = {"en": ("en", "US"), "hi": ("hi", "IN"), "fr": ("fr", "FR"), "de": ("de", "DE"), "es": ("es", "ES")}

_GN_TOPICS = {"WORLD": WORLD, "NATION": POL, "BUSINESS": BIZ, "TECHNOLOGY": TECH, "ENTERTAINMENT": ENT,
              "SPORTS": SPORT, "SCIENCE": SCI, "HEALTH": HEALTH}

# sections without a Google topic feed -> Google News search, in each language
_GN_SEARCH = {
    CLIM: {"en": "climate OR weather OR cyclone OR flood", "hi": "मौसम OR जलवायु OR बाढ़",
           "fr": "climat OR météo OR inondations", "de": "Klima OR Unwetter OR Hochwasser",
           "es": "clima OR temporal OR inundaciones"},
    LAW: {"en": "court OR verdict OR judge", "hi": "अदालत OR सुप्रीम कोर्ट OR फैसला",
          "fr": "tribunal OR justice OR procès", "de": "Gericht OR Urteil OR Prozess",
          "es": "tribunal OR juicio OR sentencia"},
    ENERGY: {"en": "energy OR oil prices OR renewable", "hi": "ऊर्जा OR बिजली OR पेट्रोल",
             "fr": "énergie OR pétrole OR électricité", "de": "Energie OR Strompreis OR Ölpreis",
             "es": "energía OR petróleo OR electricidad"},
    START: {"en": "startup funding OR IPO", "hi": "स्टार्टअप OR आईपीओ", "fr": "start-up OR levée de fonds",
            "de": "Startup OR Finanzierungsrunde", "es": "startup OR ronda de financiación"},
    POL: {"en": "election OR parliament OR minister", "hi": "चुनाव OR संसद OR मंत्री",
          "fr": "élection OR gouvernement OR ministre", "de": "Wahl OR Bundestag OR Minister",
          "es": "elecciones OR gobierno OR ministro"},
}


def _f(name, url, section=None):
    return {"name": name, "url": url, "section": section}


# ------------------------------------------------------------------ publishers (English, international)
INTERNATIONAL_EN = [
    _f("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", WORLD),
    _f("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml", BIZ),
    _f("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml", TECH),
    _f("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", SCI),
    _f("BBC Health", "https://feeds.bbci.co.uk/news/health/rss.xml", HEALTH),
    _f("BBC Entertainment", "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml", ENT),
    _f("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml", SPORT),
    _f("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", WORLD),
    _f("Guardian World", "https://www.theguardian.com/world/rss", WORLD),
    _f("Guardian Business", "https://www.theguardian.com/business/rss", BIZ),
    _f("Guardian Technology", "https://www.theguardian.com/technology/rss", TECH),
    _f("Guardian Science", "https://www.theguardian.com/science/rss", SCI),
    _f("Guardian Environment", "https://www.theguardian.com/environment/rss", CLIM),
    _f("Guardian Law", "https://www.theguardian.com/law/rss", LAW),
    _f("Guardian Sport", "https://www.theguardian.com/sport/rss", SPORT),
    _f("Guardian Culture", "https://www.theguardian.com/culture/rss", ENT),
    _f("NPR News", "https://feeds.npr.org/1001/rss.xml"),
    _f("NPR Politics", "https://feeds.npr.org/1014/rss.xml", POL),
    _f("NPR Science", "https://feeds.npr.org/1007/rss.xml", SCI),
    _f("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", WORLD),
    _f("NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", BIZ),
    _f("NYT Technology", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", TECH),
    _f("NYT Health", "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml", HEALTH),
    _f("NYT Climate", "https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml", CLIM),
    _f("DW English", "https://rss.dw.com/rdf/rss-en-all", WORLD),
    _f("France 24 English", "https://www.france24.com/en/rss", WORLD),
    _f("CNN", "http://rss.cnn.com/rss/edition.rss"),
    _f("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", BIZ),
    _f("TechCrunch", "https://techcrunch.com/feed/", START),
    _f("The Verge", "https://www.theverge.com/rss/index.xml", TECH),
    _f("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index", TECH),
    _f("ScienceDaily", "https://www.sciencedaily.com/rss/all.xml", SCI),
    _f("WHO News", "https://www.who.int/rss-feeds/news-english.xml", HEALTH),
    _f("Variety", "https://variety.com/feed/", ENT),
    _f("OilPrice", "https://oilprice.com/rss/main", ENERGY),
    _f("Carbon Brief", "https://www.carbonbrief.org/feed", CLIM),
    _f("ESPN", "https://www.espn.com/espn/rss/news", SPORT),
]

# ------------------------------------------------------------------ publishers by region (English)
REGIONAL_EN = {
    "IN": [
        _f("The Hindu National", "https://www.thehindu.com/news/national/feeder/default.rss", POL),
        _f("The Hindu International", "https://www.thehindu.com/news/international/feeder/default.rss", WORLD),
        _f("The Hindu Business", "https://www.thehindu.com/business/feeder/default.rss", BIZ),
        _f("The Hindu Sci-Tech", "https://www.thehindu.com/sci-tech/feeder/default.rss", TECH),
        _f("The Hindu Sport", "https://www.thehindu.com/sport/feeder/default.rss", SPORT),
        _f("The Hindu Entertainment", "https://www.thehindu.com/entertainment/feeder/default.rss", ENT),
        _f("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
        _f("TOI India", "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms", POL),
        _f("TOI Business", "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms", BIZ),
        _f("TOI Sports", "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms", SPORT),
        _f("TOI Tech", "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms", TECH),
        _f("TOI World", "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", WORLD),
        _f("TOI Entertainment", "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms", ENT),
        _f("Hindustan Times India", "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", POL),
        _f("Hindustan Times World", "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml", WORLD),
        _f("Hindustan Times Business", "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml", BIZ),
        _f("Hindustan Times Sports", "https://www.hindustantimes.com/feeds/rss/sports/rssfeed.xml", SPORT),
        _f("NDTV Top Stories", "https://feeds.feedburner.com/ndtvnews-top-stories"),
        _f("Economic Times", "https://economictimes.indiatimes.com/rssfeedstopstories.cms", BIZ),
        _f("Mint News", "https://www.livemint.com/rss/news"),
        _f("Mint Markets", "https://www.livemint.com/rss/markets", BIZ),
        _f("Mint Technology", "https://www.livemint.com/rss/technology", TECH),
        _f("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml", BIZ),
        _f("India Today", "https://www.indiatoday.in/rss/home"),
        _f("Indian Express", "https://indianexpress.com/feed/"),
    ],
    "US": [
        _f("NPR World", "https://feeds.npr.org/1004/rss.xml", WORLD),
        _f("NPR Business", "https://feeds.npr.org/1006/rss.xml", BIZ),
        _f("NYT Politics", "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml", POL),
        _f("NYT Sports", "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", SPORT),
        _f("NYT Arts", "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml", ENT),
        _f("ABC News", "https://feeds.abcnews.com/abcnews/topstories"),
        _f("Fox News", "https://moxie.foxnews.com/google-publisher/latest.xml"),
        _f("SCOTUSblog", "https://www.scotusblog.com/feed/", LAW),
    ],
    "GB": [
        _f("BBC UK", "https://feeds.bbci.co.uk/news/uk/rss.xml"),
        _f("BBC Politics", "https://feeds.bbci.co.uk/news/politics/rss.xml", POL),
        _f("Guardian UK", "https://www.theguardian.com/uk-news/rss"),
        _f("Guardian Politics", "https://www.theguardian.com/politics/rss", POL),
        _f("Sky News", "https://feeds.skynews.com/feeds/rss/home.xml"),
        _f("Sky News World", "https://feeds.skynews.com/feeds/rss/world.xml", WORLD),
        _f("The Independent", "https://www.independent.co.uk/news/rss"),
    ],
    "AU": [
        _f("ABC Australia", "https://www.abc.net.au/news/feed/51120/rss.xml"),
        _f("Guardian Australia", "https://www.theguardian.com/australia-news/rss", POL),
        _f("Sydney Morning Herald", "https://www.smh.com.au/rss/feed.xml"),
    ],
    "CA": [
        _f("CBC Top Stories", "https://www.cbc.ca/webfeed/rss/rss-topstories"),
        _f("CBC World", "https://www.cbc.ca/webfeed/rss/rss-world", WORLD),
        _f("CBC Business", "https://www.cbc.ca/webfeed/rss/rss-business", BIZ),
        _f("CBC Technology", "https://www.cbc.ca/webfeed/rss/rss-technology", TECH),
        _f("CBC Sports", "https://www.cbc.ca/webfeed/rss/rss-sports", SPORT),
        _f("CBC Health", "https://www.cbc.ca/webfeed/rss/rss-health", HEALTH),
        _f("CBC Politics", "https://www.cbc.ca/webfeed/rss/rss-politics", POL),
        _f("Global News", "https://globalnews.ca/feed/"),
    ],
    "SG": [
        _f("CNA Latest", "https://www.channelnewsasia.com/rssfeeds/8395986"),
        _f("Straits Times Singapore", "https://www.straitstimes.com/news/singapore/rss.xml", POL),
        _f("Straits Times World", "https://www.straitstimes.com/news/world/rss.xml", WORLD),
        _f("Straits Times Business", "https://www.straitstimes.com/news/business/rss.xml", BIZ),
    ],
}

# ------------------------------------------------------------------ publishers by language (non-English)
LANGUAGE_PUBLISHERS = {
    "hi": [
        _f("BBC Hindi", "https://feeds.bbci.co.uk/hindi/rss.xml"),
        _f("Navbharat Times", "https://navbharattimes.indiatimes.com/rssfeedsdefault.cms"),
        _f("Amar Ujala", "https://www.amarujala.com/rss/breaking-news.xml"),
        _f("NDTV India", "https://feeds.feedburner.com/ndtvkhabar-latest"),
        _f("Dainik Jagran", "https://www.jagran.com/rss/news/national.xml", POL),
    ],
    "fr": [
        _f("Le Monde à la une", "https://www.lemonde.fr/rss/une.xml"),
        _f("Le Monde International", "https://www.lemonde.fr/international/rss_full.xml", WORLD),
        _f("Le Monde Politique", "https://www.lemonde.fr/politique/rss_full.xml", POL),
        _f("Le Monde Économie", "https://www.lemonde.fr/economie/rss_full.xml", BIZ),
        _f("Le Monde Pixels", "https://www.lemonde.fr/pixels/rss_full.xml", TECH),
        _f("Le Monde Sport", "https://www.lemonde.fr/sport/rss_full.xml", SPORT),
        _f("Le Monde Planète", "https://www.lemonde.fr/planete/rss_full.xml", CLIM),
        _f("Le Monde Sciences", "https://www.lemonde.fr/sciences/rss_full.xml", SCI),
        _f("Le Monde Santé", "https://www.lemonde.fr/sante/rss_full.xml", HEALTH),
        _f("Le Monde Culture", "https://www.lemonde.fr/culture/rss_full.xml", ENT),
        _f("France 24", "https://www.france24.com/fr/rss", WORLD),
        _f("RFI", "https://www.rfi.fr/fr/rss"),
        _f("BBC Afrique", "https://feeds.bbci.co.uk/afrique/rss.xml"),
    ],
    "de": [
        _f("Tagesschau", "https://www.tagesschau.de/xml/rss2"),
        _f("Spiegel Schlagzeilen", "https://www.spiegel.de/schlagzeilen/index.rss"),
        _f("Spiegel Politik", "https://www.spiegel.de/politik/index.rss", POL),
        _f("Spiegel Wirtschaft", "https://www.spiegel.de/wirtschaft/index.rss", BIZ),
        _f("Spiegel Netzwelt", "https://www.spiegel.de/netzwelt/index.rss", TECH),
        _f("Spiegel Sport", "https://www.spiegel.de/sport/index.rss", SPORT),
        _f("Spiegel Wissenschaft", "https://www.spiegel.de/wissenschaft/index.rss", SCI),
        _f("Spiegel Gesundheit", "https://www.spiegel.de/gesundheit/index.rss", HEALTH),
        _f("Spiegel Kultur", "https://www.spiegel.de/kultur/index.rss", ENT),
        _f("DW Deutsch", "https://rss.dw.com/xml/rss-de-all", WORLD),
        _f("Zeit Online", "https://newsfeed.zeit.de/index"),
    ],
    "es": [
        _f("BBC Mundo", "https://feeds.bbci.co.uk/mundo/rss.xml", WORLD),
        _f("El País", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"),
        _f("El Mundo", "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml"),
        _f("DW Español", "https://rss.dw.com/xml/rss-es-all", WORLD),
        _f("France 24 Español", "https://www.france24.com/es/rss", WORLD),
    ],
}

GROUPS = {
    "google_news": "Google News · top stories",
    "google_sections": "Google News · every section",
    "publishers": "Publishers for this edition",
    "international": "International publishers",
}


def google_edition(lang: str, region: str) -> tuple[str, str, str]:
    if (lang, region) in _GN:
        return _GN[(lang, region)]
    lang_default = _GN_DEFAULT.get(lang, ("en", "US"))
    return _GN.get(lang_default, ("en-US", "US", "US:en"))


def edition_label(lang: str, region: str) -> str:
    hl, gl, _ = google_edition(lang, region)
    names = {v: k for k, v in LANGUAGES.items()}
    regions = {v: k for k, v in REGIONS.items()}
    return f"{names.get(lang, lang)} · {regions.get(gl, gl)}"


def edition_key(lang: str, region: str) -> str:
    return f"{lang}-{region}"


def publisher_feeds(lang: str, region: str) -> list[dict]:
    if lang == "en":
        return list(REGIONAL_EN.get(region, []))
    return list(LANGUAGE_PUBLISHERS.get(lang, []))


def build_feeds(groups: list[str], lang: str, region: str, query: str = "",
                publishers: list[str] | None = None) -> list[dict]:
    """Return feed specs: {name, url, section, kind ('google' | 'rss')}."""
    hl, gl, ceid = google_edition(lang, region)
    tail = f"hl={hl}&gl={gl}&ceid={ceid}"
    feeds: list[dict] = []
    if "google_news" in groups:
        url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:2d&{tail}" if query
               else f"https://news.google.com/rss?{tail}")
        feeds.append({"name": "Google News", "url": url, "section": None, "kind": "google"})
    if "google_sections" in groups:
        for topic, sec in _GN_TOPICS.items():
            if query:   # topic feeds can't be searched -> keyword search inside the section language
                continue
            feeds.append({"name": f"Google News {topic.title()}", "kind": "google", "section": sec,
                          "url": f"https://news.google.com/rss/headlines/section/topic/{topic}?{tail}"})
        for sec, words in _GN_SEARCH.items():
            q = words.get(lang, words["en"])
            if query:
                q = f"{query} ({q})"
            feeds.append({"name": f"Google News · {sec}", "kind": "google", "section": sec,
                          "url": f"https://news.google.com/rss/search?q={quote_plus(q)}+when:2d&{tail}"})
    if "publishers" in groups:
        for f in publisher_feeds(lang, region):
            if publishers is None or f["name"] in publishers:
                feeds.append({**f, "kind": "rss"})
    if "international" in groups and lang == "en":
        for f in INTERNATIONAL_EN:
            if publishers is None or f["name"] in publishers:
                feeds.append({**f, "kind": "rss"})
    seen, out = set(), []
    for f in feeds:
        if f["url"] not in seen:
            seen.add(f["url"])
            out.append(f)
    return out
