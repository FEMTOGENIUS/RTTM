"""Human-readable topic labels.

Extends the TOPIC_CATEGORIES dictionary used in the original prototype
(business / technology / geopolitics / healthcare / sports / entertainment)
with politics, climate, startups and science. A topic is labelled with the
category whose keyword set best overlaps its top words (weighted by rank).
"""
from __future__ import annotations

TOPIC_CATEGORIES: dict[str, set[str]] = {
    "Technology": {"ai", "artificial", "intelligence", "tech", "technology", "software", "digital",
                   "data", "chip", "app", "cyber", "cybersecurity", "internet", "computer", "robot",
                   "machine", "learning", "algorithm", "google", "apple", "microsoft", "openai",
                   "smartphone", "cloud", "semiconductor", "deepfake", "chatbot", "privacy", "computing",
                   "online", "platform", "device", "nvidia", "meta", "model", "generator", "agent"},
    "Business & Markets": {"market", "stock", "share", "investor", "economy", "economic", "inflation",
                           "bank", "rate", "interest", "rupee", "dollar", "trade", "price", "oil", "gdp",
                           "growth", "earning", "earnings", "profit", "revenue", "sensex", "nifty",
                           "index", "bond", "yield", "tariff", "tax", "budget", "fiscal", "recession",
                           "currency", "equity", "trader", "fed", "reserve", "company", "business"},
    "Startups & Funding": {"startup", "funding", "unicorn", "venture", "ipo", "valuation", "founder",
                           "investment", "capital", "raise", "raised", "round", "fintech", "layoff",
                           "listing", "backed", "anthropic", "acquisition", "deal", "stake"},
    "Politics & Elections": {"election", "vote", "voter", "poll", "polling", "campaign", "party",
                             "minister", "government", "parliament", "opposition", "candidate",
                             "president", "congress", "senate", "bjp", "democrat", "republican",
                             "policy", "bill", "law", "court", "constituency", "turnout", "manifesto",
                             "assembly", "recount", "commission", "ruling", "rally", "lawmaker",
                             "modi", "gandhi", "rahul", "dmk", "tvk", "aap", "mla", "cm", "chief",
                             "bloc", "eci", "politician", "leader", "remark", "row", "sir"},
    "Geopolitics & Conflict": {"war", "military", "ukraine", "russia", "israel", "gaza", "iran", "china",
                               "nato", "ceasefire", "attack", "strike", "missile", "drone", "sanction",
                               "troop", "conflict", "border", "diplomat", "diplomatic", "un", "security",
                               "army", "killed", "hamas", "taiwan", "summit", "peace", "civilian",
                               "frontline", "defence", "defense", "humanitarian", "aid", "fighting",
                               "quad", "asean", "apec", "trump", "kim", "korea", "pakistan", "bilateral",
                               "trilateral", "foreign", "minister", "nuclear", "radar", "bahrain"},
    "Health": {"health", "hospital", "vaccine", "vaccination", "virus", "disease", "patient", "doctor",
               "medical", "covid", "outbreak", "drug", "cancer", "treatment", "clinical", "dengue",
               "measles", "malaria", "influenza", "who", "healthcare", "infection", "death"},
    "Climate & Disaster": {"climate", "cyclone", "storm", "flood", "flooding", "rain", "weather",
                           "earthquake", "wildfire", "heat", "hurricane", "disaster", "emission",
                           "carbon", "warming", "evacuate", "evacuated", "landfall", "relief",
                           "environment", "ocean", "rescue", "outage", "alert"},
    "Sports": {"match", "cricket", "football", "team", "player", "cup", "tournament", "win", "wicket",
               "captain", "league", "goal", "coach", "series", "olympic", "championship", "final",
               "squad", "stadium", "century", "bowler", "batter", "fan", "sport", "game",
               "dhoni", "kohli", "ipl", "csk", "bcci", "fifa", "messi", "ronaldo", "nba", "tennis",
               "retirement", "olympics", "medal"},
    "Entertainment": {"film", "movie", "music", "actor", "actress", "celebrity", "box", "office",
                      "festival", "award", "album", "song", "bollywood", "hollywood", "premiere",
                      "director", "cinema", "crore", "carpet", "critic", "show"},
    "Law & Justice": {"court", "supreme", "bail", "convict", "sentence", "judge", "verdict", "petition",
                      "police", "arrest", "crime", "murder", "rape", "gangrape", "accused", "trial",
                      "hearing", "justice", "lawyer", "fir", "probe", "investigation", "killed", "case"},
    "Energy": {"fuel", "renewable", "solar", "wind", "energy", "oil", "gas", "power", "battery",
               "refinery", "aviation", "crude", "electricity", "coal", "hydrogen", "nuclear", "grid"},
    "Science & Space": {"space", "nasa", "isro", "rocket", "satellite", "moon", "mars", "scientist",
                        "research", "study", "discovery", "astronaut", "orbit", "telescope"},
}

ICONS = {"Technology": "💻", "Business & Markets": "📈", "Startups & Funding": "🚀",
         "Politics & Elections": "🗳️", "Geopolitics & Conflict": "🌍", "Health": "🏥",
         "Climate & Disaster": "🌪️", "Sports": "🏏", "Entertainment": "🎬",
         "Science & Space": "🔭", "Law & Justice": "⚖️", "Energy": "⚡", "General": "📰"}


def categorize(words: list[str]) -> tuple[str, float]:
    """Return (category, confidence 0..1) for an ordered keyword list."""
    scores = {c: 0.0 for c in TOPIC_CATEGORIES}
    total = 0.0
    for rank, w in enumerate(words[:10]):
        weight = 1.0 / (1 + 0.25 * rank)
        total += weight
        for u in w.split():
            for c, vocab in TOPIC_CATEGORIES.items():
                if u in vocab:
                    scores[c] += weight
    best = max(scores, key=scores.get)
    conf = scores[best] / total if total else 0.0
    return (best, conf) if conf >= 0.15 else ("General", conf)


def topic_label(topic_id: int, words: list[str], n_kw: int = 3) -> dict:
    cat, conf = categorize(words)
    kw = ", ".join(words[:n_kw])
    return {"category": cat, "confidence": round(conf, 3), "icon": ICONS.get(cat, "📰"),
            "keywords": kw, "name": f"Topic {topic_id} ({cat})",
            "short": f"T{topic_id}: {kw}"}
