"""Generate the bundled OFFLINE DEMO dataset (data/sample_articles.json).

The articles are SYNTHETIC (template-generated) so the app works with no
internet/API key and so tests are reproducible. Publishers and URLs are
fictional (example.com). Do not report metrics on this data as research
results - use live RSS/NewsAPI data or 20 Newsgroups (see evaluate.py).

Built-in test conditions:
* a topic that BURSTS in the last 36h (cyclone / climate)  -> trend detection
* a topic that FADES (film festival, mostly 5-7 days ago)   -> fading detection
* exact duplicate headlines                                  -> de-duplication
* non-English articles                                       -> language filter
* blank descriptions / HTML / NewsAPI "[+123 chars]" markers -> robustness
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_articles.json"
rng = random.Random(7)

SOURCES = ["Demo Daily", "Sample Herald", "Example News Network", "Test Tribune",
           "Placeholder Post", "Mock Times"]

T = {
 "ai_tech": dict(
   fills=dict(co=["Nimbus Labs", "Orbitron", "Vectra AI", "Quantix", "Helios Systems"],
              prod=["chatbot", "language model", "AI assistant", "coding agent", "image generator"],
              place=["the EU", "India", "the US", "the UK", "Japan"]),
   titles=["{co} unveils new {prod} to rival bigger tech firms",
           "Regulators in {place} draft rules for artificial intelligence safety",
           "{co} says its {prod} cuts software costs for businesses",
           "Lawmakers question {co} over AI data privacy",
           "{co} chip shortage slows rollout of {prod}",
           "Experts warn of deepfake risk as {prod} tools spread",
           "{place} launches national AI mission to train engineers"],
   desc=["The company said the {prod} uses machine learning and large datasets, raising questions about privacy, chips and regulation.",
         "Officials in {place} want transparency from AI developers on training data, safety testing and algorithm audits.",
         "Analysts said demand for AI chips and cloud computing from software firms keeps rising.",
         "Critics say the {prod} technology could spread misinformation and deepfakes without stronger regulation."]),
 "markets": dict(
   fills=dict(idx=["Sensex", "Nifty", "Dow Jones", "Nasdaq", "FTSE 100"],
              bank=["the central bank", "the Federal Reserve", "the RBI", "the ECB"],
              pct=["1.2%", "0.8%", "2.1%", "1.5%", "0.4%"]),
   titles=["{idx} falls {pct} as inflation data rattles investors",
           "Stocks rally after {bank} holds interest rates steady",
           "{idx} hits record high on strong bank earnings",
           "Investors brace for {bank} decision on interest rates",
           "Rupee weakens as oil prices climb and markets slide",
           "Bond yields rise as inflation stays above target"],
   desc=["Investors sold shares after inflation figures came in above forecasts, pushing bond yields higher.",
         "Traders expect {bank} to keep interest rates unchanged while inflation eases slowly.",
         "Banking and energy stocks led the market as investors weighed earnings and economic growth.",
         "The {idx} index moved {pct} as foreign investors pulled money from equity markets."]),
 "elections": dict(
   fills=dict(state=["Maharashtra", "Bihar", "Texas", "Ontario", "Victoria"],
              party=["the ruling party", "the opposition alliance", "the centrist bloc"]),
   titles=["Voters head to the polls in {state} election",
           "{party} releases manifesto ahead of {state} vote",
           "Exit polls predict close race in {state}",
           "Election commission orders recount in {state} constituency",
           "Turnout surges as {state} votes in assembly election",
           "Campaign rallies intensify before {state} polling day"],
   desc=["Candidates from {party} campaigned on jobs, inflation and welfare as voters queued at polling booths.",
         "The election commission said turnout was high and counting of votes will begin on Friday.",
         "Opposition leaders accused {party} of misusing government resources during the campaign.",
         "Analysts said the vote in {state} is a test for the government before the national election."]),
 "climate_cyclone": dict(  # BURSTING topic
   fills=dict(name=["Cyclone Vayu", "Cyclone Asha", "Storm Mira"],
              region=["Odisha", "Andhra Pradesh", "the Bay of Bengal coast", "West Bengal"]),
   titles=["{name} makes landfall near {region}, thousands evacuated",
           "Heavy rain and flooding as {name} batters {region}",
           "Weather agency issues red alert for {name}",
           "Rescue teams deployed after {name} floods villages in {region}",
           "Power outages hit {region} after {name}",
           "Climate scientists link stronger cyclones to warming seas"],
   desc=["Disaster response teams evacuated coastal villages as {name} brought heavy rain, strong winds and flooding.",
         "The weather department warned fishermen and said the cyclone would weaken after landfall in {region}.",
         "Scientists said warming ocean temperatures due to climate change make cyclones more intense.",
         "Relief camps were set up for thousands of families displaced by flooding in {region}."]),
 "health": dict(
   fills=dict(dis=["dengue", "measles", "influenza", "malaria", "a new virus strain"],
              place=["Delhi", "Kerala", "Brazil", "Nigeria", "the Philippines"]),
   titles=["Hospitals in {place} report surge in {dis} cases",
           "Health ministry launches vaccine drive against {dis}",
           "WHO urges vaccination as {dis} spreads in {place}",
           "Doctors warn of hospital bed shortage amid {dis} outbreak",
           "New vaccine shows strong results in clinical trial"],
   desc=["Health officials said hospitals were treating more patients with {dis} and urged people to get vaccinated.",
         "The vaccination campaign aims to reach children and elderly patients in {place} within weeks.",
         "Doctors said early treatment and testing were key to reducing deaths from {dis}.",
         "The clinical trial data showed the vaccine was safe and reduced hospital admissions."]),
 "cricket_sports": dict(
   fills=dict(team=["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand"],
              player=["the captain", "the opening batter", "the young fast bowler", "the spinner"]),
   titles=["{team} beat {team2} by six wickets in thrilling final over",
           "{player} scores century as {team} dominate test match",
           "{team} name squad for World Cup tournament",
           "Rain halts play as {team} face {team2} in series decider",
           "{team} coach praises {player} after series win"],
   desc=["{player} hit a century and the bowlers took early wickets as {team} won the match.",
         "Selectors named a strong squad for the tournament, with {player} returning from injury.",
         "Fans packed the stadium as {team} chased the target in the final over of the match.",
         "The series victory lifts {team} in the championship rankings ahead of the World Cup."]),
 "geopolitics": dict(
   fills=dict(a=["Ukraine", "Russia", "NATO", "Israel", "Iran", "China", "Taiwan"],
              city=["Geneva", "Brussels", "Istanbul", "Doha"]),
   titles=["Talks in {city} seek ceasefire between {a} and {b}",
           "NATO leaders pledge military aid for Ukraine",
           "{a} accuses {b} of drone strikes on border towns",
           "UN Security Council meets over {a} conflict",
           "Diplomats in {city} discuss sanctions on {a}",
           "Civilians flee as fighting escalates near frontline"],
   desc=["Diplomats said ceasefire negotiations in {city} remained difficult as both sides traded accusations.",
         "The military aid package includes air defence systems and drones, officials said.",
         "The UN warned of a humanitarian crisis as civilians fled the conflict and missile strikes.",
         "Western governments announced new sanctions targeting {a} defence and energy sectors."]),
 "startups": dict(
   fills=dict(st=["PayNest", "Kartly", "EduSpark", "GreenGrid", "MediQuick", "FarmLink"],
              amt=["$50 million", "$120 million", "$15 million", "$300 million"],
              vc=["venture capital firms", "a sovereign fund", "global investors"]),
   titles=["{st} raises {amt} in funding round led by {vc}",
           "{st} files for IPO as startup valuations rebound",
           "Fintech startup {st} becomes India's newest unicorn",
           "{st} lays off staff as funding winter continues",
           "{st} IPO oversubscribed on first day of bidding"],
   desc=["The startup will use the {amt} funding to expand its platform and hire engineers, its founders said.",
         "The IPO values {st} at over a billion dollars, making it a unicorn backed by {vc}.",
         "Venture capital investment in startups is recovering after two years of weak funding.",
         "Analysts said startup valuations and IPO listings are rising as investor confidence improves."]),
 "film_fading": dict(  # FADING topic
   fills=dict(film=["The Silent River", "Neon Monsoon", "Paper Kings", "Last Light"],
              fest=["the film festival", "the awards ceremony"]),
   titles=["'{film}' wins top prize at {fest}",
           "Box office: '{film}' crosses 100 crore in opening week",
           "Director of '{film}' speaks on festival success",
           "Stars walk red carpet at {fest} premiere of '{film}'"],
   desc=["The movie drew strong reviews from critics and audiences at {fest}, with the actor praised for the lead role.",
         "Cinema halls reported packed shows as '{film}' broke box office records.",
         "The director said the film's music and cast helped it win the award at {fest}."]),
}

# temporal profiles: hours-ago sampler (0 = now)
def when(topic: str) -> float:
    if topic == "climate_cyclone":
        return rng.uniform(0, 36)              # burst: last 36 hours only
    if topic == "film_fading":
        return rng.uniform(96, 168) if rng.random() < 0.85 else rng.uniform(24, 96)
    return rng.uniform(0, 168)                 # steady over 7 days

COUNTS = {"ai_tech": 50, "markets": 45, "elections": 40, "climate_cyclone": 40, "health": 35,
          "cricket_sports": 35, "geopolitics": 45, "startups": 35, "film_fading": 30}


def fill(tpl: str, f: dict) -> str:
    vals = {k: rng.choice(v) for k, v in f.items()}
    if "{team2}" in tpl:
        vals["team2"] = rng.choice([t for t in f["team"] if t != vals["team"]])
    if "{b}" in tpl:
        vals["b"] = rng.choice([x for x in f["a"] if x != vals.get("a")])
    return tpl.format(**vals)


def main() -> None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    for topic, n in COUNTS.items():
        spec = T[topic]
        for _ in range(n):
            title = fill(rng.choice(spec["titles"]), spec["fills"])
            title = title[0].upper() + title[1:]
            d1 = fill(rng.choice(spec["desc"]), spec["fills"])
            d2 = fill(rng.choice(spec["desc"]), spec["fills"])
            ts = now - timedelta(hours=when(topic))
            slug = "-".join(title.lower().replace("'", "").split()[:7])
            rows.append(dict(title=title, description=d1, content=f"{d1} {d2}",
                             source=rng.choice(SOURCES),
                             url=f"https://example.com/news/{slug}-{rng.randint(1000, 9999)}",
                             published_at=ts.isoformat(), language="en", _topic=topic))
    # duplicates (same headline, different URL) -> must be removed by dedupe
    for r in rng.sample(rows, 10):
        d = dict(r)
        d["url"] = d["url"] + "?utm_source=dup"
        rows.append(d)
    # robustness cases
    rows[3]["description"] = ""
    rows[17]["content"] = ""
    rows[41]["description"] = "<p>Markets <b>closed</b> lower &amp; volatile.</p> Read more at https://example.com/x"
    rows[42]["content"] = rows[42]["content"] + " … [+2345 chars]"
    # non-English -> filtered by language detection
    non_en = [
        ("चुनाव आयोग ने मतदान की तारीखों की घोषणा की", "राज्य में मतदान अगले महीने होगा और मतगणना उसके बाद की जाएगी।"),
        ("El banco central mantiene las tasas de interés", "Los inversores esperan que la inflación baje en los próximos meses."),
        ("Le gouvernement annonce un plan pour l'intelligence artificielle", "Le plan vise à former des ingénieurs et à soutenir les entreprises."),
        ("Ciclón provoca inundaciones en la costa", "Miles de personas fueron evacuadas por las fuertes lluvias."),
        ("Die Börse schließt mit Verlusten", "Die Anleger reagierten auf neue Inflationsdaten aus den Vereinigten Staaten."),
    ]
    for t, d in non_en:
        rows.append(dict(title=t, description=d, content=d, source="Sample Herald",
                         url=f"https://example.com/intl/{rng.randint(10000, 99999)}",
                         published_at=(now - timedelta(hours=rng.uniform(0, 100))).isoformat(),
                         language="", _topic="non_english"))
    rng.shuffle(rows)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
