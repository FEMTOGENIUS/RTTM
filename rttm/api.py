"""Optional REST API (FastAPI) exposing the same pipeline as the dashboard.

Run:  uvicorn api:app --reload --port 8000      (docs at http://localhost:8000/docs)

Endpoints
  GET  /health                      -> status + back-ends
  POST /fetch?query=&sources=       -> fetch live news into the store
  GET  /articles?hours=24           -> stored articles
  GET  /topics?model=BERTopic&hours=48&n_topics=8 -> topics, metrics, trends, alerts
"""
from __future__ import annotations

from fastapi import FastAPI, Query  # type: ignore

from data.fetchers import fetch_all, load_sample
from data.store import ArticleStore
from models.bertopic_model import backend_summary
from models.engine import run_pipeline
from utils.preprocess import backend_info

app = FastAPI(title="Real-Time News Topic Modeling API", version="1.0")
store = ArticleStore()


@app.get("/health")
def health():
    return {"status": "ok", "stored_articles": len(store.all()),
            "backends": {"preprocessing": backend_info(), "bertopic": backend_summary()}}


@app.post("/fetch")
def fetch(query: str = "", sources: list[str] = Query(default=["google_news", "rss"]),
          limit: int = 40, hours: int = 48):
    rep = fetch_all(sources, query=query, limit=limit, hours=hours)
    for a in rep.articles:
        a["query"] = query
    return {"fetched": len(rep.articles), "new": store.add(rep.articles), "errors": rep.errors,
            "per_source": rep.per_source, "seconds": rep.seconds}


@app.get("/articles")
def articles(hours: int | None = 24, limit: int = 500):
    return store.load(hours=hours)[-limit:]


@app.get("/topics")
def topics(model: str = "BERTopic", hours: int | None = 48, n_topics: int = 8, sample: bool = False):
    rows = load_sample() if sample else store.load(hours=hours)
    out = run_pipeline(rows, models=[model], n_topics=n_topics,
                       bertopic_nr_topics="auto" if model == "BERTopic" else n_topics)
    resp = {"n_articles": len(out.df), "warnings": out.warnings, "timings_s": out.timings, "models": {}}
    for m, r in out.results.items():
        sizes = r.sizes()
        met = {k: v for k, v in out.metrics[m].items() if k != "coherence_cv_per_topic"}
        resp["models"][m] = {
            "backend": r.backend, "metrics": met, "alerts": out.alerts[m],
            "trends": out.trends[m].to_dict("records"),
            "topics": [{"id": t, **out.labels[m][t], "size": sizes.get(t, 0),
                        "words": r.top_words(t, 10)} for t in r.topic_ids],
        }
    return resp
