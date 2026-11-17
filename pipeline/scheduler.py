"""Background ingestion service (cron-like).

Fetches news every N minutes into the shared article store, so the dashboard
always has fresh data even when nobody is looking at it. Optionally re-runs
the topic model after each fetch and pushes alerts for emerging / bursting
topics to a webhook (Slack/Discord/Teams-compatible JSON `{"text": ...}`).

Usage:
    python -m pipeline.scheduler                       # every 10 min (config)
    python -m pipeline.scheduler --interval 5 --query "india" --sources google_news rss
    python -m pipeline.scheduler --once                # single fetch, then exit
Environment:
    ALERT_WEBHOOK_URL=https://hooks.slack.com/...      # optional
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

import config  # noqa: E402
from data.fetchers import fetch_all  # noqa: E402
from data.store import ArticleStore  # noqa: E402
from models.engine import run_pipeline  # noqa: E402
from utils.logging_setup import get_logger  # noqa: E402

log = get_logger("scheduler")
_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def push_alerts(messages: list[str], url: str | None) -> None:
    if not messages or not url:
        return
    try:
        requests.post(url, json={"text": "\n".join(f"• {m}" for m in messages)}, timeout=10)
        log.info("pushed %d alerts to webhook", len(messages))
    except Exception as exc:
        log.warning("alert webhook failed: %s", exc)


def run_once(args, store: ArticleStore, seen_alerts: set[str]) -> dict:
    rep = fetch_all(args.sources, query=args.query, language=args.language, limit=args.limit,
                    hours=args.hours, country=args.country)
    for a in rep.articles:
        a["query"] = args.query
    new = store.add(rep.articles)
    store.prune()
    summary = {"fetched": len(rep.articles), "new": new, "errors": rep.errors, "fetch_s": round(rep.seconds, 2)}
    if args.model and new:
        from data.sources import edition_key

        ed = edition_key(args.language, args.country)
        rows = [r for r in store.load(hours=args.hours) if r.get("edition", ed) == ed]
        out = run_pipeline(rows, models=[args.model], n_topics=args.topics, language=args.language)
        msgs = [m for model in out.alerts for m in out.alerts[model]]
        fresh = [m for m in msgs if m not in seen_alerts]
        seen_alerts.update(fresh)
        for m in fresh:
            log.warning("ALERT %s", m)
        push_alerts(fresh, args.webhook)
        summary.update(topics=sum(len(r.topic_ids) for r in out.results.values()), alerts=fresh,
                       model_s=round(out.timings.get("total", 0), 2))
    log.info("scheduler cycle: %s", summary)
    print(time.strftime("%H:%M:%S"), summary, flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=float, default=config.FETCH_INTERVAL_MIN, help="minutes between fetches")
    p.add_argument("--sources", nargs="+", default=["google_news", "google_sections", "publishers"],
                   choices=["google_news", "google_sections", "publishers", "international", "rss", "newsapi",
                            "gnews", "sample"])
    p.add_argument("--query", default="")
    p.add_argument("--language", default="en")
    p.add_argument("--country", default="IN")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--model", default="BERTopic", choices=["BERTopic", "LDA", ""],
                   help="re-run this model after each fetch to detect emerging topics ('' = off)")
    p.add_argument("--topics", type=int, default=config.DEFAULT_N_TOPICS)
    p.add_argument("--webhook", default=os.getenv("ALERT_WEBHOOK_URL"))
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)
    store = ArticleStore()
    seen: set[str] = set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:  # use the `schedule` package when installed (as in the report), else a plain loop
        import schedule  # type: ignore
    except Exception:
        schedule = None
    if args.once:
        return run_once(args, store, seen)
    run_once(args, store, seen)
    if schedule is not None:
        schedule.every(args.interval).minutes.do(run_once, args, store, seen)
        while not _STOP:
            schedule.run_pending()
            time.sleep(1)
    else:
        nxt = time.time() + args.interval * 60
        while not _STOP:
            if time.time() >= nxt:
                try:
                    run_once(args, store, seen)
                except Exception as exc:  # keep the service alive
                    log.exception("cycle failed: %s", exc)
                nxt = time.time() + args.interval * 60
            time.sleep(1)
    log.info("scheduler stopped")


if __name__ == "__main__":
    main()
