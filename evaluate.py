"""Reproducible evaluation: LDA vs BERTopic.

Metrics: coherence C_V / NPMI / UMass, topic diversity, raw outlier rate,
latency per stage, (when labels exist) ARI / NMI against ground truth, and a
stress test (burst of 500+ articles incl. blank / malformed records).

Examples
  python evaluate.py --source live --runs 3                 # live Google News + RSS
  python evaluate.py --source 20newsgroups --max-docs 2000  # offline benchmark (slides: validation set)
  python evaluate.py --source store --hours 72              # whatever the dashboard collected
  python evaluate.py --source file:my_articles.csv
  python evaluate.py --source sample --stress               # synthetic demo data + stress test

Writes results/eval_<timestamp>.json and .md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data.fetchers import fetch_all, load_records, load_sample  # noqa: E402
from data.store import ArticleStore  # noqa: E402
from models.engine import run_pipeline  # noqa: E402
from utils.preprocess import backend_info  # noqa: E402
from models.bertopic_model import backend_summary  # noqa: E402

METRICS = ["coherence_cv", "coherence_npmi", "coherence_umass", "diversity", "n_topics",
           "outlier_rate_raw", "runtime_s"]


def load_source(args) -> tuple[list[dict], str]:
    src = args.source
    if src == "sample":
        return load_sample(), "bundled SYNTHETIC demo dataset (not a research benchmark)"
    if src == "live":
        rep = fetch_all(["google_news", "rss"], query=args.query, limit=60, hours=args.hours)
        if rep.errors:
            print("fetch warnings:", *rep.errors, sep="\n  ")
        return rep.articles, f"live Google News RSS + RSS feeds, fetched {datetime.now():%Y-%m-%d %H:%M}"
    if src == "store":
        return ArticleStore().load(hours=args.hours), f"article store, last {args.hours}h"
    if src == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        ds = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                random_state=config.RANDOM_STATE)
        idx = list(range(len(ds.data)))
        random.Random(config.RANDOM_STATE).shuffle(idx)
        now = datetime.now(timezone.utc)
        rows = []
        for k, i in enumerate(idx[: args.max_docs]):
            text = " ".join(ds.data[i].split())
            if len(text) < 40:
                continue
            rows.append({"title": text[:90], "description": "", "content": text[:3000],
                         "source": "20newsgroups", "url": f"20ng://{i}",
                         "published_at": (now - timedelta(minutes=k)).isoformat()})
        recs = load_records(rows, origin="20newsgroups")
        for r, row in zip(recs, rows):
            r["gold_topic"] = ds.target_names[ds.target[int(row["url"].split("//")[1])]]
        return recs, f"20 Newsgroups (subset of {len(recs)} posts)"
    if src.startswith("file:"):
        path = Path(src[5:])
        recs = pd.read_csv(path).to_dict("records") if path.suffix == ".csv" else json.loads(path.read_text())
        return load_records(recs), f"file {path.name}"
    raise SystemExit(f"unknown source {src}")


def stress_test(base: list[dict], models: list[str]) -> dict:
    """Burst of 500+ articles incl. blank titles/descriptions & malformed records."""
    rng = random.Random(1)
    burst = []
    for i in range(520):
        a = dict(rng.choice(base))
        a["id"] = f"stress{i}"
        a["title"] = f"{a['title']} #{i}" if rng.random() > 0.03 else ""
        if rng.random() < 0.05:
            a["description"] = ""
        if rng.random() < 0.03:
            a["content"] = None
        if rng.random() < 0.02:
            a["published_at"] = "not-a-date"
        burst.append(a)
    t0 = time.perf_counter()
    out = run_pipeline(burst, models=models, bertopic_nr_topics="auto")
    return {"articles_in": len(burst), "articles_used": len(out.df), "seconds": round(time.perf_counter() - t0, 2),
            "preprocessing": out.prep_stats, "warnings": out.warnings,
            "models_ok": list(out.results), "crashed": False}


def fmt(mean, std=None, pct=False):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    if pct:
        return f"{100 * mean:.1f}%" + (f" ± {100 * std:.1f}" if std else "")
    return f"{mean:.3f}" + (f" ± {std:.3f}" if std else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="sample")
    p.add_argument("--query", default="")
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--max-docs", type=int, default=2000)
    p.add_argument("--models", nargs="+", default=["LDA", "BERTopic"])
    p.add_argument("--topics", type=int, default=config.DEFAULT_N_TOPICS)
    p.add_argument("--bertopic-auto", action="store_true", help="let HDBSCAN choose #topics")
    p.add_argument("--runs", type=int, default=3, help="repetitions with different random seeds")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    args = p.parse_args(argv)

    rows, desc = load_source(args)
    print(f"Loaded {len(rows)} articles from {desc}")
    runs = []
    for k in range(args.runs):
        out = run_pipeline(rows, models=args.models, n_topics=args.topics,
                           bertopic_nr_topics="auto" if args.bertopic_auto else args.topics,
                           random_state=config.RANDOM_STATE + k)
        rec = {"seed": config.RANDOM_STATE + k, "timings": out.timings, "metrics": {}, "warnings": out.warnings}
        gold = out.df["gold_topic"] if "gold_topic" in out.df and out.df["gold_topic"].astype(bool).all() else None
        for m, r in out.results.items():
            mm = {key: out.metrics[m][key] for key in METRICS}
            if gold is not None:
                mm["ari_vs_gold"] = adjusted_rand_score(gold, r.assignments)
                mm["nmi_vs_gold"] = normalized_mutual_info_score(gold, r.assignments)
            mm["backend"] = r.backend
            mm["topics"] = {int(t): r.top_words(t, 10) for t in r.topic_ids}
            rec["metrics"][m] = mm
        if out.comparison:
            rec["lda_vs_bertopic"] = {"ari": out.comparison["ari"], "nmi": out.comparison["nmi"]}
        runs.append(rec)
        print(f"run {k + 1}/{args.runs}: " + ", ".join(
            f"{m}: C_V={v['coherence_cv']:.3f} TD={v['diversity']:.2f} K={v['n_topics']}"
            for m, v in rec["metrics"].items()))

    summary = {}
    for m in args.models:
        vals = [r["metrics"][m] for r in runs if m in r["metrics"]]
        if not vals:
            continue
        keys = METRICS + [k for k in ("ari_vs_gold", "nmi_vs_gold") if k in vals[0]]
        summary[m] = {k: (float(np.nanmean([v[k] for v in vals])), float(np.nanstd([v[k] for v in vals])))
                      for k in keys}
        summary[m]["backend"] = vals[0]["backend"]
    stage = pd.DataFrame([r["timings"] for r in runs]).mean().round(3).to_dict()
    stress = stress_test(rows, args.models) if args.stress else None

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"dataset": desc, "n_input": len(rows), "n_used": int(len(out.df)),
              "preprocessing": out.prep_stats, "settings": vars(args), "backends":
              {"preprocessing": backend_info(), "bertopic": backend_summary()},
              "summary": summary, "mean_stage_latency_s": stage, "runs": runs, "stress_test": stress}
    jpath = Path(args.out) / f"eval_{stamp}.json"
    jpath.write_text(json.dumps(report, indent=1, default=str))

    md = [f"# Evaluation report - {datetime.now():%Y-%m-%d %H:%M}", "",
          f"**Dataset:** {desc}  ", f"**Articles:** {len(rows)} loaded, {len(out.df)} after preprocessing "
          f"({out.prep_stats})  ", f"**Runs:** {args.runs} (different random seeds)  ",
          f"**Settings:** topics={args.topics}, BERTopic auto={args.bertopic_auto}", "",
          "| Metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    labels = {"coherence_cv": "Coherence C_V ↑", "coherence_npmi": "Coherence NPMI ↑",
              "coherence_umass": "Coherence UMass ↑", "diversity": "Topic diversity (top-25) ↑",
              "n_topics": "# topics", "outlier_rate_raw": "Raw HDBSCAN outliers",
              "runtime_s": "Model runtime (s)", "ari_vs_gold": "ARI vs ground truth ↑",
              "nmi_vs_gold": "NMI vs ground truth ↑"}
    for k, lab in labels.items():
        if all(k in summary[m] for m in summary):
            pct = k in ("diversity", "outlier_rate_raw")
            md.append(f"| {lab} | " + " | ".join(fmt(*summary[m][k], pct=pct) for m in summary) + " |")
    md.append("| Backend | " + " | ".join(summary[m]["backend"] for m in summary) + " |")
    md += ["", "**Mean latency per stage (s):** " + ", ".join(f"{k}={v}" for k, v in stage.items()), ""]
    if "lda_vs_bertopic" in runs[0]:
        md.append(f"**LDA vs BERTopic agreement (run 1):** ARI={runs[0]['lda_vs_bertopic']['ari']:.3f}, "
                  f"NMI={runs[0]['lda_vs_bertopic']['nmi']:.3f}")
    for m in summary:
        md += ["", f"## {m} topics (run 1)", ""]
        for t, w in runs[0]["metrics"][m]["topics"].items():
            md.append(f"- **{t}**: {', '.join(w)}")
    if stress:
        md += ["", "## Stress test", "", f"```\n{json.dumps(stress, indent=1)}\n```"]
    mpath = Path(args.out) / f"eval_{stamp}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:18 + len(labels)]))
    print(f"\nSaved {jpath} and {mpath}")
    return report


if __name__ == "__main__":
    main()
