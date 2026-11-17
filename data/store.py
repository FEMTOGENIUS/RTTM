"""File-based article store (JSON Lines).

* Persists articles between dashboard sessions and the background scheduler.
* De-duplicates on headline hash (and URL).
* Periodically flushes old data to prevent unbounded growth.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from utils.logging_setup import get_logger

log = get_logger("store")
_LOCK = threading.Lock()


class ArticleStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.STORE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- read
    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("skipping corrupt line in store")
        return rows

    def ids(self) -> set[str]:
        return {r["id"] for r in self.all()}

    def load(self, hours: float | None = None, start: datetime | None = None,
             end: datetime | None = None) -> list[dict]:
        rows = self.all()
        if hours:
            start = datetime.now(timezone.utc) - timedelta(hours=hours)
        if start or end:
            out = []
            for r in rows:
                try:
                    ts = datetime.fromisoformat(r["published_at"])
                except Exception:
                    continue
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                out.append(r)
            rows = out
        return rows

    # ---------------------------------------------------------------- write
    def add(self, articles: list[dict]) -> int:
        """Append unseen articles. Returns number of *new* articles."""
        with _LOCK:
            existing = self.all()
            seen_ids = {r["id"] for r in existing}
            seen_urls = {r.get("url") for r in existing if r.get("url")}
            new = []
            for a in articles:
                if a["id"] in seen_ids or (a.get("url") and a["url"] in seen_urls):
                    continue
                seen_ids.add(a["id"])
                if a.get("url"):
                    seen_urls.add(a["url"])
                new.append(a)
            if new:
                with open(self.path, "a", encoding="utf-8") as fh:
                    for a in new:
                        fh.write(json.dumps(a, ensure_ascii=False) + "\n")
            log.info("store: +%d new (%d duplicates skipped)", len(new), len(articles) - len(new))
            return len(new)

    def prune(self, max_age_days: int | None = None, max_items: int | None = None) -> int:
        max_age_days = max_age_days or config.STORE_MAX_AGE_DAYS
        max_items = max_items or config.STORE_MAX_ITEMS
        with _LOCK:
            rows = self.all()
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            keep = []
            for r in rows:
                try:
                    if datetime.fromisoformat(r["published_at"]) < cutoff:
                        continue
                except Exception:
                    pass
                keep.append(r)
            keep = sorted(keep, key=lambda r: r.get("published_at", ""))[-max_items:]
            removed = len(rows) - len(keep)
            if removed:
                self._rewrite(keep)
                log.info("store: pruned %d old articles", removed)
            return removed

    def clear(self) -> None:
        with _LOCK:
            if self.path.exists():
                self.path.unlink()

    def _rewrite(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- export
    def to_frame(self, rows: list[dict] | None = None) -> pd.DataFrame:
        return pd.DataFrame(rows if rows is not None else self.all())
