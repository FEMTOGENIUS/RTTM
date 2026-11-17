"""Structured logging to console and logs/rttm.log (INFO / WARNING / ERROR)."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

import config

_CONFIGURED = False


def get_logger(name: str = "rttm") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        for stream in (sys.stdout, sys.stderr):  # Windows consoles: never crash on emoji
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
        root = logging.getLogger("rttm")
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(config.LOG_DIR / "rttm.log", maxBytes=2_000_000,
                                     backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:  # read-only FS etc. -> console only
            pass
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
        root.propagate = False
        _CONFIGURED = True
    if name == "rttm" or name.startswith("rttm."):
        return logging.getLogger(name)
    return logging.getLogger(f"rttm.{name}")


def tail_log(n: int = 200) -> str:
    path = config.LOG_DIR / "rttm.log"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[-n:])
