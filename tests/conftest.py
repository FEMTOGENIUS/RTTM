import os
import sys

os.environ.setdefault("RTTM_NO_DOWNLOAD", "1")
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Every test gets its own store / feedback files."""
    import config

    monkeypatch.setattr(config, "STORE_PATH", tmp_path / "store" / "articles.jsonl")
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    yield


class FakeResp:
    def __init__(self, status=200, json_data=None, content=b"", text=""):
        self.status_code = status
        self._json = json_data
        self.content = content
        self.text = text or (content.decode("utf-8", "replace") if content else "")

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes URLs to canned responses; records calls."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        for key, resp in self.routes.items():
            if key in url:
                return resp(params, headers) if callable(resp) else resp
        return FakeResp(404, text="not found")


@pytest.fixture
def fixture_bytes():
    return lambda name: (FIX / name).read_bytes()
