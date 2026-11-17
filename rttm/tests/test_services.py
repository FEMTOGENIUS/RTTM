"""Background scheduler, alert webhook, REST API, evaluation CLI."""

import pytest

from pipeline import scheduler


def test_scheduler_once_fetches_models_and_alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(scheduler.requests, "post", lambda url, json=None, timeout=None: sent.append(json))
    out = scheduler.main(["--once", "--sources", "sample", "--model", "BERTopic",
                          "--webhook", "https://hooks.example.com/x", "--hours", "200"])
    assert out["new"] > 0 and out["topics"] > 0
    assert any("Emerging" in a for a in out["alerts"])
    assert sent and "Emerging" in sent[0]["text"]
    again = scheduler.main(["--once", "--sources", "sample", "--model", "BERTopic"])
    assert again["new"] == 0                     # de-duplicated against the store


def test_evaluate_cli(tmp_path):
    import evaluate

    rep = evaluate.main(["--source", "sample", "--runs", "2", "--topics", "9", "--stress", "--out", str(tmp_path)])
    s = rep["summary"]
    assert {"LDA", "BERTopic"} <= set(s)
    assert s["BERTopic"]["ari_vs_gold"][0] > 0.8
    assert rep["stress_test"]["crashed"] is False and rep["stress_test"]["articles_in"] >= 500
    assert list(tmp_path.glob("eval_*.md")) and list(tmp_path.glob("eval_*.json"))


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api

    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/topics", params={"sample": True, "model": "LDA", "n_topics": 6}).json()
    assert len(r["models"]["LDA"]["topics"]) == 6
