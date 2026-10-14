import json
from fastapi.testclient import TestClient

import src.feedback as fb
from api.main import app

client = TestClient(app)


def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert "ShopEase" in r.json()["message"]


def test_health():
    with TestClient(app) as c:      # "with" so the startup (lifespan) runs
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["index_loaded"] is True
    assert r.json()["llm"] == "mock"


def test_query():
    r = client.post("/query", json={"session_id": "api-test", "query": "What payment methods do you accept?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"]
    assert data["intent"] == "check_payment_methods"
    assert "planner" in data["agents_invoked"]
    assert len(data["retrieved_context"]) >= 1
    assert 0 <= data["confidence"] <= 1


def test_query_validation_error():
    r = client.post("/query", json={"query": "hi"})       # shorter than min_length
    assert r.status_code == 422


def test_feedback_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "FEEDBACK_DIR", tmp_path)
    r = client.post("/feedback", json={"session_id": "api-test", "query": "q", "answer": "a", "rating": "up"})
    assert r.status_code == 200
    assert r.json()["saved"] is True
    rows = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert rows[0]["rating"] == "up"
    assert not (tmp_path / "corrections.jsonl").exists()


def test_feedback_invalid_rating():
    r = client.post("/feedback", json={"query": "q", "rating": "meh"})
    assert r.status_code == 422
