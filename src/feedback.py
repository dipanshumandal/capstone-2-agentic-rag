"""
Feedback loop storage. Thumbs up / down go to feedback.jsonl, corrections additionally go to
corrections.jsonl which the retriever reads so the corrected answer is used next time.
"""
import json
from datetime import datetime
from pathlib import Path

from src.config import FEEDBACK_DIR


def save_feedback(session_id, query, answer, rating, correction=None, intent=None):
    Path(FEEDBACK_DIR).mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "rating": rating,            # "up" or "down"
        "correction": correction,
        "intent": intent,
    }
    with open(Path(FEEDBACK_DIR) / "feedback.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    if correction:
        with open(Path(FEEDBACK_DIR) / "corrections.jsonl", "a") as f:
            f.write(json.dumps({"query": query, "correction": correction, "intent": intent}) + "\n")
    return row


def load_feedback():
    path = Path(FEEDBACK_DIR) / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def feedback_summary():
    rows = load_feedback()
    ups = sum(r["rating"] == "up" for r in rows)
    downs = sum(r["rating"] == "down" for r in rows)
    by_intent = {}
    for r in rows:
        if r["rating"] == "down":
            by_intent[r["intent"]] = by_intent.get(r["intent"], 0) + 1
    return {"total": len(rows), "up": ups, "down": downs, "corrections": sum(bool(r["correction"]) for r in rows),
            "thumbs_down_by_intent": by_intent}
