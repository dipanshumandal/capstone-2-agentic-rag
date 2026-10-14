from src.tools import classify_intent, analyse_sentiment, rewrite_query, check_escalation


def test_classify_intent():
    out = classify_intent.invoke({"text": "I want to cancel the order I placed yesterday"})
    assert out["intent"] == "cancel_order"
    assert out["category"] == "order"
    assert 0 < out["confidence"] <= 1


def test_sentiment_negative_and_urgent():
    text = "This is the third time I am asking, my refund is still not here, this is unacceptable!!"
    out = analyse_sentiment.invoke({"text": text})
    assert out["sentiment"] == "negative"
    assert out["urgency"] == "high"


def test_sentiment_positive():
    out = analyse_sentiment.invoke({"text": "Thanks a lot, that was really helpful"})
    assert out["sentiment"] == "positive"
    assert out["urgency"] == "low"


def test_rewrite_query_removes_ids_and_adds_intent():
    out = rewrite_query.invoke({"query": "where is order SE12345678?", "intent": "track_order"})
    assert "SE12345678" not in out
    assert "track order" in out


def test_escalation_for_complaint():
    out = check_escalation.invoke({"intent": "complaint", "sentiment": "negative", "urgency": "high",
                                   "validation_passed": True, "confidence": 0.8})
    assert out["escalate"] is True
    assert out["ticket_id"].startswith("CS-")
    assert out["priority"] == "high"


def test_no_escalation_for_normal_question():
    out = check_escalation.invoke({"intent": "track_order", "sentiment": "neutral", "urgency": "low",
                                   "validation_passed": True, "confidence": 0.8})
    assert out["escalate"] is False
    assert out["ticket_id"] is None
