"""
Every agent that another agent might need is also exposed as a LangChain tool here
(Agents as Tools). The graph nodes in agents.py call these.
"""
import re
import uuid
import joblib
from pathlib import Path
from langchain_core.tools import tool
from nltk.sentiment import SentimentIntensityAnalyzer

from src.config import INDEX_DIR, TOP_K
from src.retrieval import get_retriever

_intent_clf = None
_sia = None


def _intent_model():
    global _intent_clf
    if _intent_clf is None:
        _intent_clf = joblib.load(Path(INDEX_DIR) / "intent_clf.pkl")
    return _intent_clf


def _sentiment_model():
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
        # a few words customers use that are missing from the VADER lexicon
        _sia.lexicon.update({"unacceptable": -2.5, "scam": -3.0, "useless": -2.0, "pathetic": -2.5,
                             "ridiculous": -2.0, "nobody": -1.0, "waste": -1.5, "fraud": -3.0})
    return _sia


@tool
def rag_search(query: str, k: int = TOP_K) -> list:
    """Search the ShopEase knowledge base (FAQ answers and policy documents) and return the top k chunks."""
    return get_retriever().hybrid_search(query, k=k)


@tool
def classify_intent(text: str) -> dict:
    """Query Understanding agent. Predicts the customer's intent (27 bitext intents) and its category with a confidence."""
    bundle = _intent_model()
    clf = bundle["model"]
    vec = get_retriever().embeddings.embed_query(text)
    proba = clf.predict_proba([vec])[0]
    best = int(proba.argmax())
    intent = clf.classes_[best]
    return {
        "intent": intent,
        "category": bundle["intent_to_category"].get(intent, "unknown"),
        "confidence": round(float(proba[best]), 3),
    }


URGENT_WORDS = ["urgent", "immediately", "asap", "right now", "right away", "still waiting", "third time",
                "second time", "again", "unacceptable", "legal", "consumer court", "lawyer", "furious",
                "worst", "ridiculous", "scam", "fraud", "never", "complaint", "emergency"]

# VADER thinks words like "cancel", "penalty" or "refund" are negative. On a support desk they are just
# the topic of the question, so they are removed before scoring
DOMAIN_WORDS = re.compile(r"\b(penalty|penalties|cancel\w*|fees?|charges?|charged|refunds?|refunded|returns?|"
                          r"delete\w*|deletion|problems?)\b", re.I)


@tool
def analyse_sentiment(text: str) -> dict:
    """Sentiment agent. Returns the sentiment label (negative / neutral / positive), VADER score and urgency."""
    # scored sentence by sentence, otherwise a "not" in one sentence flips the words of the next one
    sentences = [x for x in re.split(r"[.!?]+\s*", DOMAIN_WORDS.sub("", text)) if x.strip()] or [text]
    scores = [_sentiment_model().polarity_scores(x)["compound"] for x in sentences]
    compound = sum(scores) / len(scores)
    if compound <= -0.3:
        label = "negative"
    elif compound >= 0.3:
        label = "positive"
    else:
        label = "neutral"

    low = text.lower()
    hits = sum(w in low for w in URGENT_WORDS)
    hits += low.count("!") // 2
    if hits >= 2 or (compound <= -0.5 and hits >= 1):
        urgency = "high"
    elif compound <= -0.3 or hits == 1:
        urgency = "medium"
    else:
        urgency = "low"
    return {"sentiment": label, "score": round(compound, 3), "urgency": urgency}


@tool
def rewrite_query(query: str, intent: str) -> str:
    """Rewrites a query whose context failed validation: drops order / invoice numbers, adds the intent keywords."""
    cleaned = re.sub(r"\b[A-Za-z]{0,3}\d{5,}\b", "", query)     # SE12345678, 987654 etc
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
    keywords = intent.replace("_", " ")
    if keywords.lower() in cleaned.lower():
        return cleaned
    return f"{cleaned} {keywords}".strip()


@tool
def check_escalation(intent: str, sentiment: str, urgency: str, validation_passed: bool, confidence: float) -> dict:
    """Escalation agent. Decides whether the conversation goes to a human support executive, creates a ticket if so."""
    reasons = []
    if intent in ("contact_human_agent", "complaint"):
        reasons.append(f"customer intent is '{intent}'")
    if sentiment == "negative" and urgency == "high":
        reasons.append("negative sentiment with high urgency")
    if not validation_passed:
        reasons.append("retrieved context did not pass validation after retries")
    if confidence < 0.4:
        reasons.append(f"low answer confidence ({confidence})")

    escalate = len(reasons) > 0
    priority = "high" if (urgency == "high" or intent == "complaint") else "normal"
    return {
        "escalate": escalate,
        "reasons": reasons,
        "priority": priority if escalate else None,
        "ticket_id": f"CS-{uuid.uuid4().hex[:6].upper()}" if escalate else None,
    }


ALL_TOOLS = [rag_search, classify_intent, analyse_sentiment, rewrite_query, check_escalation]
