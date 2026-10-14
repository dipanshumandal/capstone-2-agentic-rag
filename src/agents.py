"""
The agents of the support workflow, written as LangGraph nodes.

planner (query understanding) -> sentiment -> retriever -> validator
   -> if the context is weak: rewriter -> retriever again (max 2 retries)
   -> responder (gemini) -> escalation -> END
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from src.config import TOP_K, MAX_RETRIES, COMPANY
from src.tools import rag_search, classify_intent, analyse_sentiment, rewrite_query, check_escalation
from src.llm import call_llm, parse_json


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]   # conversation memory, kept per session by the checkpointer
    query: str
    intent: str
    category: str
    intent_confidence: float
    plan: List[str]
    search_query: str
    k: int
    retries: int
    retrieved: List[Dict[str, Any]]
    validation: Dict[str, Any]
    sentiment: Dict[str, Any]
    answer: str
    sources: List[str]
    confidence: float
    llm_model: str
    escalation: Dict[str, Any]


# ---------------- planner / query understanding ----------------
def planner_agent(state: SupportState):
    query = state["query"]

    # short follow ups like "and what if it is late?" mean nothing on their own, so the previous
    # customer message is added for intent detection and retrieval (this is where the memory helps)
    previous = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    text_for_search = query
    if previous and (len(query.split()) <= 6 or query.lower().startswith(("and ", "what about", "also ", "then "))):
        text_for_search = f"{previous[-1]} {query}"

    pred = classify_intent.invoke({"text": text_for_search})

    # the plan is the list of steps the executor (the graph) will follow for this query
    if pred["intent"] in ("contact_human_agent", "complaint"):
        plan = ["classify_intent", "analyse_sentiment", "rag_search", "validate_context",
                "generate_answer", "escalate_to_human"]
    else:
        plan = ["classify_intent", "analyse_sentiment", "rag_search", "validate_context",
                "generate_answer", "check_escalation"]

    # query expansion: when the classifier is fairly sure, add the intent words to the search query,
    # bm25 picks them up and it pulls in the right faq group
    search_query = text_for_search
    if pred["confidence"] >= 0.5:
        search_query = f"{text_for_search} ({pred['intent'].replace('_', ' ')})"

    return {
        "messages": [HumanMessage(content=query)],
        "intent": pred["intent"],
        "category": pred["category"],
        "intent_confidence": pred["confidence"],
        "plan": plan,
        "search_query": search_query,
        "k": TOP_K,
        "retries": 0,
        "retrieved": [],
        "validation": {},
        "escalation": {},
    }


# ---------------- retriever ----------------
def retriever_agent(state: SupportState):
    chunks = rag_search.invoke({"query": state["search_query"], "k": state["k"]})
    return {"retrieved": chunks}


# ---------------- sentiment ----------------
def sentiment_agent(state: SupportState):
    return {"sentiment": analyse_sentiment.invoke({"text": state["query"]})}


# ---------------- validator ----------------
def validator_agent(state: SupportState):
    retrieved = state["retrieved"]
    if not retrieved:
        return {"validation": {"passed": False, "relevance": 0.0, "intent_agreement": 0.0,
                               "score": 0.0, "reason": "nothing retrieved"}}

    # 1. relevance: how close is the best chunk to the question (cosine)
    best = max(r.get("sem", r["score"]) for r in retrieved)

    # 2. grounding: do the faq chunks talk about the intent the planner found?
    #    policy chunks and human corrections do not have an intent so they are left out
    faq = [r for r in retrieved if r.get("intent") not in ("policy", "unknown", None)]
    agree = sum(r["intent"] == state["intent"] for r in faq) / len(faq) if faq else 1.0

    # if the classifier itself was not sure we do not punish the retriever for disagreeing with it
    passed = best >= 0.45 and (agree >= 0.5 or state["intent_confidence"] < 0.5)
    if passed:
        reason = "context is relevant and matches the intent"
    elif best < 0.45:
        reason = f"best chunk similarity {best:.2f} is below 0.45"
    else:
        reason = f"only {agree:.0%} of the faq chunks match intent '{state['intent']}'"

    return {"validation": {"passed": passed, "relevance": round(best, 3), "intent_agreement": round(agree, 2),
                           "score": round(0.6 * best + 0.4 * agree, 3), "reason": reason}}


def route_after_validation(state: SupportState):
    if state["validation"]["passed"] or state["retries"] >= MAX_RETRIES:
        return "responder"
    return "rewriter"


# ---------------- rewriter (feedback inside the graph) ----------------
def rewriter_agent(state: SupportState):
    new_query = rewrite_query.invoke({"query": state["query"], "intent": state["intent"]})
    if state["retries"] >= 1:
        # second retry: drop the intent hint in case the classifier was the problem
        new_query = state["query"]
    return {"search_query": new_query, "k": state["k"] + 2, "retries": state["retries"] + 1}


# ---------------- responder (the only agent that calls gemini) ----------------
RESPONSE_PROMPT = """You are the customer support assistant of {company}, an Indian e-commerce store.
Answer the customer's question using ONLY the context below. The context has FAQ answers and policy text.
If the context does not contain the answer, say that you do not have that information and suggest contacting support.
Do not invent order details, fees or dates. Keep the answer short (3 to 6 sentences), plain text, no markdown.
{tone}

CONTEXT:
{context}

CHAT HISTORY (older messages first):
{history}

CUSTOMER QUESTION: {question}

Reply with JSON only, in this exact shape:
{{"answer": "<your answer>", "sources_used": [<numbers of the context items you used>], "confidence": <0 to 1>}}
"""


def _tone(sentiment):
    if sentiment.get("sentiment") == "negative":
        return "The customer is upset. Start with a short apology, stay calm and give concrete next steps."
    if sentiment.get("urgency") == "high":
        return "The customer is in a hurry, give the fastest option first."
    return "Be friendly and to the point."


def responder_agent(state: SupportState):
    retrieved = state["retrieved"]
    context = "\n".join(f"[{i + 1}] ({r['source']}) {r['content']}" for i, r in enumerate(retrieved))
    history_msgs = state["messages"][:-1][-6:]       # last 3 exchanges, the current question is the last message
    history = "\n".join(f"{'Customer' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                        for m in history_msgs) or "(no previous messages)"

    prompt = RESPONSE_PROMPT.format(company=COMPANY, tone=_tone(state.get("sentiment", {})),
                                    context=context, history=history, question=state["query"])
    raw, model_used = call_llm(prompt)
    out = parse_json(raw)

    used = [i for i in out.get("sources_used", []) if isinstance(i, int) and 1 <= i <= len(retrieved)]
    if not used:
        used = list(range(1, min(2, len(retrieved)) + 1))
    sources = []
    for i in used:
        s = retrieved[i - 1].get("chunk_id") or retrieved[i - 1]["source"]
        if s not in sources:
            sources.append(s)

    llm_conf = out.get("confidence", 0.5)
    try:
        llm_conf = float(llm_conf)
    except (TypeError, ValueError):
        llm_conf = 0.5
    confidence = round(0.5 * llm_conf + 0.5 * state["validation"].get("score", 0.0), 2)

    return {"answer": str(out.get("answer", "")).strip(), "sources": sources,
            "confidence": confidence, "llm_model": model_used}


# ---------------- escalation ----------------
def escalation_agent(state: SupportState):
    sentiment = state.get("sentiment", {})
    decision = check_escalation.invoke({
        "intent": state["intent"],
        "sentiment": sentiment.get("sentiment", "neutral"),
        "urgency": sentiment.get("urgency", "low"),
        "validation_passed": state["validation"].get("passed", False),
        "confidence": state["confidence"],
    })
    answer = state["answer"]
    if decision["escalate"]:
        answer += (f"\n\nI have also raised ticket {decision['ticket_id']} with our support team "
                   f"({decision['priority']} priority). A support executive will get back to you within 24 hours.")
    return {"escalation": decision, "answer": answer, "messages": [AIMessage(content=answer)]}
