"""
Builds the LangGraph workflow and gives one function, run_query(), used by the API and the notebook.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents import (SupportState, planner_agent, retriever_agent, sentiment_agent, validator_agent,
                        rewriter_agent, responder_agent, escalation_agent, route_after_validation)


def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)

    graph.add_node("planner", planner_agent)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("rewriter", rewriter_agent)
    graph.add_node("responder", responder_agent)
    graph.add_node("escalation", escalation_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "sentiment")
    graph.add_edge("sentiment", "retriever")
    # (first tried retriever + sentiment in parallel joined at the validator, but a join waits for
    #  both branches and the retry loop only re-runs the retriever, so the validator never fired again)
    graph.add_edge("retriever", "validator")
    graph.add_conditional_edges("validator", route_after_validation, {"responder": "responder", "rewriter": "rewriter"})
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("responder", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(query, session_id="default", app=None):
    """Runs one customer message through the graph. The session_id is the memory thread."""
    app = app or get_app()
    config = {"configurable": {"thread_id": session_id}}

    agents_invoked = []
    for update in app.stream({"query": query}, config, stream_mode="updates"):
        agents_invoked.extend(update.keys())

    state = app.get_state(config).values
    return {
        "answer": state["answer"],
        "sources": state["sources"],
        "retrieved_context": [r["content"] for r in state["retrieved"]],
        "agents_invoked": agents_invoked,
        "confidence": state["confidence"],
        "intent": state["intent"],
        "category": state["category"],
        "intent_confidence": state["intent_confidence"],
        "sentiment": state["sentiment"],
        "validation": state["validation"],
        "retries": state["retries"],
        "escalation": state["escalation"],
        "plan": state["plan"],
        "llm_model": state["llm_model"],
    }
