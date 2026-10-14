from src.graph import build_graph, run_query


def test_run_query_shape():
    app = build_graph()
    out = run_query("How do I track my order?", session_id="t1", app=app)
    for key in ["answer", "sources", "retrieved_context", "agents_invoked", "confidence", "intent", "sentiment",
                "validation", "escalation"]:
        assert key in out
    assert out["answer"]
    assert out["agents_invoked"][0] == "planner"
    assert out["agents_invoked"][-1] == "escalation"
    assert out["intent"] == "track_order"
    assert out["validation"]["passed"] is True
    assert out["llm_model"] == "mock"


def test_memory_keeps_conversation_per_session():
    app = build_graph()
    run_query("I want to change my shipping address", session_id="s1", app=app)
    run_query("and how long will it take?", session_id="s1", app=app)
    run_query("hello", session_id="s2", app=app)

    s1 = app.get_state({"configurable": {"thread_id": "s1"}}).values
    s2 = app.get_state({"configurable": {"thread_id": "s2"}}).values
    assert len(s1["messages"]) == 4     # 2 human + 2 ai
    assert len(s2["messages"]) == 2


def test_human_agent_request_is_escalated():
    out = run_query("I want to talk to a real person right now, this bot is useless", session_id="t2", app=build_graph())
    assert out["escalation"]["escalate"] is True
    assert out["escalation"]["ticket_id"] in out["answer"]
    assert out["sentiment"]["sentiment"] == "negative"


def test_retry_limit_is_respected():
    # gibberish should fail validation, the graph must still finish within the retry limit
    out = run_query("qwerty zxcvb asdfg lorem ipsum", session_id="t3", app=build_graph())
    assert out["retries"] <= 2
    assert out["agents_invoked"].count("retriever") == out["retries"] + 1
