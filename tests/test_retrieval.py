from src.retrieval import get_retriever, tokenize


def test_tokenize():
    assert tokenize("Track my ORDER SE12345678, please!") == ["track", "my", "order", "se12345678", "please"]


def test_index_loads():
    r = get_retriever()
    assert len(r.docs) > 500
    assert r.docs[0].metadata["chunk_id"]


def test_semantic_search_finds_right_intent():
    results = get_retriever().semantic_search("how can I track my order", k=3)
    assert len(results) == 3
    assert results[0]["intent"] == "track_order"
    assert 0 <= results[0]["score"] <= 1.01


def test_keyword_search_returns_results():
    results = get_retriever().keyword_search("refund policy", k=3)
    assert len(results) == 3
    assert all(r["method"] == "bm25" for r in results)


def test_hybrid_search_sorted_and_sized():
    results = get_retriever().hybrid_search("I forgot my password", k=4)
    assert len(results) == 4
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["intent"] == "recover_password"


def test_policy_documents_are_retrievable():
    results = get_retriever().hybrid_search("how much does same day delivery cost", k=4)
    assert any(r["source"] == "shipping_and_delivery.md" for r in results)
