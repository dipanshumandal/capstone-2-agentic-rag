"""
FastAPI backend for the ShopEase support agent.
run:  uvicorn api.main:app --reload --port 8001
(8000 is used by the capstone-1 API on my machine)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.config import GEMINI_MODEL, MOCK_LLM, INDEX_DIR
from src.retrieval import get_retriever
from src.graph import run_query
from src.feedback import save_feedback, feedback_summary
from api.schemas import QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse


@asynccontextmanager
async def lifespan(app):
    # load the embedding model + faiss once at startup instead of on the first request
    get_retriever()
    yield


app = FastAPI(
    title="ShopEase Support - Agentic RAG API",
    description="Multi-agent RAG (LangGraph + Gemini) over the Bitext customer support FAQs and ShopEase policies",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "ShopEase Support Agentic RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": (INDEX_DIR / "index.faiss").exists(),
        "chunks": len(get_retriever().docs),
        "llm": "mock" if MOCK_LLM else GEMINI_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = run_query(req.query, session_id=req.session_id)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_context=result["retrieved_context"],
        agents_invoked=result["agents_invoked"],
        confidence=result["confidence"],
        intent=result["intent"],
        category=result["category"],
        sentiment=result["sentiment"],
        escalated=result["escalation"].get("escalate", False),
        ticket_id=result["escalation"].get("ticket_id"),
        validation=result["validation"],
        retries=result["retries"],
        llm_model=result["llm_model"],
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    save_feedback(req.session_id, req.query, req.answer, req.rating, req.correction, req.intent)
    if req.correction:
        # make the correction available to the retriever right away
        get_retriever().reload_corrections()
        return FeedbackResponse(saved=True, message="thanks, the correction will be used for similar questions")
    return FeedbackResponse(saved=True, message="thanks for the feedback")


@app.get("/feedback/summary")
def feedback_stats():
    return feedback_summary()
