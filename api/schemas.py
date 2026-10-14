from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: str = Field("default", description="same id = same conversation (memory)")
    query: str = Field(..., min_length=3, max_length=1000, examples=["How do I track my order?"])


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_context: List[str]
    agents_invoked: List[str]
    confidence: float
    intent: str
    category: str
    sentiment: Dict[str, Any]
    escalated: bool
    ticket_id: Optional[str] = None
    validation: Dict[str, Any]
    retries: int
    llm_model: str


class FeedbackRequest(BaseModel):
    session_id: str = "default"
    query: str = Field(..., min_length=1)
    answer: str = ""
    rating: Literal["up", "down"]
    correction: Optional[str] = Field(None, description="the right answer, if the customer / agent gives one")
    intent: Optional[str] = None


class FeedbackResponse(BaseModel):
    saved: bool
    message: str
