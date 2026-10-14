# ShopEase Customer Support - Agentic RAG

Capstone Project 2 (Use case C - Enterprise Customer Support Automation). A multi-agent RAG system
for the support desk of a made-up e-commerce store "ShopEase". The knowledge base is the Bitext
customer support dataset (27 intents) plus 6 policy documents I wrote for the store. The agents are
orchestrated with LangGraph, answers are generated with Gemini, exposed through FastAPI and a
Streamlit chat UI, and tested with GitHub Actions.

```
Capstone-2 Agentic RAG/
│
├── data
|    ├── raw/                        # bitext csv (not in git, 18 MB)
|    ├── policies/                   # 6 ShopEase policy files (.md)
|    ├── index/                      # faiss index + intent classifier (small, kept in git)
|    ├── eval/                       # 20 held out questions + evaluation results
|    ├── feedback/                   # thumbs up/down and corrections (created at runtime)
|    ├── llm_cache/                  # cached gemini responses (created at runtime)
├── notebooks
|    ├── Customer Support Agentic RAG.ipynb   # section 1 walkthrough + evaluation
├── src
|    ├── config.py                   # paths, model names, thresholds
|    ├── ingestion.py                # load, clean, chunk, embed, faiss, intent classifier
|    ├── retrieval.py                # hybrid search (faiss + bm25 + re-rank), corrections
|    ├── llm.py                      # gemini wrapper with cache, fallback model and mock mode
|    ├── tools.py                    # agents exposed as tools (rag_search, classify_intent, ...)
|    ├── agents.py                   # graph nodes: planner, sentiment, retriever, validator, rewriter, responder, escalation
|    ├── graph.py                    # langgraph workflow + run_query()
|    ├── feedback.py                 # feedback storage
|    ├── evaluation.py               # BLEU / ROUGE / relevance, baseline vs agentic
├── api
|    ├── main.py                     # FastAPI - /query, /health, /feedback
|    ├── schemas.py                  # pydantic models
├── ui
|    ├── main.py                     # streamlit chat
├── tests                           # pytest - retriever, tools, graph, api
├── .github/workflows/ci.yml        # ruff + pytest on every push / PR
├── requirements.txt
├── .env.example
└── README.md
```

## How it works

```
START -> planner -> sentiment -> retriever -> validator --(ok)--> responder -> escalation -> END
                                     ^             |
                                     |          (weak context, max 2 retries)
                                  rewriter <-------+
```

| Agent | What it does | LLM? |
|---|---|---|
| planner (Query Understanding) | predicts intent + category with a logistic regression on MiniLM embeddings, makes the plan, expands the search query | no |
| sentiment | VADER sentiment + urgency keywords | no |
| retriever | hybrid search: FAISS semantic + BM25 keyword, reciprocal rank fusion, re-rank | no |
| validator | checks relevance (cosine of best chunk) and grounding (chunks match the intent) | no |
| rewriter | rewrites the query and increases k when validation fails | no |
| responder | Gemini answers from the context + chat history, returns JSON with answer, sources and confidence | yes |
| escalation | opens a ticket when the customer asks for a human, is angry and urgent, or the answer is weak | no |

Only the responder calls Gemini, because the free tier allows very few requests per day. Every
response is cached on disk and the lite model is used as fallback when the quota is finished.
Each agent is also a LangChain `@tool` (see `src/tools.py`), so agents call each other as tools.
Memory is per `session_id` through the LangGraph checkpointer.

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
cp .env.example .env        # add the gemini key
```

The index is already in `data/index`, so the API works right after cloning. To rebuild it, run the
first cells of the notebook (they download the Bitext csv into `data/raw/`), or:

```
python -m src.ingestion
```

## Section 1 - RAG + agents

Open `notebooks/Customer Support Agentic RAG.ipynb`. The evaluation can also be run directly:

```
python -m src.evaluation
```

## Section 2 - API and chat UI

```
uvicorn api.main:app --reload --port 8001
streamlit run ui/main.py
```

Swagger UI: http://127.0.0.1:8001/docs

Request:
```json
POST /query
{"session_id": "abc123", "query": "How much does same day delivery cost?"}
```

Response (shortened):
```json
{
  "answer": "Same day delivery costs Rs. 149 and is available in Bangalore, Mumbai, Delhi, Hyderabad, Chennai and Pune if you order before 12 PM ...",
  "sources": ["shipping_and_delivery.md#1"],
  "retrieved_context": ["..."],
  "agents_invoked": ["planner", "sentiment", "retriever", "validator", "responder", "escalation"],
  "confidence": 0.84,
  "intent": "delivery_options",
  "sentiment": {"sentiment": "neutral", "score": 0.0, "urgency": "low"},
  "escalated": false
}
```

## Section 3 - CI

`.github/workflows/ci.yml` runs on every push and pull request to main: install, `ruff check`,
`pytest`. The tests run with `MOCK_LLM=1` so no API key is needed in GitHub.
Docker was left out of this submission (see the report).

```
MOCK_LLM=1 pytest -v tests/
```
