"""
Evaluation: baseline RAG (semantic top-3 + gemini, nothing else) vs the agentic graph.
Metrics: BLEU, ROUGE-1/2/L, answer relevance (cosine to the reference answer),
context precision (share of retrieved faq chunks with the right intent) and intent accuracy.

run:  python -m src.evaluation
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.config import EVAL_DIR, COMPANY
from src.retrieval import get_retriever
from src.llm import call_llm, parse_json
from src.graph import build_graph, run_query

BASELINE_PROMPT = """You are the customer support assistant of {company}, an Indian e-commerce store.
Answer the customer's question using the context below. Keep it short, plain text, no markdown.

CONTEXT:
{context}

CUSTOMER QUESTION: {question}

Reply with JSON only: {{"answer": "<your answer>", "sources_used": [], "confidence": <0 to 1>}}
"""

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def load_eval_set(path=None):
    path = path or Path(EVAL_DIR) / "eval_questions.json"
    return json.loads(Path(path).read_text())


def baseline_rag(question, k=3):
    # plain RAG: semantic search only, no intent, no validation, no retry, no memory
    chunks = get_retriever().semantic_search(question, k=k)
    context = "\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))
    raw, model = call_llm(BASELINE_PROMPT.format(company=COMPANY, context=context, question=question))
    out = parse_json(raw)
    return {"answer": str(out.get("answer", "")).strip(), "retrieved": chunks, "llm_model": model}


def text_scores(answer, reference):
    ref_tokens = reference.lower().split()
    ans_tokens = answer.lower().split()
    bleu = sentence_bleu([ref_tokens], ans_tokens, smoothing_function=_smooth) if ans_tokens else 0.0
    r = _rouge.score(reference, answer)
    emb = get_retriever().embeddings
    a, b = np.array(emb.embed_query(answer)), np.array(emb.embed_query(reference))
    return {
        "bleu": round(bleu, 4),
        "rouge1": round(r["rouge1"].fmeasure, 4),
        "rouge2": round(r["rouge2"].fmeasure, 4),
        "rougeL": round(r["rougeL"].fmeasure, 4),
        "relevance": round(float(np.dot(a, b)), 4),
    }


def context_precision(retrieved, intent):
    faq = [c for c in retrieved if c.get("intent") not in ("policy", "unknown", None)]
    if not faq:
        return np.nan
    return round(sum(c["intent"] == intent for c in faq) / len(faq), 4)


def evaluate(pipeline, questions=None, verbose=True):
    questions = questions or load_eval_set()
    rows = []
    app = build_graph() if pipeline == "agentic" else None
    for i, q in enumerate(questions):
        if pipeline == "baseline":
            res = baseline_rag(q["question"])
            retrieved, pred_intent = res["retrieved"], None
        else:
            # fresh session per question so memory from one question does not leak into the next
            res = run_query(q["question"], session_id=f"eval-{i}", app=app)
            retrieved = app.get_state({"configurable": {"thread_id": f"eval-{i}"}}).values["retrieved"]
            pred_intent = res["intent"]

        scores = text_scores(res["answer"], q["reference"])
        rows.append({"pipeline": pipeline, "question": q["question"], "intent": q["intent"], **scores,
                     "context_precision": context_precision(retrieved, q["intent"]),
                     "intent_correct": (pred_intent == q["intent"]) if pred_intent else np.nan,
                     "llm_model": res["llm_model"], "answer": res["answer"]})
        if res["llm_model"] not in ("cache", "mock"):
            time.sleep(12)      # free tier: 5 requests per minute on the main model
        if verbose:
            print(f"[{pipeline}] {i + 1}/{len(questions)} bleu={scores['bleu']:.3f} rougeL={scores['rougeL']:.3f} "
                  f"rel={scores['relevance']:.3f} ({res['llm_model']})")
    return pd.DataFrame(rows)


def summarise(df):
    metrics = ["bleu", "rouge1", "rouge2", "rougeL", "relevance", "context_precision", "intent_correct"]
    summary = df.groupby("pipeline")[metrics].mean().round(4)
    return summary.rename(columns={"intent_correct": "intent_accuracy"})


def compare(save=True):
    questions = load_eval_set()
    base = evaluate("baseline", questions)
    agent = evaluate("agentic", questions)
    df = pd.concat([base, agent], ignore_index=True)
    summary = summarise(df)
    if save:
        Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(EVAL_DIR) / "results_per_question.csv", index=False)
        summary.to_csv(Path(EVAL_DIR) / "results_summary.csv")
    return df, summary


if __name__ == "__main__":
    _, s = compare()
    print("\n", s.T)
