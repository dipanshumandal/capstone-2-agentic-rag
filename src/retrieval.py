"""
Hybrid retrieval over the FAISS index: semantic search (MiniLM) + keyword search (BM25),
combined with reciprocal rank fusion and a small re-ranking step.
Also pulls in the human corrections saved by the feedback loop.
"""
import re
import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from src.config import INDEX_DIR, FEEDBACK_DIR, TOP_K
from src.ingestion import get_embeddings


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:

    def __init__(self, index_dir=INDEX_DIR, embeddings=None):
        self.embeddings = embeddings or get_embeddings()
        self.store = FAISS.load_local(str(index_dir), self.embeddings, allow_dangerous_deserialization=True)
        # bm25 needs the raw text of every chunk, pull it out of the docstore once
        self.docs = list(self.store.docstore._dict.values())
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])
        self.corrections = self._load_corrections()

    # ---------- feedback loop ----------
    def _load_corrections(self):
        # corrections given through /feedback, stored with their embedding so a similar
        # question later gets the corrected answer on top
        path = Path(FEEDBACK_DIR) / "corrections.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for r in rows:
            r["vec"] = np.array(self.embeddings.embed_query(r["query"]))
        return rows

    def reload_corrections(self):
        self.corrections = self._load_corrections()

    def matching_corrections(self, query, threshold=0.80):
        if not self.corrections:
            return []
        q = np.array(self.embeddings.embed_query(query))
        hits = []
        for r in self.corrections:
            sim = float(np.dot(q, r["vec"]))
            if sim >= threshold:
                hits.append({
                    "content": f"Verified answer (from support team): {r['correction']}",
                    "source": "human_correction",
                    "chunk_id": "human_correction",
                    "intent": r.get("intent", "unknown"),
                    "category": "feedback",
                    "score": round(sim, 4),
                    "method": "correction",
                })
        return sorted(hits, key=lambda h: -h["score"])

    # ---------- the three searches ----------
    def semantic_search(self, query, k=TOP_K):
        results = self.store.similarity_search_with_score(query, k=k)
        out = []
        for doc, dist in results:
            sim = 1 - (dist ** 2) / 2      # faiss gives L2 distance, vectors are normalised so this is cosine
            out.append(self._to_dict(doc, sim, "semantic"))
        return out

    def keyword_search(self, query, k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [self._to_dict(self.docs[i], float(scores[i]), "bm25") for i in top if scores[i] > 0]

    def hybrid_search(self, query, k=TOP_K, candidates=None):
        # take more candidates than k from each side, fuse them, then re-rank
        candidates = candidates or k * 3
        sem = self.semantic_search(query, k=candidates)
        kw = self.keyword_search(query, k=candidates)

        fused = {}
        for rank, r in enumerate(sem):
            fused.setdefault(r["chunk_id"], {**r, "rrf": 0.0, "sem": r["score"], "bm25": 0.0})
            fused[r["chunk_id"]]["rrf"] += 1 / (60 + rank)
        for rank, r in enumerate(kw):
            fused.setdefault(r["chunk_id"], {**r, "rrf": 0.0, "sem": 0.0, "bm25": 0.0})
            fused[r["chunk_id"]]["rrf"] += 1 / (60 + rank)
            fused[r["chunk_id"]]["bm25"] = r["score"]

        # re-rank: mostly cosine, a bit of normalised bm25 and the fusion score
        max_bm25 = max([v["bm25"] for v in fused.values()] + [1e-9])
        for v in fused.values():
            if v["sem"] == 0.0:    # came only from bm25, compute its cosine so it can be compared
                v["sem"] = self._cosine(query, v["content"])
            v["score"] = round(0.7 * v["sem"] + 0.2 * (v["bm25"] / max_bm25) + 0.1 * (v["rrf"] * 60), 4)
            v["method"] = "hybrid"

        ranked = sorted(fused.values(), key=lambda v: -v["score"])[:k]
        for v in ranked:
            v.pop("rrf", None)
        return self.matching_corrections(query) + ranked

    # ---------- helpers ----------
    def _cosine(self, query, text):
        a = np.array(self.embeddings.embed_query(query))
        b = np.array(self.embeddings.embed_query(text))
        return float(np.dot(a, b))

    def _to_dict(self, doc, score, method):
        return {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "intent": doc.metadata.get("intent"),
            "category": doc.metadata.get("category"),
            "score": round(float(score), 4),
            "method": method,
        }


_retriever = None


def get_retriever():
    # loaded once and shared by the tools, the API and the tests
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
