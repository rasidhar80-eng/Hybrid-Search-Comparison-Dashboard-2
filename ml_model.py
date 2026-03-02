from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SearchMode = Literal["bm25", "vector", "hybrid"]


@dataclass
class SearchResult:
    rank: int
    doc_id: int
    text: str
    score: float
    bm25_score: float
    vector_score: float


class HybridSearchEngine:
    """
    Simple hybrid search over an in‑memory list of documents.

    - BM25 using rank_bm25 (keyword match)
    - Vector search using TF‑IDF + cosine similarity
    - Hybrid score = alpha * bm25 + (1 - alpha) * vector
    """

    def __init__(self, documents: List[str]) -> None:
        self.documents = documents

        # --- BM25 index ---
        tokenized_docs = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized_docs)

        # --- TF‑IDF vector index (for "vector"/semantic style search) ---
        self._tfidf = TfidfVectorizer()
        tfidf_matrix = self._tfidf.fit_transform(documents)
        # L2‑normalize each document vector
        self._doc_matrix = normalize(tfidf_matrix)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v = scores.min()
        max_v = scores.max()
        if max_v - min_v < 1e-8:
            # Avoid divide‑by‑zero; treat all as equal
            return np.ones_like(scores)
        return (scores - min_v) / (max_v - min_v)

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict]:
        query = (query or "").strip()
        if not query:
            return []

        # --- BM25 keyword scores ---
        tokens = query.lower().split()
        bm25_raw = np.array(self._bm25.get_scores(tokens))
        bm25_norm = self._normalize(bm25_raw)

        # --- Vector TF‑IDF cosine similarity ---
        q_vec = self._tfidf.transform([query])
        q_vec = normalize(q_vec)
        vec_raw = (self._doc_matrix @ q_vec.T).toarray().ravel()
        vec_norm = self._normalize(vec_raw)

        if mode == "bm25":
            final_scores = bm25_norm
        elif mode == "vector":
            final_scores = vec_norm
        elif mode == "hybrid":
            final_scores = alpha * bm25_norm + (1.0 - alpha) * vec_norm
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_indices = np.argsort(-final_scores)[:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(idx),
                    "text": self.documents[idx],
                    "score": float(final_scores[idx]),
                    "bm25_score": float(bm25_norm[idx]),
                    "vector_score": float(vec_norm[idx]),
                }
            )
        return results

    def evaluate_recall_and_noise(
        self,
        queries: List[str],
        relevance: Dict[int, List[int]],
        top_k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Tiny synthetic evaluation helper.

        relevance: maps query index -> list of relevant doc_ids
        Returns average recall and noise for each mode.
        """
        modes: List[SearchMode] = ["bm25", "vector", "hybrid"]
        metrics: Dict[str, Dict[str, float]] = {
            m: {"recall": 0.0, "noise": 0.0} for m in modes
        }
        n_eval = 0

        for qi, q in enumerate(queries):
            if qi not in relevance or not relevance[qi]:
                continue

            rel_set = set(relevance[qi])
            n_rel = len(rel_set)
            if n_rel == 0:
                continue

            n_eval += 1

            for mode in modes:
                results = self.search(q, mode=mode, top_k=top_k)
                retrieved_ids = [r["doc_id"] for r in results]
                retrieved_set = set(retrieved_ids)

                hits = len(rel_set & retrieved_set)
                recall = hits / n_rel

                non_relevant = len(retrieved_set - rel_set)
                noise = non_relevant / max(len(retrieved_set), 1)

                metrics[mode]["recall"] += recall
                metrics[mode]["noise"] += noise

        if n_eval == 0:
            return metrics

        for mode in modes:
            metrics[mode]["recall"] /= n_eval
            metrics[mode]["noise"] /= n_eval

        return metrics


# Example in‑memory corpus (you can replace this with your own)
DOCUMENTS: List[str] = [
    "Neural networks are a class of machine learning models.",
    "BM25 is a ranking function used by many search engines.",
    "Vector search uses TF-IDF embeddings for similarity.",
    "Hybrid retrieval combines keyword and vector scores.",
    "Noise in retrieval refers to irrelevant yet retrieved documents.",
    "Recall measures how many relevant documents were found.",
    "Transformers achieve state-of-the-art in many NLP tasks.",
]

ENGINE = HybridSearchEngine(DOCUMENTS)


if __name__ == "__main__":
    # Small manual smoke test
    engine = ENGINE
    q = "semantic search with bm25"
    for m in ["bm25", "vector", "hybrid"]:
        print(f"\n=== {m.upper()} ===")
        for r in engine.search(q, mode=m, top_k=3):
            print(f"[{r['rank']}] ({r['score']:.3f}) {r['text']}")

