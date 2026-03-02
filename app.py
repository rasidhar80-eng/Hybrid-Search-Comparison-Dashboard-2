from __future__ import annotations

from typing import Dict, List

from flask import Flask, jsonify, request, send_from_directory

from ml_model import ENGINE, DOCUMENTS


app = Flask(
    __name__,
    static_folder="web",      # serves your existing index.html
    static_url_path="",       # so "/" maps to index.html
)

@app.after_request
def add_cors_headers(response):
    # Allow the dashboard to call the API even if index.html is opened via file://
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def index():
    # Serve the dashboard page
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/search")
def search() -> Dict:
    """
    POST /api/search
    JSON body:
      { "query": "...", "mode": "bm25" | "vector" | "hybrid", "top_k": 5, "alpha": 0.5 }
    """
    data = request.get_json(force=True, silent=True) or {}
    query: str = data.get("query", "") or ""
    mode: str = data.get("mode", "hybrid") or "hybrid"
    top_k: int = int(data.get("top_k", 5) or 5)
    alpha: float = float(data.get("alpha", 0.5) or 0.5)

    results = ENGINE.search(query=query, mode=mode, top_k=top_k, alpha=alpha)
    return jsonify({"query": query, "mode": mode, "results": results})


@app.get("/api/documents")
def list_documents() -> Dict:
    docs: List[Dict] = [
        {"doc_id": i, "text": text} for i, text in enumerate(DOCUMENTS)
    ]
    return jsonify({"documents": docs})


@app.get("/api/metrics/demo")
def metrics_demo() -> Dict:
    """
    Tiny synthetic metrics example for recall / noise comparison.
    """
    queries = [
        "semantic search",
        "keyword ranking",
        "hybrid retrieval",
    ]
    relevance = {
        0: [2],      # "semantic search" -> vector doc
        1: [1],      # "keyword ranking" -> BM25 doc
        2: [3, 4],   # "hybrid retrieval" -> hybrid + noise docs
    }
    metrics = ENGINE.evaluate_recall_and_noise(queries, relevance, top_k=5)
    return jsonify({"queries": queries, "metrics": metrics})


if __name__ == "__main__":
    # Run: python app.py
    # Then open http://127.0.0.1:5000 in your browser.
    app.run(debug=True)

