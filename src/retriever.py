"""Core Retrieval Module for Day 7: Lexical, Dense, and Hybrid (RRF) Retrievers.
Compatible with standard library baseline and Day 6 Qdrant + Gemini Embedder.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Try importing google-genai
try:
    from google import genai
except ImportError:
    genai = None

# Try importing qdrant_client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
except ImportError:
    QdrantClient = None


def normalize_vietnamese(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_vietnamese(text))


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Check if chunk matches source_id, page and contains exact evidence substring."""
    return any(
        chunk.get("source_id") == g.get("source_id")
        and chunk.get("page") == g.get("page")
        and g.get("evidence", "") in chunk.get("text", "")
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Pack top retrieved chunks into context string within character budget."""
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()

    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        source = c.get("source", "source")
        page = c.get("page", 1)
        block = f"[{cid} | {source} | page {page}]\n{c.get('text', '')}"
        
        candidate = "\n\n".join(blocks + [block]) if blocks else block
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(cid)
        else:
            # Context budget reached
            continue

    return "\n\n".join(blocks), ids


# ============================================================================
# 1. Lexical Retriever (TF-IDF Baseline)
# ============================================================================
class TfidfRetriever:
    """Lexical retriever using TF-IDF and cosine similarity."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results: list[dict[str, Any]] = []

        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                match = all(chunk.get(k) == v or chunk.get("metadata", {}).get(k) == v for k, v in filters.items())
                if not match:
                    continue

            score = sum(val * vector.get(term, 0.0) for term, val in query_vec.items())
            if score > 0:
                item = dict(chunk)
                item["score"] = float(score)
                item["strategy"] = "tfidf"
                results.append(item)

        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


# ============================================================================
# 2. Dense Retriever (Gemini Embedding 3072d + Qdrant)
# ============================================================================
class GeminiEmbedder:
    """Embedder using Google Gemini Embedding API with persistent disk cache."""

    def __init__(self, model_name: str = "models/gemini-embedding-001", cache_path: Path | str | None = None) -> None:
        self.model_name = model_name
        self.dimensions = 3072

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key) if (genai and api_key) else None

        if cache_path:
            self.cache_file = Path(cache_path)
        else:
            self.cache_file = Path("data/cache/gemini_embeddings_cache.json")

        self.cache: dict[str, list[float]] = {}
        self._load_cache()

    def _hash(self, text: str) -> str:
        return hashlib.sha256((self.model_name + "::" + text).encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception:
            pass

    def embed_text(self, text: str) -> list[float]:
        h = self._hash(text)
        if h in self.cache:
            return self.cache[h]

        if not self.client:
            raise RuntimeError("GEMINI_API_KEY not configured and embedding not found in cache.")

        # API call with exponential backoff
        for attempt in range(1, 5):
            try:
                resp = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                if hasattr(resp, "embeddings") and resp.embeddings:
                    vec = list(resp.embeddings[0].values)
                elif hasattr(resp, "embedding"):
                    vec = list(resp.embedding.values)
                else:
                    raise ValueError(f"Unexpected embed_content response format: {resp}")
                self.cache[h] = vec
                self._save_cache()
                return vec
            except Exception as e:
                if attempt == 4:
                    raise e
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to compute embedding.")


class DenseRetriever:
    """Dense retriever using Qdrant vector store and Gemini embeddings."""

    def __init__(self, qdrant_path: Path | str, collection_name: str, embedder: GeminiEmbedder, chunks: list[dict[str, Any]]) -> None:
        if QdrantClient is None:
            raise ImportError("qdrant_client is required for DenseRetriever.")

        self.qdrant_path = str(qdrant_path)
        self.collection_name = collection_name
        self.embedder = embedder
        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}

        self.client = QdrantClient(path=self.qdrant_path)
        self._init_index()

    def _init_index(self) -> None:
        """Ensure collection exists and is populated with all chunks."""
        exists = self.client.collection_exists(self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedder.dimensions, distance=Distance.COSINE),
            )

        count = self.client.get_collection(self.collection_name).points_count
        if count < len(self.chunks):
            # Populate index
            points: list[PointStruct] = []
            for c in self.chunks:
                vec = self.embedder.embed_text(c["text"])
                p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.collection_name}::{c['id']}"))
                points.append(PointStruct(id=p_id, vector=vec, payload=c))
            self.client.upsert(self.collection_name, points=points)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed_text(query)

        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            query_filter = Filter(must=conditions)

        query_res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
        )

        results: list[dict[str, Any]] = []
        for hit_pt in query_res.points:
            item = dict(hit_pt.payload)
            item["score"] = float(hit_pt.score)
            item["strategy"] = "dense"
            results.append(item)

        return results


# ============================================================================
# 3. Hybrid Retriever (Reciprocal Rank Fusion - RRF)
# ============================================================================
class HybridRetriever:
    """Combines Dense and Lexical retrievers via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        lexical_retriever: TfidfRetriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.lexical = lexical_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.chunks_by_id = lexical_retriever.chunk_by_id

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
        candidate_k: int = 10,
        dense_weight: float | None = None,
        lexical_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        w_dense = self.dense_weight if dense_weight is None else dense_weight
        w_lex = self.lexical_weight if lexical_weight is None else lexical_weight

        # Fetch candidate sets
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_results = self.lexical.search(query, filters=filters, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}

        # Dense rank contributions
        for rank, item in enumerate(dense_results, start=1):
            cid = item["id"]
            items[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_dense / (self.rrf_k + rank)

        # Lexical rank contributions
        for rank, item in enumerate(lexical_results, start=1):
            cid = item["id"]
            if cid not in items:
                items[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_lex / (self.rrf_k + rank)

        # Sort combined results by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))

        results: list[dict[str, Any]] = []
        for cid in sorted_cids[:top_k]:
            res = dict(items[cid])
            res["score"] = float(rrf_scores[cid])
            res["strategy"] = "hybrid_rrf"
            results.append(res)

        return results


# ============================================================================
# 4. Benchmark Runner & Evaluator
# ============================================================================
def evaluate_method(
    retriever: Any,
    questions: list[dict[str, Any]],
    strategy_name: str,
    top_k: int = 3,
    budget: int = 800,
    filters: dict[str, Any] | None = None,
    candidate_k: int | None = None,
    dense_weight: float | None = None,
    lexical_weight: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluates a retrieval method on a list of answerable questions."""
    traces = []
    latencies = []

    for q in questions:
        start_time = time.perf_counter()

        if strategy_name.startswith("hybrid") and candidate_k is not None:
            results = retriever.search(
                q["question"],
                filters=filters,
                top_k=top_k,
                candidate_k=candidate_k,
                dense_weight=dense_weight,
                lexical_weight=lexical_weight,
            )
        else:
            results = retriever.search(q["question"], filters=filters, top_k=top_k)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]

        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked[:3])
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": strategy_name,
            "top_k": top_k,
            "budget_chars": budget,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
        })

    n = len(traces)
    summary = {
        "strategy": strategy_name,
        "split": questions[0]["split"] if questions else "unknown",
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4) if n else 0.0,
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4) if n else 0.0,
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_evidence_rate": round(sum(t["context_hit"] for t in traces) / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "service_cost_usd": 0.0,
    }

    return traces, summary
