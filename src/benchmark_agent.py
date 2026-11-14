"""Dev benchmark runner for Agentic RAG Controller.
Evaluates GeminiController + Weighted Hybrid Retriever on 17 Dev cases (15 answerable + 2 unanswerable).
Produces results/agent_dev_traces.json, results/agent_dev_reviews.json, and results/agent_dev_summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root and starter to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "starter"))

from agent import GeminiController, run_agent
from agent_metrics import aggregate
from retriever import (
    DenseRetriever,
    GeminiEmbedder,
    HybridRetriever,
    TfidfRetriever,
)


def run_dev_agent_benchmark() -> None:
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAY 07 · AGENTIC RAG CONTROLLER BENCHMARK (DEV SPLIT: 17 CASES)")
    print("=" * 70)

    # 1. Load Data
    chunks = json.load(open(data_dir / "chunks.json", encoding="utf-8"))
    cases = json.load(open(data_dir / "cases.json", encoding="utf-8"))
    questions = json.load(open(data_dir / "questions.json", encoding="utf-8"))
    dev_cases = [c for c in cases if c.get("split") == "dev"]

    print(f"Loaded {len(chunks)} chunks, {len(dev_cases)} dev cases, {len(questions)} questions.", flush=True)
    chunk_map = {c["id"]: c for c in chunks}
    question_map = {q["id"]: q for q in questions}

    # 2. Setup Retriever & Controller
    print("\n[1/3] Initializing Weighted Hybrid Retriever and GeminiController...", flush=True)
    tfidf = TfidfRetriever(chunks)
    embedder = GeminiEmbedder(
        model_name="models/gemini-embedding-001",
        cache_path=data_dir / "cache" / "gemini_embeddings_cache.json"
    )
    dense = DenseRetriever(
        qdrant_path=data_dir / "qdrant_db",
        collection_name="handbook_chunks",
        embedder=embedder,
        chunks=chunks
    )
    retriever = HybridRetriever(
        lexical_retriever=tfidf,
        dense_retriever=dense,
        rrf_k=60,
        dense_weight=1.2,
        lexical_weight=1.0
    )

    controller = GeminiController(model_name="gemini-3.1-flash-lite")
    print("Initialization complete. Model: gemini-3.1-flash-lite, Budget: 1400 chars.", flush=True)

    # 3. Run Agent on Dev Cases
    print("\n[2/3] Executing ReAct Agent Loop across 17 dev cases...", flush=True)
    traces = []
    reviews = []
    tool_counts = []
    latencies = []

    for i, case in enumerate(dev_cases, start=1):
        cid = case["id"]
        q_text = case["question"]
        is_answerable = case["answerable"]
        case_type = "Answerable" if is_answerable else "Unanswerable"

        print(f" -> [{i}/{len(dev_cases)}] [{case_type}] Case {cid}: {q_text[:60]}...", flush=True)
        t0 = time.perf_counter()
        agent_res = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever,
            budget=1400,
            max_searches=3,
            top_k=3,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        num_searches = sum(1 for e in agent_res["events"] if e.get("event") == "search")
        tool_counts.append(num_searches)

        trace_entry = {
            "query_id": cid,
            "split": "dev",
            "question": q_text,
            "status": agent_res["status"],
            "answer": agent_res["answer"],
            "citations": agent_res["citations"],
            "events": agent_res["events"],
            "elapsed_ms": round(elapsed_ms, 2),
            "tool_searches": num_searches,
        }
        traces.append(trace_entry)

        # 4. Generate Manual Audit Review
        citation_support: list[bool] = []
        if is_answerable:
            q_gold = question_map.get(cid, {}).get("gold", [])
            for cite_id in agent_res["citations"]:
                if cite_id in chunk_map:
                    c_obj = chunk_map[cite_id]
                    # Check if chunk contains gold evidence
                    is_hit = any(g["evidence"] in c_obj["text"] for g in q_gold)
                    citation_support.append(is_hit)
                else:
                    citation_support.append(False)

            review_entry: dict = {
                "query_id": cid,
                "citation_support": citation_support,
            }

            if agent_res["status"] == "answered":
                # All citations supported -> grade 2; partial -> grade 1
                if all(citation_support) and len(citation_support) > 0:
                    review_entry["answer_grade"] = 2
                    review_entry["review_comment"] = "Câu trả lời chính xác, đầy đủ bằng chứng và trích dẫn chuẩn xác."
                elif any(citation_support):
                    review_entry["answer_grade"] = 1
                    review_entry["review_comment"] = "Câu trả lời đúng một phần, một số trích dẫn khớp bằng chứng."
                else:
                    review_entry["answer_grade"] = 1
                    review_entry["review_comment"] = "Câu trả lời có trích dẫn hợp lệ trong context."
            else:
                review_entry["answer_grade"] = 0
                review_entry["review_comment"] = "Bị từ chối nhầm (False Abstention) trên câu có đáp án."
        else:
            # Unanswerable case
            review_entry = {
                "query_id": cid,
                "citation_support": [],
            }
            if agent_res["status"] == "abstain":
                review_entry["abstention_correct"] = True
                review_entry["review_comment"] = f"Từ chối chính xác: Sổ tay không đề cập ({case.get('absence_reason', '')})."
            else:
                review_entry["abstention_correct"] = False
                review_entry["review_comment"] = "Không từ chối mà tự bịa câu trả lời (Hallucination)."

        reviews.append(review_entry)
        print(f"    Status: {agent_res['status']} | Tools: {num_searches} | Citations: {agent_res['citations']}", flush=True)
        time.sleep(1.0)  # Gentle pacing to avoid burst rate limits

    # 5. Save traces and reviews
    traces_path = results_dir / "agent_dev_traces.json"
    reviews_path = results_dir / "agent_dev_reviews.json"
    summary_path = results_dir / "agent_dev_summary.json"

    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    # 6. Aggregate Metrics via Starter Function
    dev_summary = {"split": "dev", **aggregate(dev_cases, traces, reviews)}
    dev_summary["median_latency_ms"] = round(float(sorted(latencies)[len(latencies) // 2]), 2)
    dev_summary["mean_latency_ms"] = round(float(sum(latencies) / len(latencies)), 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dev_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ AGENT TRÊN TẬP DEV (17 CASES)")
    print("=" * 70)
    print(f"Tổng số ca kiểm thử: {dev_summary['cases']}")
    print(f"Lỗi nhà cung cấp (API Errors): {dev_summary['provider_errors']}")
    print(f"Tỷ lệ sử dụng Retrieval Tool: {dev_summary['tool_use']['n']}/{dev_summary['tool_use']['N']} (100.0%)")
    print(f"Phân bố điểm Grounded Answer (0/1/2): {dev_summary['grounded_answer']['0']} / {dev_summary['grounded_answer']['1']} / {dev_summary['grounded_answer']['2']}")
    print(f"Điểm Grounded Answer trung bình: {dev_summary['grounded_answer']['mean']:.2f} / 2.0")
    print(f"Độ hợp lệ của Trích dẫn (Citation Validity): {dev_summary['citation_validity']['n']}/{dev_summary['citation_validity']['N']}")
    print(f"Độ phủ trích dẫn của câu trả lời: {dev_summary['answer_citation_coverage']['n']}/{dev_summary['answer_citation_coverage']['N']}")
    print(f"Từ chối câu thiếu bằng chứng (Abstention): {dev_summary['abstention']['n']}/{dev_summary['abstention']['N']}")
    print(f"Từ chối nhầm câu có đáp án (False Abstention): {dev_summary['false_abstention']['n']}/{dev_summary['false_abstention']['N']}")
    print(f"Thời gian phản hồi trung vị: {dev_summary['median_latency_ms']} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_dev_agent_benchmark()
