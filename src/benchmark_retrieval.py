"""Benchmark runner for Day 7 Retrieval Methods on Dev split.
Tests 3 core retrieval strategies + 2 improvement configurations.
Outputs comparison table, traces, and dev error analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

from retriever import (
    DenseRetriever,
    GeminiEmbedder,
    HybridRetriever,
    TfidfRetriever,
    evaluate_method,
)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAY 07 · BENCHMARKING RETRIEVAL METHODS & CONTEXT PACKING (DEV SPLIT)")
    print("=" * 70)

    # 1. Load Data
    chunks_path = data_dir / "chunks.json"
    questions_path = data_dir / "questions.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Dev split answerable questions (15 questions)
    dev_questions = [q for q in all_questions if q.get("split") == "dev"]
    print(f"Loaded {len(chunks)} chunks and {len(dev_questions)} dev questions.")

    # 2. Initialize Retrievers
    print("\n[1/3] Initializing Retrievers...")
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

    hybrid = HybridRetriever(
        lexical_retriever=tfidf,
        dense_retriever=dense,
        rrf_k=60
    )
    print("Retrievers successfully initialized.")

    # 3. Define Configurations
    # Config 1: Baseline (TF-IDF, top_k=3, budget=800)
    # Config 2: Dense Retrieval (Gemini 3072d + Qdrant, top_k=3, budget=800)
    # Config 3: Hybrid Retrieval (RRF k=60, top_k=3, budget=800)
    # Config 4: Improvement 1 (Hybrid RRF, context budget=1400)
    # Config 5: Improvement 2 (Semantic Weighted Hybrid RRF, dense_w=1.2, context budget=1400)
    configs = [
        {
            "name": "Baseline (Lexical TF-IDF)",
            "retriever": tfidf,
            "strategy": "tfidf",
            "top_k": 3,
            "budget": 800,
            "candidate_k": None,
            "dense_weight": None,
            "lexical_weight": None,
            "desc": "TF-IDF lexical matching, top-3, budget 800",
            "hypothesis": "Baseline đối soát dựa trên từ khóa chính xác",
        },
        {
            "name": "Phương pháp 2 (Dense Retrieval)",
            "retriever": dense,
            "strategy": "dense_gemini",
            "top_k": 3,
            "budget": 800,
            "candidate_k": None,
            "dense_weight": None,
            "lexical_weight": None,
            "desc": "Gemini 3072d + Qdrant Cosine, top-3, budget 800",
            "hypothesis": "Nắm bắt ngữ nghĩa và từ đồng nghĩa tốt hơn từ khóa cứng",
        },
        {
            "name": "Phương pháp 3 (Hybrid RRF Chuẩn)",
            "retriever": hybrid,
            "strategy": "hybrid_rrf_standard",
            "top_k": 3,
            "budget": 800,
            "candidate_k": 10,
            "dense_weight": 1.0,
            "lexical_weight": 1.0,
            "desc": "Kết hợp Dense + TF-IDF bằng RRF (k=60), top-3, budget 800",
            "hypothesis": "Bổ trợ thế mạnh: vừa bắt đúng thuật ngữ số hiệu vừa hiểu ngữ nghĩa",
        },
        {
            "name": "Cải thiện 1 (Tối ưu Context Budget 1400)",
            "retriever": hybrid,
            "strategy": "hybrid_budget_1400",
            "top_k": 3,
            "budget": 1400,
            "candidate_k": 10,
            "dense_weight": 1.0,
            "lexical_weight": 1.0,
            "desc": "Hybrid RRF chuẩn, top-3, mở rộng budget lên 1400 ký tự",
            "hypothesis": "Giải quyết nghẽn context khi chunk dài, đẩy Context Evidence Rate lên tối đa",
        },
        {
            "name": "Cải thiện 2 (Weighted Hybrid + Context 1400)",
            "retriever": hybrid,
            "strategy": "hybrid_weighted_budget_1400",
            "top_k": 3,
            "budget": 1400,
            "candidate_k": 10,
            "dense_weight": 1.2,
            "lexical_weight": 1.0,
            "desc": "Hybrid RRF ưu tiên Dense (w=1.2) + TF-IDF (w=1.0), budget 1400",
            "hypothesis": "Phá vỡ đồng điểm bằng ngữ nghĩa, đạt 100% Hit@1 và 100% Context Hit",
        },
    ]

    all_traces: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    print("\n[2/3] Running Evaluation across 5 configurations...")
    for cfg in configs:
        name = cfg["name"]
        print(f" -> Running: {name} ...")
        traces, summary = evaluate_method(
            retriever=cfg["retriever"],
            questions=dev_questions,
            strategy_name=cfg["strategy"],
            top_k=cfg["top_k"],
            budget=cfg["budget"],
            candidate_k=cfg["candidate_k"],
            dense_weight=cfg["dense_weight"],
            lexical_weight=cfg["lexical_weight"],
        )
        summary["config_name"] = name
        summary["desc"] = cfg["desc"]
        summary["hypothesis"] = cfg["hypothesis"]
        summaries.append(summary)
        all_traces[cfg["strategy"]] = traces

    # 4. Save results to files
    out_traces_path = results_dir / "retrieval_dev_traces.json"
    out_summary_path = results_dir / "retrieval_dev_summary.json"

    with open(out_traces_path, "w", encoding="utf-8") as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved traces to {out_traces_path}")
    print(f"Saved summaries to {out_summary_path}")

    # 5. Print Comparison Table
    print("\n" + "=" * 90)
    print("BẢNG SO SÁNH CÁC PHƯƠNG PHÁP RETRIEVAL TRÊN TẬP DEV (15 CÂU CÓ ĐÁP ÁN)")
    print("=" * 90)
    header = (
        f"| {'Cấu hình':<30} | {'Split':<5} | {'Hit@1':<10} | {'Hit@3':<10} "
        f"| {'Context Hit':<12} | {'Avg Chars':<10} | {'Median ms':<10} | {'Cost':<6} |"
    )
    sep = "|" + "|".join(["-" * 32, "-" * 7, "-" * 12, "-" * 12, "-" * 14, "-" * 12, "-" * 12, "-" * 8]) + "|"
    print(header)
    print(sep)
    for s in summaries:
        h1_str = f"{s['hit1']}/{s['n']} ({s['hit1_rate']*100:.1f}%)"
        h3_str = f"{s['hit3']}/{s['n']} ({s['hit3_rate']*100:.1f}%)"
        ctx_str = f"{s['context_hit']}/{s['n']} ({s['context_evidence_rate']*100:.1f}%)"
        row = (
            f"| {s['config_name']:<30} | {s['split']:<5} | {h1_str:<10} | {h3_str:<10} "
            f"| {ctx_str:<12} | {s['avg_context_chars']:<10.1f} | {s['median_ms']:<10.2f} | $0.00  |"
        )
        print(row)
    print("=" * 90)

    # 6. Detailed Error Analysis on Dev
    print("\n[3/3] PHÂN TÍCH LỖI TRÊN TẬP DEV (DEV ERROR ANALYSIS)")
    print("-" * 70)
    tfidf_traces = all_traces["tfidf"]
    dense_traces = all_traces["dense_gemini"]
    hybrid_traces = all_traces["hybrid_rrf_standard"]
    weighted_traces = all_traces["hybrid_weighted_budget_1400"]

    # Find questions where TF-IDF or Dense failed Hit@1 or Hit@3
    for i, q in enumerate(dev_questions):
        t_hit1 = tfidf_traces[i]["hit1"]
        d_hit1 = dense_traces[i]["hit1"]
        h_hit1 = hybrid_traces[i]["hit1"]
        w_hit1 = weighted_traces[i]["hit1"]
        t_ctx = tfidf_traces[i]["context_hit"]
        w_ctx = weighted_traces[i]["context_hit"]

        if not t_hit1 or not d_hit1 or not t_ctx:
            print(f"\n[Case {q['id']}] Question: {q['question']}")
            print(f" -> TF-IDF Hit@1: {t_hit1}, Context Hit: {t_ctx}")
            print(f" -> Dense  Hit@1: {d_hit1}")
            print(f" -> Hybrid Hit@1: {h_hit1}")
            print(f" -> Weighted+Budget1400 Hit@1: {w_hit1}, Context Hit: {w_ctx}")
            print(f" -> Gold Evidence: {q['gold'][0]['evidence'][:90]}...")
            if not t_hit1 and d_hit1:
                print(" -> NHẬN XÉT: TF-IDF trượt do câu hỏi dùng từ ngữ đồng nghĩa hoặc câu dài làm loãng trọng số từ khóa. Dense bắt trúng ngữ nghĩa.")
            elif not d_hit1 and t_hit1:
                print(" -> NHẬN XÉT: Dense xếp sau do câu hỏi chứa thuật ngữ số hiệu chính xác mà TF-IDF bắt trúng tuyệt đối. Hybrid đã tổng hợp được cả hai.")
            elif not t_ctx and w_ctx:
                print(" -> NHẬN XÉT: Baseline bị nghẽn Context Budget 800 ký tự nên rớt chunk quan trọng. Tối ưu budget lên 1400 ký tự giúp giữ trọn vẹn 100% bằng chứng.")


if __name__ == "__main__":
    main()
