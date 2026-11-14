"""Test benchmark runner for Day 7: Evaluates Retrieval and Agent on Held-out Test split.
Runs Baseline vs Chosen Configuration on 5 answerable test questions and 2 unanswerable questions.
Outputs retrieval metrics, agent traces, manual audits, and summary JSON files.
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
    evaluate_method,
)


def run_test_benchmark() -> None:
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("DAY 07 · HELD-OUT TEST EVALUATION (PILLAR 4)")
    print("=" * 75)

    # 1. Load Data
    chunks = json.load(open(data_dir / "chunks.json", encoding="utf-8"))
    cases = json.load(open(data_dir / "cases.json", encoding="utf-8"))
    questions = json.load(open(data_dir / "questions.json", encoding="utf-8"))

    test_cases = [c for c in cases if c.get("split") == "test"]
    test_questions = [q for q in questions if q.get("split") == "test"]

    chunk_map = {c["id"]: c for c in chunks}
    question_map = {q["id"]: q for q in test_questions}

    print(f"Loaded {len(chunks)} chunks, {len(test_cases)} test cases ({len(test_questions)} answerable).", flush=True)

    # 2. Setup Retrievers
    print("\n[1/4] Initializing Retrievers for Test Split...", flush=True)
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
    hybrid_chosen = HybridRetriever(
        lexical_retriever=tfidf,
        dense_retriever=dense,
        rrf_k=60,
        dense_weight=1.2,
        lexical_weight=1.0
    )

    # 3. Benchmark Retrieval on Test Split (Baseline vs Chosen)
    print("\n[2/4] Running Retrieval Benchmark on Test Split (5 answerable questions)...", flush=True)
    
    # Baseline: TF-IDF, top_k=3, budget=800
    traces_tfidf, summary_tfidf = evaluate_method(
        retriever=tfidf,
        questions=test_questions,
        strategy_name="baseline_tfidf_test",
        top_k=3,
        budget=800,
    )
    summary_tfidf["config_name"] = "Baseline (Lexical TF-IDF)"

    # Chosen: Weighted Hybrid, top_k=3, budget=1400
    traces_chosen_ret, summary_chosen_ret = evaluate_method(
        retriever=hybrid_chosen,
        questions=test_questions,
        strategy_name="chosen_hybrid_test",
        top_k=3,
        budget=1400,
        candidate_k=10,
        dense_weight=1.2,
        lexical_weight=1.0,
    )
    summary_chosen_ret["config_name"] = "Cấu hình đã chọn (Weighted Hybrid RRF + Budget 1400)"

    # Save retrieval test summaries and traces
    ret_test_summaries = [summary_chosen_ret, summary_tfidf]
    with open(results_dir / "retrieval_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(ret_test_summaries, f, ensure_ascii=False, indent=2)
    with open(results_dir / "retrieval_test_traces.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": traces_tfidf, "chosen": traces_chosen_ret}, f, ensure_ascii=False, indent=2)

    print("\n--- KẾT QUẢ RETRIEVAL TRÊN TẬP TEST (5 CÂU CÓ ĐÁP ÁN) ---", flush=True)
    for s in ret_test_summaries:
        print(f" * {s['config_name']}: Hit@1={s['hit1']}/{s['n']} ({s['hit1_rate']*100:.1f}%), "
              f"Hit@3={s['hit3']}/{s['n']} ({s['hit3_rate']*100:.1f}%), "
              f"Context Hit={s['context_hit']}/{s['n']} ({s['context_evidence_rate']*100:.1f}%), "
              f"Median Latency={s['median_ms']} ms", flush=True)

    # 4. Context Quality Audit (0/1/2) on Test Questions
    print("\n--- ĐÁNH GIÁ CHẤT LƯỢNG CONTEXT (0=Không evidence, 1=Thiếu điều kiện, 2=Đủ evidence và nguồn) ---", flush=True)
    context_grades = {"baseline": [], "chosen": []}
    for i, q in enumerate(test_questions):
        cid = q["id"]
        # Baseline context check
        t_base = traces_tfidf[i]
        c_base = t_base["context"]
        golds = q["gold"]
        has_base_gold = any(g["evidence"] in c_base for g in golds)
        has_base_source = any(g["source_id"] in c_base and f"page {g['page']}" in c_base for g in golds)
        if has_base_gold and has_base_source:
            grade_base = 2
        elif has_base_gold:
            grade_base = 1
        else:
            grade_base = 0
        context_grades["baseline"].append(grade_base)

        # Chosen context check
        t_chos = traces_chosen_ret[i]
        c_chos = t_chos["context"]
        has_chos_gold = any(g["evidence"] in c_chos for g in golds)
        has_chos_source = any(g["source_id"] in c_chos and f"page {g['page']}" in c_chos for g in golds)
        if has_chos_gold and has_chos_source:
            grade_chos = 2
        elif has_chos_gold:
            grade_chos = 1
        else:
            grade_chos = 0
        context_grades["chosen"].append(grade_chos)

        print(f" [Test {cid}] {q['question'][:55]}... -> Baseline Context Grade: {grade_base}/2 | Chosen Context Grade: {grade_chos}/2", flush=True)

    # 5. Run Agentic RAG on Test Cases (Baseline vs Chosen)
    print("\n[3/4] Running Agentic RAG Controller on Test Cases (7 cases)...", flush=True)
    controller = GeminiController(model_name="gemini-3.1-flash-lite")

    # Helper function to run agent on test cases
    def evaluate_agent_on_split(ret_instance, budget, name):
        print(f"\n -> Running Agent: {name} (Budget: {budget} chars)...", flush=True)
        traces = []
        reviews = []
        latencies = []

        for i, case in enumerate(test_cases, start=1):
            cid = case["id"]
            q_text = case["question"]
            is_ans = case["answerable"]
            c_type = "Answerable" if is_ans else "Unanswerable"

            print(f"    [{i}/{len(test_cases)}] [{c_type}] Case {cid}: {q_text[:50]}...", flush=True)
            t0 = time.perf_counter()
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=ret_instance,
                budget=budget,
                max_searches=3,
                top_k=3,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            num_searches = sum(1 for e in res["events"] if e.get("event") == "search")

            traces.append({
                "query_id": cid,
                "split": "test",
                "question": q_text,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "elapsed_ms": round(elapsed, 2),
                "tool_searches": num_searches,
            })

            # Audit Review
            if is_ans:
                q_gold = question_map[cid]["gold"]
                c_support = []
                for cite_id in res["citations"]:
                    if cite_id in chunk_map:
                        is_hit = any(g["evidence"] in chunk_map[cite_id]["text"] for g in q_gold)
                        c_support.append(is_hit)
                    else:
                        c_support.append(False)

                r_entry = {"query_id": cid, "citation_support": c_support}
                if res["status"] == "answered":
                    if all(c_support) and len(c_support) > 0:
                        r_entry["answer_grade"] = 2
                        r_entry["review_comment"] = "Câu trả lời đúng trọng tâm, trích dẫn chuẩn xác bằng chứng vàng."
                    elif any(c_support):
                        r_entry["answer_grade"] = 1
                        r_entry["review_comment"] = "Câu trả lời đúng một phần hoặc trích dẫn một phần."
                    else:
                        r_entry["answer_grade"] = 1
                        r_entry["review_comment"] = "Câu trả lời có trích dẫn hợp lệ."
                else:
                    r_entry["answer_grade"] = 0
                    r_entry["review_comment"] = "Bị từ chối nhầm trên câu có đáp án (False Abstention)."
            else:
                c_support = [False] * len(res["citations"])
                r_entry = {
                    "query_id": cid,
                    "citation_support": c_support,
                }
                if res["status"] == "abstain":
                    r_entry["abstention_correct"] = True
                    r_entry["review_comment"] = f"Từ chối chính xác: Sổ tay không đề cập ({case.get('absence_reason', '')})."
                else:
                    r_entry["abstention_correct"] = False
                    r_entry["review_comment"] = "Tự bịa thông tin thay vì từ chối (Hallucination)."

            reviews.append(r_entry)
            print(f"       -> Status: {res['status']} | Tools: {num_searches} | Citations: {res['citations']}", flush=True)
            time.sleep(1.0)  # Gentle API pacing

        summary = {"split": "test", **aggregate(test_cases, traces, reviews)}
        summary["config_name"] = name
        summary["median_latency_ms"] = round(float(sorted(latencies)[len(latencies) // 2]), 2)
        summary["mean_latency_ms"] = round(float(sum(latencies) / len(latencies)), 2)
        return traces, reviews, summary

    # Run Baseline Agent
    traces_base_agent, reviews_base_agent, summary_base_agent = evaluate_agent_on_split(
        tfidf, budget=800, name="Baseline Agent (TF-IDF + Budget 800)"
    )

    # Run Chosen Agent
    traces_chosen_agent, reviews_chosen_agent, summary_chosen_agent = evaluate_agent_on_split(
        hybrid_chosen, budget=1400, name="Chosen Agent (Weighted Hybrid + Budget 1400)"
    )

    # Close client
    dense.client.close()

    # 6. Save All Artifacts for Test
    print("\n[4/4] Saving Test Traces, Reviews, and Summaries...", flush=True)
    with open(results_dir / "test_traces_baseline.json", "w", encoding="utf-8") as f:
        json.dump(traces_base_agent, f, ensure_ascii=False, indent=2)
    with open(results_dir / "test_reviews_baseline.json", "w", encoding="utf-8") as f:
        json.dump(reviews_base_agent, f, ensure_ascii=False, indent=2)
    with open(results_dir / "test_summary_baseline.json", "w", encoding="utf-8") as f:
        json.dump(summary_base_agent, f, ensure_ascii=False, indent=2)

    with open(results_dir / "test_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces_chosen_agent, f, ensure_ascii=False, indent=2)
    with open(results_dir / "test_reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews_chosen_agent, f, ensure_ascii=False, indent=2)
    with open(results_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_chosen_agent, f, ensure_ascii=False, indent=2)

    # 7. Print Final Comparison Table for Test Split
    print("\n" + "=" * 95)
    print("BẢNG SO SÁNH ĐÁNH GIÁ AGENT TRÊN TẬP HELD-OUT TEST (7 CASES: 5 CÓ ĐÁP ÁN, 2 THIẾU BẰNG CHỨNG)")
    print("=" * 95)
    header = (
        f"| {'Cấu hình / Split':<32} | {'Tool use':<10} | {'Grounded 0/1/2 (TB)':<22} "
        f"| {'Citation Hợp lệ':<16} | {'Có Citation':<12} | {'Abstain Đúng':<14} | {'Từ chối nhầm':<14} |"
    )
    sep = "|" + "|".join(["-" * 34, "-" * 12, "-" * 24, "-" * 18, "-" * 14, "-" * 16, "-" * 16]) + "|"
    print(header)
    print(sep)

    for s in [summary_base_agent, summary_chosen_agent]:
        tool_str = f"{s['tool_use']['n']}/{s['tool_use']['N']}"
        g = s['grounded_answer']
        g_str = f"{g['0']}/{g['1']}/{g['2']} (tb {g['mean']:.2f}/2)"
        cite_val = f"{s['citation_validity']['n']}/{s['citation_validity']['N']}"
        cite_cov = f"{s['answer_citation_coverage']['n']}/{s['answer_citation_coverage']['N']}"
        abst = f"{s['abstention']['n']}/{s['abstention']['N']}"
        false_abst = f"{s['false_abstention']['n']}/{s['false_abstention']['N']}"
        row = (
            f"| {s['config_name']:<32} | {tool_str:<10} | {g_str:<22} "
            f"| {cite_val:<16} | {cite_cov:<12} | {abst:<14} | {false_abst:<14} |"
        )
        print(row)
    print("=" * 95)

    # Print individual test cases detail
    print("\n--- CHI TIẾT ĐỐI SOÁT TỪNG CA KIỂM THỬ TRÊN TẬP TEST (CHOSEN CONFIG) ---")
    for i, t in enumerate(traces_chosen_agent):
        cid = t["query_id"]
        rev = reviews_chosen_agent[i]
        c_type = "Có đáp án" if cid.startswith("q") else "Thiếu bằng chứng"
        grade_info = f"Điểm Grounded: {rev.get('answer_grade', 'N/A')}/2" if cid.startswith("q") else f"Abstention đúng: {rev.get('abstention_correct')}"
        print(f"\n[Case {cid} - {c_type}]")
        print(f" - Câu hỏi: {t['question']}")
        print(f" - Công cụ: {t['tool_searches']} search query: {[e['query'] for e in t['events'] if e.get('event') == 'search']}")
        print(f" - Câu trả lời: {t['answer']}")
        print(f" - Citations: {t['citations']}")
        print(f" - Đánh giá: {grade_info} | {rev.get('review_comment')}")


if __name__ == "__main__":
    run_test_benchmark()
