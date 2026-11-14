"""Gemini LLM Controller and Agent Orchestrator for Day 7.
Connects real Gemini LLM with Weighted Hybrid Retriever and Context Packing.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Try importing google-genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from retriever import (
    DenseRetriever,
    GeminiEmbedder,
    HybridRetriever,
    TfidfRetriever,
    pack_context,
)


class GeminiController:
    """Controller using Google Gemini API to make search/answer decisions."""

    SYSTEM_PROMPT = """Bạn là Trợ lý AI tra cứu Sổ tay sinh viên chính thức của trường đại học.
Nhiệm vụ của bạn là trả lời các câu hỏi dựa HOÀN TOÀN vào các đoạn văn tài liệu (chunks) được cung cấp.

Quy tắc bắt buộc:
1. Bạn phải phản hồi bằng DUY NHẤT một đối tượng JSON hợp lệ (không kèm Markdown giải thích ngoài JSON).
2. Nếu chưa có ngữ cảnh hoặc ngữ cảnh chưa đủ để trả lời và bạn muốn tìm kiếm thêm, hãy chọn:
   {"action": "search", "query": "<từ khóa tìm kiếm tiếng Việt ngắn gọn, đúng trọng tâm>"}
3. Nếu ngữ cảnh đã chứa đầy đủ bằng chứng để trả lời câu hỏi:
   - Viết câu trả lời ngắn gọn, chuẩn xác bằng tiếng Việt.
   - BẮT BUỘC chèn mã trích dẫn dạng [chunk-id] ngay sau thông tin được trích dẫn (ví dụ: "...đạt tối thiểu 600 điểm TOEIC [student-handbook-2024_p01_par_001].").
   - Liệt kê tất cả các chunk-id đã dùng vào danh sách citations.
   {"action": "answer", "answer": "<câu trả lời kèm [chunk-id]>", "citations": ["<chunk-id>"]}
4. Nếu ngữ cảnh KHÔNG chứa thông tin hoặc câu hỏi hỏi về điều không có trong tài liệu:
   - TUYỆT ĐỐI KHÔNG tự bịa ra thông tin (hallucination).
   - Hãy từ chối trả lời bằng định dạng chuẩn:
   {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
"""

    def __init__(self, model_name: str = "gemini-3.1-flash-lite") -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        if genai is None:
            raise ImportError("google-genai is not installed.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def _clean_json_text(self, text: str) -> str:
        text = text.strip()
        # Remove markdown code blocks if wrapped
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Calls Gemini model to determine next action: search or answer."""
        prompt_parts = [self.SYSTEM_PROMPT, f"\nCâu hỏi của sinh viên: {question}"]

        if not history:
            prompt_parts.append("\nHiện tại bạn chưa có tài liệu nào trong ngữ cảnh. Hãy chọn hành động 'search' với câu truy vấn tối ưu.")
        else:
            prompt_parts.append("\nLịch sử tra cứu và Ngữ cảnh tài liệu hiện có:")
            for h in history:
                prompt_parts.append(f"- Các truy vấn đã tìm: {', '.join(h.get('queries', []))}")
                prompt_parts.append(f"- Ngữ cảnh trích xuất:\n{h.get('context', '(Trống)')}")
            prompt_parts.append("\nHãy phân tích ngữ cảnh trên. Nếu đủ bằng chứng, hãy trả lời kèm trích dẫn [chunk-id]. Nếu tài liệu không có, hãy từ chối trả lời.")

        user_content = "\n".join(prompt_parts)

        # Call Gemini with retry logic
        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = self._clean_json_text(response.text or "")
                data = json.loads(raw_text)

                action = data.get("action")
                if action == "search":
                    query = data.get("query", "").strip()
                    if not query:
                        query = question
                    return {"action": "search", "query": query}
                elif action == "answer":
                    answer = data.get("answer", "").strip()
                    citations = data.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []
                    # Normalize abstention
                    if "không đủ bằng chứng" in answer.lower() or "không có thông tin" in answer.lower():
                        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
                    return {"action": "answer", "answer": answer, "citations": citations}
                else:
                    # Fallback to search if unclear
                    return {"action": "search", "query": question}
            except Exception as e:
                if attempt == 3:
                    # On final failure, fallback safely
                    if not history:
                        return {"action": "search", "query": question}
                    return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
                time.sleep(1.5 * attempt)

        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


def run_agent(
    question: str,
    controller: Any,
    retriever: Any,
    *,
    filters: dict[str, Any] | None = None,
    max_searches: int = 3,
    top_k: int = 3,
    budget: int = 1400,
    dense_weight: float = 1.2,
    lexical_weight: float = 1.0,
) -> dict[str, Any]:
    """Agent loop strictly adhering to Day 7 starter contract.

    Returns answer, citation IDs and trace; no benchmark gold enters this loop.
    A final answer without valid citations, an empty context, or reaching search limit leads to abstention.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("question, max_searches, top_k hoặc budget không hợp lệ")

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)

        if action.get("action") == "search":
            if sum(e["event"] == "search" for e in events) >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                }

            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query phải là chuỗi không rỗng")

            # Execute search with weighted hybrid if available
            if hasattr(retriever, "search"):
                try:
                    results = retriever.search(
                        query,
                        filters=filters,
                        top_k=top_k,
                        dense_weight=dense_weight,
                        lexical_weight=lexical_weight,
                    )
                except TypeError:
                    results = retriever.search(query, filters=filters, top_k=top_k)
            else:
                raise ValueError("Retriever must have search method")

            if len(results) > top_k or len({r["id"] for r in results}) != len(results):
                raise ValueError("retriever phải trả tối đa top_k chunk ID duy nhất")

            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "search",
                "query": query,
                "filter": filters,
                "results": results,
                "context": context,
                "context_ids": ids,
            })

            # Pass cumulative packed context in history
            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e["event"] == "search"],
                "context": context,
                "context_ids": ids,
            }]

        elif action.get("action") == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError("answer/citations không hợp lệ")

            # Check if answer is grounded with citations in packed context
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))
            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer"}],
                }

            # If model abstains or citations are missing/invalid
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations"}],
            }

        else:
            raise ValueError("action phải là search hoặc answer")

    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events,
    }
