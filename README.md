# Day 07 — Retrieval & Context Engineering (Agentic RAG)

- **Học viên:** Trần Thanh Tùng
- **Mã số sinh viên (MSSV):** 2A202603418
- **Lớp:** K5-L3A
- **Báo cáo cá nhân chi tiết:** [report/individual/2A202603418_TranThanhTung.md](report/individual/2A202603418_TranThanhTung.md)
- **Repository:** https://github.com/tungtran306/K5-L3A-Day07-TranThanhTung-2A202603418-Retrieval-Context-Engineering

---

## 1. Tổng quan Dự án

Hệ thống **Agentic RAG** tra cứu Sổ tay sinh viên đại học (32 trang nội dung, 145 đoạn văn cấu trúc), sử dụng:
- **LLM Controller thật:** Google Gemini (`gemini-3.1-flash-lite`) điều phối chu trình ReAct, quyết định tra cứu (`search`) hoặc tổng hợp câu trả lời (`answer`) kèm trích dẫn `[chunk-id]`.
- **Vector Store & Embedder thật:** Qdrant Vector Store kết hợp Gemini Embeddings (`models/gemini-embedding-001`, 3072 chiều) với bộ nhớ đệm cache cục bộ (300 vectors).
- **Bộ 3 Phương pháp Retrieval cốt lõi:**
  1. *Lexical Baseline (TF-IDF)*: Đối soát từ khóa tiếng Việt (Unicode NFC).
  2. *Dense Retrieval*: Truy vấn tương đồng Cosine trên không gian vector 3072 chiều.
  3. *Weighted Hybrid Retrieval (RRF)*: Thuật toán Reciprocal Rank Fusion ($w_{\text{dense}}=1.2, w_{\text{lexical}}=1.0, k=60$).
- **Context Engineering:** Mở rộng Context Budget từ 800 lên 1400 ký tự giúp tăng vọt `Context Evidence Rate` từ 40% lên **100%**.

---

## 2. Hướng dẫn Chạy Tái lập (Quickstart)

### Cài đặt môi trường
```bash
# Tạo môi trường ảo và cài đặt thư viện cần thiết
pip install qdrant-client google-genai python-dotenv
```

### Thiết lập API Key
Tạo file `.env` tại thư mục gốc (hoặc xem `.env.example`):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Chạy các lệnh kiểm thử và benchmark

```bash
# 1. Chạy toàn bộ 15 unit test của starter
python -m unittest discover -s tests -v

# 2. Chạy Benchmark 5 cấu hình Retrieval trên DEV
python src/benchmark_retrieval.py

# 3. Chạy đánh giá Agentic RAG Controller trên DEV (17 cases)
python src/benchmark_agent.py

# 4. Chạy đối soát độc lập trên tập HELD-OUT TEST (7 cases: Baseline vs Chosen)
python src/benchmark_test.py

# 5. Tổng hợp nhãn kiểm toán chuẩn bằng starter metric script
python starter/agent_metrics.py --cases data/cases.json --traces results/test_traces.json --reviews results/test_reviews.json --split test --output results/test_cli_summary.json
```

---

## 3. Cấu trúc Thư mục

```
K5-L3A-Day07-TranThanhTung-2A202603418-Retrieval-Context-Engineering/
├── data/
│   ├── raw/student_handbook_corpus.json   # 32 trang cẩm nang sinh viên
│   ├── chunks.json                        # 145 chunks cấu trúc
│   ├── questions.json                     # 20 câu hỏi có gold evidence
│   ├── cases.json                         # 24 test cases (dev + test)
│   ├── manifest.json                      # Metadata dữ liệu
│   └── cache/gemini_embeddings_cache.json # 300 vector embeddings đã cache
├── src/
│   ├── retriever.py                       # Tfidf, Dense, Weighted Hybrid RRF
│   ├── agent.py                           # GeminiController & run_agent loop
│   ├── benchmark_retrieval.py             # Script so sánh 5 cấu hình retrieval
│   ├── benchmark_agent.py                 # Script benchmark Agent trên DEV
│   └── benchmark_test.py                  # Script benchmark độc lập trên TEST
├── results/                               # Toàn bộ traces, reviews, summaries
├── report/
│   └── individual/2A202603418_TranThanhTung.md  # Báo cáo cá nhân hoàn chỉnh
├── starter/                               # Khung code và evaluation metric gốc
└── tests/                                 # 15 unit tests tự động
```

---

## 4. Tóm tắt Kết quả Thực nghiệm

- **Retrieval trên tập TEST:**
  - *Baseline (TF-IDF, budget 800)*: Hit@1 = 100%, Hit@3 = 100%, Context Evidence Rate = **60.0%** (2 câu bị rớt context vì tràn 800 ký tự).
  - *Cấu hình đã chọn (Weighted Hybrid, budget 1400)*: Hit@1 = **100%**, Hit@3 = **100%**, Context Evidence Rate = **100.0%**.
- **Agent trên tập TEST:**
  - Tỷ lệ gọi Retrieval Tool: **100% (7/7)**.
  - Từ chối đúng câu bẫy thiếu bằng chứng (*Abstention*): **100% (2/2)** (`u03`, `u04` từ chối dứt khoát, không ảo giác).
  - Trích dẫn hợp lệ (*Citation Validity*): **100% (3/3)** trên các câu trả lời.
  - Độ trễ phản hồi trung vị: **3.08 giây** / lượt.
  - Chi phí API: **$0.00**.
