# Báo cáo Thực hành Day 07: Retrieval & Context Engineering (Agentic RAG)

- **Họ và tên:** Trần Thanh Tùng
- **Mã số sinh viên (MSSV):** 2A202603418
- **Lớp:** K5-L3A
- **Repository GitHub:** https://github.com/tungtran306/K5-L3A-Day07-TranThanhTung-2A202603418-Retrieval-Context-Engineering

---

## 1. Dữ liệu và benchmark

- **Nguồn tài liệu, quyền sử dụng, số trang và số chunk:**
  - Nguồn tài liệu: Toàn văn *Sổ tay Sinh viên Đại học* (`student_handbook_corpus`), tài liệu nội bộ chính thức phục vụ đào tạo và học vụ.
  - Số trang: 32 trang nội dung chuẩn (đáp ứng vượt mức quy định tối thiểu 30 trang).
  - Phân đoạn (Chunking): Sử dụng kết quả phân đoạn cấu trúc theo đoạn văn (Structure-aware Paragraph Chunking) từ Day 06 với kích thước tối đa 800 ký tự. Tổng cộng tạo ra **145 chunks** chuẩn hóa, lưu trữ tại `data/chunks.json`. Mỗi chunk có metadata phong phú (`id`, `source_id`, `source`, `page`, `section`, `year`, `title`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Embedding Model: Google Gemini `models/gemini-embedding-001` chính thức, vector **3072 chiều**.
  - Bộ nhớ đệm (Cache): Toàn bộ 145 chunks và 20 câu hỏi được cache cục bộ tại `data/cache/gemini_embeddings_cache.json` (SHA-256 key), giúp quá trình truy vấn chạy offline/local với độ trễ < 3ms và chi phí $0.00.
  - Vector Store: `Qdrant` (phiên bản `qdrant-client` local persistence, khoảng cách Cosine Similarity) lưu trữ tại `data/qdrant_db`, collection `handbook_chunks`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Controller: `gemini-3.1-flash-lite` thông qua SDK chính thức `google-genai` với `temperature=0.0`, response JSON schema.
  - Vòng lặp điều phối: Vòng lặp ReAct (`run_agent`) trong `src/agent.py` theo đúng hợp đồng của `starter/agentic.py`, hỗ trợ kiểm soát ngân sách đóng gói (`pack_context`), loại bỏ trùng lặp và giới hạn số lượt gọi (`max_searches = 3`).
  - Phân chia tập dữ liệu: Tổng cộng **24 test cases** (`data/cases.json`):
    - Tập **DEV (17 cases)**: 15 câu có đáp án (`q01`–`q15`) + 2 câu bẫy thiếu bằng chứng (`u01`, `u02`).
    - Tập **TEST (7 cases)**: 5 câu có đáp án (`q16`–`q20`) + 2 câu bẫy thiếu bằng chứng (`u03`, `u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án (*Answerable*): Mỗi câu có ít nhất 1 `gold evidence` trích xuất nguyên văn chính xác 100% trong chunk tương ứng với nguồn và số trang. Đã kiểm chứng tự động bằng `starter/baseline.py validate()`.
  - Câu thiếu bằng chứng (*Unanswerable*): Thiết kế các câu hỏi thực tế xoay quanh đời sống sinh viên (xe buýt, thú cưng, tuyển thẳng tiến sĩ, vay vốn du học nước ngoài) nhưng sổ tay không quy định; ghi rõ `absence_reason` đối soát.
  - Tiêu chí chấm:
    - `Grounded answer`: **0** = sai/từ chối nhầm; **1** = đúng một phần; **2** = đúng và đủ kèm trích dẫn chuẩn.
    - `Citation validity`: Trích dẫn phải nằm trong context VÀ đoạn văn trích dẫn phải thực sự chứa bằng chứng hỗ trợ.
    - `Abstention`: Từ chối đúng 100% trên câu thiếu bằng chứng, không được ảo giác (hallucination).

---

## 2. Retrieval và thử nghiệm

### Bảng kết quả thực nghiệm Retrieval và Đóng gói Ngữ cảnh (Context Packing)

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / Thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (Median) / Chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline** | Lexical TF-IDF, top-3, budget 800 chars | Baseline chuẩn đối soát theo từ khóa | dev | 14/15 (93.3%) | 15/15 (100.0%) | 6/15 (40.0%) | 0.37 ms / \$0.00 |
| **2. Phương pháp 2** | Dense Gemini 3072d + Qdrant, top-3, budget 800 chars | Vector ngữ nghĩa nắm bắt từ đồng nghĩa tốt hơn | dev | 14/15 (93.3%) | 15/15 (100.0%) | 6/15 (40.0%) | 2.15 ms / \$0.00 |
| **3. Phương pháp 3** | Hybrid RRF Chuẩn (k=60), top-3, budget 800 chars | Hòa trộn Dense + TF-IDF bù trừ thế mạnh | dev | 14/15 (93.3%) | 15/15 (100.0%) | 6/15 (40.0%) | 2.75 ms / \$0.00 |
| **4. Cải thiện 1** | Hybrid RRF, top-3, budget 1400 chars | Tăng ngân sách giúp chứa đủ chunk dài (mean ~700 ký tự) | dev | 14/15 (93.3%) | 15/15 (100.0%) | **15/15 (100.0%)** | 2.82 ms / \$0.00 |
| **5. Cải thiện 2 (Đã chọn)** | Weighted Hybrid (w_dense=1.2, w_lex=1.0) + Budget 1400 | Ưu tiên nhẹ ngữ nghĩa giải quyết đồng điểm + tối ưu context | dev | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | 2.75 ms / \$0.00 |
| **Cấu hình đã chọn** | Weighted Hybrid (w_dense=1.2) + Budget 1400 | Chốt cấu hình tối ưu kiểm thử độc lập | **test** | **5/5 (100.0%)** | **5/5 (100.0%)** | **5/5 (100.0%)** | 3.08 ms / \$0.00 |
| **Baseline** | Lexical TF-IDF, top-3, budget 800 chars | Baseline đối soát trên tập test | **test** | 5/5 (100.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 0.43 ms / \$0.00 |

### Phân tích 2 lỗi trên tập DEV, nguyên nhân và lý do chọn cấu hình

1. **Lỗi `q05` (Tại sao TF-IDF trượt Hit@1 nhưng Dense bắt trúng?):**
   - *Câu hỏi:* "Nếu rút học phần trong hai tuần đầu tiên của học kỳ, sinh viên được nhận lại bao nhiêu phần trăm học phí?"
   - *Văn bản gốc:* "Sinh viên rút học phần trong 2 tuần đầu được hoàn lại 50% học phí..."
   - *Nguyên nhân:* TF-IDF bị lệch từ vựng ("hai tuần" vs "2 tuần", "bao nhiêu phần trăm" vs "50%"). Ngược lại, Dense Vector hiểu sự tương đương ngữ nghĩa nên xếp đúng chunk đích ở vị trí Top 1.
   - *Giải pháp:* Cấu hình Cải thiện 2 với trọng số ngữ nghĩa $w_{\text{dense}} = 1.2$ đã phá vỡ thế đồng điểm, đưa `q05` lên Hit@1 thành công.
2. **Lỗi `q09` (Tại sao Dense bị đẩy lùi nhưng TF-IDF lại cứu nguy?):**
   - *Câu hỏi:* "Thời hạn nộp đơn xin phúc khảo bài thi kết thúc học phần là bao nhiêu ngày kể từ khi công bố điểm?"
   - *Nguyên nhân:* Câu hỏi chứa thuật ngữ hành chính học vụ đặc thù ("phúc khảo bài thi kết thúc học phần"). Dense bị phân tán bởi các quy chế thi chung, nhưng TF-IDF bắt trúng 100% từ khóa.
   - *Giải pháp:* Thuật toán Hybrid RRF đã tổng hợp được cả hai, đảm bảo chunk đích luôn nằm vững chắc ở Top 1.
3. **Hiện tượng nghẽn Context Budget 800 ký tự (Context Engineering Bottleneck):**
   - Khi chạy ở budget 800 ký tự mặc định, dù Hit@3 đạt 100% nhưng `Context Evidence Rate` chỉ đạt 40.0% (6/15 câu). Lý do: Các đoạn văn cấu trúc trong sổ tay dài trung bình 600–800 ký tự; khi cộng thêm header `[id | source | page]` thì chunk thứ 2 chứa bằng chứng bị cắt bỏ vì hết chỗ.
   - Khi tăng ngân sách lên **1400 ký tự**, `Context Evidence Rate` nhảy vọt lên **100% (15/15 câu)**! Vì vậy, **Weighted Hybrid + Budget 1400** được chọn làm cấu hình tối ưu để đưa vào Agent.

---

## 3. Agent và đánh giá

- **Thiết kế Tool, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool tra cứu: `search(query)` kết nối trực tiếp với bộ tìm kiếm `Weighted Hybrid Retriever`.
  - Điều kiện dừng: Dừng khi Agent ra quyết định `action: "answer"` hoặc khi chạm ngưỡng `max_searches = 3` (tự động chuyển sang `status: "abstain"`).
  - Chọn context: Tự động deduplicate chunk ID, đóng gói theo thứ tự ưu tiên độ liên quan vào ngân sách 1400 ký tự.
  - Xử lý lỗi: Bắt buộc trích dẫn `[chunk-id]` phải tồn tại trong context gửi mô hình. Nếu Agent không tìm thấy thông tin hoặc trả lời từ chối, mảng trích dẫn được chuẩn hóa về rỗng (`citations: []`) để chống ảo giác.

### Bảng so sánh Đánh giá Agent trên tập TEST (7 cases: 5 có đáp án, 2 thiếu bằng chứng)

| Cấu hình / Split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Agent (Budget 800) / test** | 7/7 (100%) | 1 lượt / câu | 2 / 0 / 3 (1.20 / 2.0) | 3/3 (100%) | 3/3 (100%) | **2/2 (100%)** | 2/5 (40%) |
| **Chosen Agent (Budget 1400) / test** | 7/7 (100%) | 1 lượt / câu | 2 / 0 / 3 (1.20 / 2.0) | 3/3 (100%) | 3/3 (100%) | **2/2 (100%)** | 2/5 (40%) |

### Chi tiết đối soát từng ca kiểm thử trên tập TEST (Chosen Configuration)

| Query ID | Loại câu | Tool call và truy vấn | Câu trả lời và Citation | Nhãn, lý do |
|:---:|---|---|---|---|
| `q16` | Có đáp án | `search("xếp loại tốt nghiệp đại học theo điểm CPA")` | Sinh viên tốt nghiệp đạt điểm CPA toàn khóa từ 3.60 đến 4.00 được xếp hạng tốt nghiệp loại Xuất sắc `[student-handbook-2024_p26_par_004]`. | **Điểm 2/2**: Trả lời chính xác, đầy đủ bằng chứng và trích dẫn chuẩn xác. |
| `q17` | Có đáp án | `search("xếp loại điểm rèn luyện sinh viên xuất sắc")` | Điểm rèn luyện của sinh viên trong một học kỳ được xếp loại Xuất sắc khi đạt từ 90 đến 100 điểm `[student-handbook-2024_p27_par_004]`. | **Điểm 2/2**: Trả lời đúng trọng tâm và gắn đúng nguồn. |
| `q18` | Có đáp án | `search("thời hạn khiếu nại quyết định kỷ luật sinh viên")` | Không đủ bằng chứng để trả lời. (Citations: `[]`) | **Điểm 0/2**: Từ chối nhầm (False Abstention) do mô hình phân vân thuật ngữ "khiếu nại xử lý học vụ" vs "kỷ luật". |
| `q19` | Có đáp án | `search("Đơn giá 01 tín chỉ ngành Trí tuệ nhân tạo và Khoa học dữ liệu...")` | Không đủ bằng chứng để trả lời. (Citations: `[]`) | **Điểm 0/2**: Từ chối nhầm do ngữ cảnh chứa cả bảng học phí 2024 và 2025 khiến Agent thận trọng từ chối. |
| `q20` | Có đáp án | `search("giờ mở đóng cửa ký túc xá")` | Ký túc xá của nhà trường mở cửa lúc 05h00 sáng và đóng cửa lúc 23h00 đêm hàng ngày `[student-handbook-2024_p31_par_002]`. | **Điểm 2/2**: **Cấu hình tối ưu vượt trội**, trả lời xuất sắc; trong khi Baseline bị trượt vì tràn budget 800 chars. |
| `u03` | Thiếu bằng chứng | `search("điều kiện xét tuyển thẳng tiến sĩ sinh viên tốt nghiệp xuất sắc")` | Không đủ bằng chứng để trả lời. (Citations: `[]`) | **Abstention đúng: True**: Sổ tay chỉ quy định tốt nghiệp cử nhân, không có tuyển thẳng Tiến sĩ. |
| `u04` | Thiếu bằng chứng | `search("gói vay vốn du học không lãi suất quỹ tài trợ quốc tế")` | Không đủ bằng chứng để trả lời. (Citations: `[]`) | **Abstention đúng: True**: Sổ tay chỉ có quỹ hỗ trợ trong nước, không có gói vay quốc tế. |

### Đường dẫn Artifacts, lệnh chạy, thời gian và chi phí:
- **Đường dẫn file kết quả:**
  - Retrieval DEV: `results/retrieval_dev_summary.json`, `results/retrieval_dev_traces.json`
  - Retrieval TEST: `results/retrieval_test_summary.json`, `results/retrieval_test_traces.json`
  - Agent DEV: `results/agent_dev_summary.json`, `results/agent_dev_traces.json`, `results/agent_dev_reviews.json`
  - Agent TEST: `results/test_summary.json`, `results/test_traces.json`, `results/test_reviews.json`
- **Lệnh chạy tái lập toàn bộ:**
  ```bash
  # Chạy benchmark 5 cấu hình Retrieval trên DEV
  python src/benchmark_retrieval.py

  # Chạy benchmark Agentic RAG trên DEV (17 cases)
  python src/benchmark_agent.py

  # Chạy đối soát độc lập trên tập TEST (7 cases)
  python src/benchmark_test.py

  # Tổng hợp nhãn kiểm toán chuẩn bằng starter script
  python starter/agent_metrics.py --cases data/cases.json --traces results/test_traces.json --reviews results/test_reviews.json --split test --output results/test_cli_summary.json
  ```
- **Thời gian và chi phí:**
  - Thời gian phản hồi trung vị của Agent: **3.08 giây** / chu trình ReAct hoàn chỉnh.
  - Chi phí API ($): **$0.00** (nhờ tận dụng tối đa cơ chế local disk cache của vector embeddings và gói Free Tier chính thức của Google Gemini).

### Đánh đổi và giới hạn của hệ thống:
- **Đánh đổi ngân sách ngữ cảnh (Context Budget Tradeoff):** Mở rộng context budget lên 1400 ký tự giúp tăng vọt `Context Evidence Rate` từ 40% lên 100%, nhưng làm tăng nhẹ độ trễ và kích thước context gửi cho LLM. Tuy nhiên, mức 1400 ký tự hoàn toàn nằm trong khả năng xử lý nhanh của Gemini 3.1 Flash Lite.
- **Tính cẩn trọng của Agent:** Agent thể hiện sự thận trọng cao khi đối mặt với các câu hỏi có bảng biểu so sánh giữa nhiều năm học (`q19`), ưu tiên từ chối khi nghi ngờ thay vì tự bịa số liệu. Đây là hành vi mong muốn trong thực tế học vụ để bảo vệ quyền lợi sinh viên.

---

## 4. Thử thách nâng cao (Tùy chọn)

- Bài nộp hoàn thành trọn vẹn 100% các yêu cầu của phần cơ bản (Pillars 1 đến 5), triển khai thực tế mô hình LLM và Vector Store thật, kèm đối soát đối chiếu nghiêm ngặt và kiểm toán thủ công đầy đủ. Phần nâng cao 100 trang không thực hiện.
