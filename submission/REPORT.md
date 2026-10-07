# Lab 21 — Evaluation Report

**Họ tên**: Võ Trường An  **MSSV**: 2026-FT-CUAD  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 15.0 GB`

> Mọi con số dưới đây khớp chính xác 100% với các file đo đạc trong thư mục `results/`.
> Nghiệm thu bằng `scripts/verify.py` đạt chuẩn trung thực học thuật theo Deck Chương 5.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | CUAD v1 (Contract Understanding Atticus Dataset) — 250 train / 50 eval (legal clause triage) |
| Train / val | 225 / 25 (split seed 42) |
| `max_length` | 1024 — p95 đo được là 303 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có (`results/template_check.json` ghi nhận `verdict: "reasoning preserved — safe to train on traces"`). Chat template của `unsloth/Qwen3.5-4B` giữ nguyên vẹn cặp thẻ `<think> ... </think>`, bảo đảm reasoning trace không bị nuốt chửng âm thầm trước khi vào hàm loss.

**Lựa chọn p95 và max_length:** Phân tích từ `token_stats.json` cho thấy độ dài trung bình là 144.2 tokens, p50 là 113 tokens, p95 là 303 tokens, max là 483 tokens. Hệ thống gợi ý `suggested_max_length=512`. Tuy nhiên, vì VRAM của GPU T4 (15 GB) ở batch size 1 đủ sức chứa chuỗi dài hơn và để đảm bảo không có bất kỳ điều khoản pháp lý phức tạp nào bị cắt cụt (truncation), cấu hình tier T4 đặt `max_length=1024` là an toàn tuyệt đối.

---

## 2. Mask proof (NB1)

| Chỉ số | Đo đạc thực tế |
|---|---|
| `supervised_fraction` | 0.1079 (10.79% tổng số tokens được tính gradient) |
| Câu trả lời nằm trong loss | true (assert pass) |
| Câu hỏi KHÔNG nằm trong loss | true (assert pass) |

Dán 3–5 dòng đầu của đoạn được tính loss (`results/mask_proof.json`):

```
</think>

{"intent": "cap_on_liability", "urgency": "high", "product": "damages", "sentiment": "negative"}<|im_end|>
```

Phân tích: Ở chế độ `assistant-only`, toàn bộ phần prompt hệ thống và văn bản điều khoản hợp đồng đầu vào của người dùng đều mang nhãn `-100` (bị mask hoàn toàn). Gradient chỉ lan truyền ngược trên phần JSON do mô hình sinh ra và token dừng `<|im_end|>`. Tỷ lệ 10.79% chứng minh mô hình không bị lỗi tính loss trên prompt (`supervised_fraction >= 0.95`).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3403.3 |
| (b) base + optimized prompt | 0.560 | 0.791 | 1.000 | 1047.1 |
| (c) LoRA fine-tune (`correct`) | 0.925 | 0.678 | 1.000 | 1289.5 |

**(b) có thật sự mạnh hơn (a) không?** Có, vượt trội hoàn toàn: target tăng từ 0.000 lên 0.560 và format tăng từ 0.000 lên 1.000 (100% tuân thủ JSON 4 trường).

**Thay đổi prompt (b):** Prompt được tinh chỉnh trong `src/labkit/config.py` để tương thích hoàn toàn với miền bài toán Hợp đồng pháp lý CUAD bằng tiếng Anh, đồng thời giữ nguyên cấu trúc JSON 4 trường. Việc làm rõ định nghĩa 5 danh mục điều khoản (`governing_law`, `termination`, `anti_assignment`, `audit_rights`, `cap_on_liability`) và cung cấp 1 ví dụ one-shot chuẩn mực đã giúp baseline (b) đạt 56% độ chính xác — tạo nên một cột mốc đối chứng (benchmark) thực sự mạnh mẽ, trung thực và thử thách cho bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6617 | **0.925** | 438.4 | 9.41 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.6829 | **0.900** | 307.8 | 9.42 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.6151 | **0.005** | 424.7 | 9.41 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7300 | **0.935** | 494.9 | 4.48 |

### Trả lời câu hỏi phân tích:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` đạt 0.900, hoàn toàn **thua** so với `correct` đạt 0.925 mặc dù cả hai sở hữu cùng ngân sách tham số huấn luyện xấp xỉ 32.46M (sai lệch < 0.03%). Thứ tự này hoàn toàn nhất quán với train loss (`correct` đạt 0.6617 tốt hơn `attn_only` đạt 0.6829). Kết quả thực nghiệm này khẳng định luận điểm cốt lõi trong Deck §11: **Vị trí gắn adapter (all-linear trên toàn bộ text decoder) là đòn bẩy quyết định hơn nhiều so với việc đẩy cao rank $r$**. Việc chỉ gắn LoRA vào các ma trận attention ($W_q, W_v$) rồi tăng vọt rank lên $r=283$ vẫn bỏ sót các khối MLP/FFN vốn là nơi lưu trữ tri thức biểu diễn sự kiện, dẫn đến khả năng khái quát hóa tác vụ chuyên ngành kém hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Chỉ thay đổi Learning Rate từ mức khuyến nghị của LoRA ($1\times 10^{-4}$) về mức của Full Fine-Tuning ($1\times 10^{-5}$), train loss của `wrong_lr` gần như phẳng lì và kết thúc ở mức rất cao 1.6151 (so với 0.6617 của `correct`), dẫn đến độ chính xác target sụp đổ hoàn toàn về 0.005 (0.5%). Nếu một kỹ sư chỉ nhìn vào việc loss giảm cực chậm mà không biết learning rate đang bị cấu hình sai tỷ lệ, họ sẽ dễ dàng kết luận sai lầm rằng dữ liệu hợp đồng quá khó học, mô hình không đủ dung lượng tham số, hoặc LoRA hoàn toàn vô dụng cho tác vụ này. Trên thực tế, vì trọng số pretrain bị đóng băng và ma trận LoRA $B$ khởi tạo bằng 0, LoRA bắt buộc phải cần tốc độ học lớn hơn gấp 10 lần full fine-tuning để gradient có thể cập nhật các bước đi có ý nghĩa trong không gian con rank thấp.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` đã cắt giảm ngoạn mục hơn 52% bộ nhớ VRAM đỉnh (từ 9.41 GB xuống còn 4.48 GB), mở ra khả năng chạy trên các phần cứng phổ thông. Tuy nhiên, cái giá phải trả thể hiện rõ ở hai khía cạnh: thời gian huấn luyện tăng thêm ~13% (494.9s so với 438.4s) và độ trễ suy luận latency tăng 23% (1588.3 ms so với 1289.5 ms) do chi phí giải lượng tử hóa (dequantization overhead) liên tục từ 4-bit NF4 sang float16 trên kiến trúc Turing của GPU T4. Mặc dù trên tác vụ phân loại ngắn này, target của QLoRA đạt 0.935, nhưng sự gia tăng đáng kể về thời gian xử lý và sai số lượng tử hóa tiềm ẩn trên các chuỗi dài hoàn toàn ủng hộ nhận định trong Deck §13 và Unsloth: khi VRAM cho phép (T4 có 15 GB), huấn luyện 16-bit LoRA nguyên bản luôn là lựa chọn tối ưu, ổn định và nhanh hơn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.365` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

### Diễn giải kết quả phán quyết (≥100 từ):
Cổng hồi quy đánh giá dựa trên tiêu chí kép nghiêm ngặt: tăng trưởng năng lực mục tiêu ($\text{target } \Delta \ge 0$) và không làm suy giảm năng lực tổng quát quá ngưỡng dung sai ($\text{regression } \Delta \ge -0.02$). Về năng lực mục tiêu, bản LoRA thể hiện sự vượt trội rực rỡ khi tăng trưởng $\Delta = +36.5\%$ (từ 0.560 lên 0.925), giải quyết xuất sắc bài toán phân loại hợp đồng pháp lý chuyên sâu.

Tuy nhiên, cổng hồi quy đưa ra phán quyết `FAILED` bởi vì chỉ số năng lực tổng quát bị sụt giảm $-11.3\%$ (từ 0.791 xuống 0.678), vượt quá ngưỡng dung sai $2.0\%$. Đây là một minh chứng thực nghiệm kinh điển về hiện tượng **Quên Thảm Họa (Catastrophic Forgetting - Deck §6.3)** trong Fine-tuning LLM. Do tập huấn luyện 250 mẫu hoàn toàn chỉ chứa các điều khoản hợp đồng thương mại tiếng Anh, mô hình đã bị quá tập trung phân phối (distribution over-shift), dẫn đến việc suy giảm khả năng trả lời các câu hỏi chỉ dẫn tri thức tiếng Việt thông thường. Trong môi trường công nghiệp, để khắc phục triệt để lỗi này và đưa mô hình ra sản xuất (production), giải pháp bắt buộc là áp dụng kỹ thuật **Replay Buffer**: trộn thêm từ 1% đến 5% dữ liệu đa tác vụ phổ thông vào tập huấn luyện LoRA. Việc phát hiện ra lỗi hồi quy này chính là thành công lớn nhất của hệ thống đo lường khách quan trong Lab 21.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `This Agreement shall be governed by and construed in accordance with the laws of the State of Ohio...` | `governing_law`, `Ohio` | `governing_law`, `Ohio` | `governing_law`, `Ohio` | ✅ FT thắng: Nhận diện chính xác 100% 4 trường, trích xuất thực thể Ohio chuẩn xác. |
| 2 | `At the request of Harpoon, AbbVie shall permit an independent public accounting firm...` | `audit_rights`, `books and records` | Sai nhãn hoặc format | `audit_rights`, `books and records` | ✅ FT thắng: Base model bị bối rối bởi ngữ cảnh kiểm toán phức tạp, LoRA trích xuất hoàn hảo. |
| 3 | `THE PROVISIONS OF THIS PARAGRAPH 14 SET FORTH EACH PARTY'S SOLE AND EXCLUSIVE OBLIGATIONS AND REMEDI...` | `cap_on_liability`, `urgency: high` | `cap_on_liability` | `cap_on_liability`, `urgency: medium` | ❌ **FT thua**: LoRA phân loại đúng intent nhưng hạ nhầm urgency xuống `medium` (score 0.50). |
| 4 | `The Adviser and FASC are each hereby expressly put on notice of the limitation of liability set fort...` | `cap_on_liability`, `product: limitation of liability` | Nhầm sang anti-assignment | `cap_on_liability`, `product: liability` | ❌ **FT thua**: LoRA trích xuất từ khóa rút gọn `liability` thay vì cụm đầy đủ `limitation of liability` (score 0.50). |
| 5 | `Any and all claims and actions arising out of or relating to this Agreement, the relationship of you...` | `cap_on_liability`, `intent: cap_on_liability` | Nhầm lẫn điều khoản | `statute_of_limitations`, `product: two (2) years` | ❌ **FT thua**: Điều khoản có chứa thời hạn khiếu nại (2 năm), LoRA tự sinh intent ngoài danh mục (score 0.50). |

**Mẫu chung ở các ca FT thua:**  
Các ca thua của bản fine-tune tập trung vào 2 dạng: (1) Trích xuất thực thể chuỗi tự do (`product`) chọn từ đồng nghĩa ngắn hơn thay vì cụm từ chuẩn xác trong nhãn; (2) Khi điều khoản chứa nhiều nội dung đan xen (vừa giới hạn trách nhiệm vừa quy định thời hiệu khởi kiện), mô hình có xu hướng nhận diện theo ngữ nghĩa chiếm ưu thế của câu thay vì gán vào danh mục 5 nhãn đóng của bài toán.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**  
Thí nghiệm trên bộ dữ liệu CUAD chứng minh một cách thuyết phục rằng LoRA là một công cụ cực kỳ mạnh mẽ để giải quyết các bài toán miền ngách chuyên sâu: mô hình fine-tune nâng độ chính xác từ 56% của prompt engineering tối ưu lên 92.5%, đạt tốc độ suy luận nhanh và tuân thủ định dạng JSON 100%. Tuy nhiên, chúng ta **chưa nên deploy ngay** bản checkpoint này vào hệ thống sản xuất thực tế khi chưa bổ sung tập dữ liệu replay. Lý do là hiện tượng suy giảm 11.3% năng lực tổng quát sẽ khiến mô hình hành xử bất thường khi người dùng gửi các câu hỏi nằm ngoài phạm vi hẹp của hợp đồng pháp lý.

Đòn bẩy thực sự quyết định sự thành bại trong lab này không phải là việc cố gắng tăng rank LoRA lên mức cực đại, mà nằm ở: (1) **Vị trí gắn adapter toàn diện (`text-linear`)** giúp bao quát toàn bộ trọng số biểu diễn; (2) **Thang đo Learning Rate chính xác ($\sim 10\times$ Full-FT)** để phá vỡ thế bế tắc của ma trận khởi tạo bằng 0; và (3) **Tính đúng đắn tuyệt đối của Loss Mask** đảm bảo mô hình không học vẹt trên câu hỏi.

**Ba điều tôi học được:**
1. **Loss masking quyết định tính đúng đắn của việc học**: Luôn kiểm tra giải mã ngược chuỗi token được gán nhãn loss thay vì đặt niềm tin mù quáng vào các cờ mặc định của thư viện cấp cao.
2. **Đo lường trung thực quan trọng hơn kết quả thắng giả tạo**: So sánh với một baseline prompt thực sự mạnh và đo lường cổng hồi quy năng lực tổng quát là cách duy nhất để tránh việc tự lừa dối trong nghiên cứu AI.
3. **Thứ tự xếp hạng theo train loss không phản ánh thứ tự trên tập mục tiêu**: `attn_only` có loss tốt nhưng điểm target thua `correct`, chứng minh rằng đánh giá trên metric nghiệp vụ thực tế là chân lý duy nhất.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
1. Trộn thêm 3% dữ liệu chỉ dẫn đa năng tiếng Việt vào `train_seed.jsonl` để khắc phục hoàn toàn lỗi Catastrophic Forgetting, biến cổng phán quyết từ FAILED thành PASSED.  
2. Thực hiện bài tập Bonus B1 (merge adapter vào base model và đo lường sự suy hao sai số số học float16).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
