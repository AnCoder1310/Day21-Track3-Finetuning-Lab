# Custom Dataset Declaration: Contract Understanding Atticus Dataset (CUAD)

## 1. Nguồn dữ liệu & Nguồn gốc (Source & Provenance)
- **Tên dataset**: Contract Understanding Atticus Dataset (CUAD) v1.
- **Nguồn**: Atticus Project AI (arXiv:2103.06268, GitHub: TheAtticusProject/cuad).
- **Bộ file sử dụng**: `master_clauses.csv` chứa 510 hợp đồng thương mại thực tế và các điều khoản pháp lý được luật sư gắn nhãn thủ công qua 41 danh mục.
- **Tác vụ mục tiêu**: Phân loại và trích xuất thông tin điều khoản hợp đồng thương mại (Legal Contract Clause Triage) thành JSON 4 trường (`intent`, `urgency`, `product`, `sentiment`).

## 2. Kích thước & Cấu trúc (Size & Scope)
Chúng tôi trích xuất và chọn lọc 5 loại điều khoản quan trọng nhất trong rà soát hợp đồng thương mại:
1. `governing_law`: Luật áp dụng & Tòa án có thẩm quyền.
2. `termination`: Quyền đơn phương chấm dứt hợp đồng vì thuận tiện (Termination for Convenience).
3. `anti_assignment`: Ràng buộc cấm chuyển nhượng quyền và nghĩa vụ hợp đồng mà không có chấp thuận.
4. `audit_rights`: Quyền kiểm tra sổ sách kế toán và kiểm toán tuân thủ.
5. `cap_on_liability`: Điều khoản giới hạn trần trách nhiệm bồi thường thiệt hại.

- **Tập huấn luyện (`data/train_seed.jsonl`)**: 250 mẫu (cân bằng đúng 50 mẫu cho mỗi loại điều khoản).
- **Tập đánh giá mục tiêu (`data/eval_target.jsonl`)**: 50 mẫu (cân bằng đúng 10 mẫu cho mỗi loại điều khoản).
- **Tập kiểm tra suy giảm năng lực (`data/eval_regression.jsonl`)**: Giữ nguyên 15 mẫu kiến thức tổng quát của lab để kiểm tra hiện tượng quên kiến thức cũ (*catastrophic forgetting*).

Mỗi mẫu có độ dài từ 15 đến 250 từ (trung bình ~60 từ/mẫu), hoàn toàn nằm trong ngưỡng tính toán an toàn của `max_length = 1024` trên GPU T4.

## 3. Quy trình khử nhiễm dữ liệu (Decontamination Methodology)
Để loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm tra:
1. **Phân vùng ở cấp độ tài liệu hợp đồng (Contract Document-Level Partition)**:
   - Dữ liệu 510 hợp đồng được phân nhóm theo mã định danh tài liệu (`Document Name`).
   - 210 tài liệu hợp đồng được đưa vào nhóm huấn luyện (Train candidate pool).
   - 64 tài liệu hợp đồng hoàn toàn độc lập được đưa vào nhóm đánh giá (Eval candidate pool).
2. **Kiểm tra giao thoa (Overlap Assertion)**:
   - Số lượng tài liệu trùng lặp giữa train và eval: **0 tài liệu (0%)**.
   - Số lượng chuỗi văn bản điều khoản trùng lặp: **0 mẫu (0%)**.
   - Không có bất kỳ đoạn văn nào trong `eval_target.jsonl` từng xuất hiện trong `train_seed.jsonl`.

## 4. Độ lệch phân phối so với Base Model (Distribution Shift - Deck §3.3)
- Các mô hình nền tảng (Base models như Qwen3.5) đã được tiền huấn luyện trên lượng lớn văn bản web phổ thông (Wikipedia, tin tức, mạng xã hội), nơi các bài toán phân loại thông thường có thể dễ dàng giải quyết bằng kỹ thuật Prompt Engineering.
- Văn bản hợp đồng pháp lý thương mại trong CUAD có độ phức tạp cú pháp cao (*legal syntax*), mật độ thuật ngữ chuyên ngành dày đặc và cấu trúc điều kiện ràng buộc ngặt nghèo.
- Sự kết hợp giữa văn bản pháp lý tiếng Anh và yêu cầu định dạng JSON có cấu trúc nghiêm ngặt tạo nên một phân phối hoàn toàn mới (OOD) so với dữ liệu thông thường, tạo điều kiện lý tưởng để chứng minh giá trị thực tế của việc fine-tune bằng LoRA so với base model chỉ dùng prompt.
