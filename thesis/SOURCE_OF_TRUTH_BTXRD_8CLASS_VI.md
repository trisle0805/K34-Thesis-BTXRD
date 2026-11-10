# SOURCE OF TRUTH — BTXRD 8 lớp (v2)

## Phân loại u xương với tri thức y khoa phân tầng dạng proposition

> **Trạng thái:** v2 (đã khóa), cập nhật 11/11/2026. Thay thế toàn bộ v1 (kiến trúc gộp proposition B2/B3/A1 và decoder R1). Đã chỉnh 3 điểm sau review: F1 tắt PECL; H2-N dùng tên nút cho cả query và prototype PECL; φ lấy chiều từ encoder.
>
> **Mục đích:** tài liệu chuẩn duy nhất cho hướng nghiên cứu BTXRD: bài toán, tri thức, kiến trúc, thí nghiệm, đánh giá và phạm vi claim. Nếu tài liệu khác mâu thuẫn với file này thì file này thắng.
>
> **Mục tiêu tối thượng:** đưa tri thức y khoa vào mô hình deep learning. Tri thức được tổ chức theo tầng, từ đặc trưng chung tới đặc trưng riêng, và được dùng trực tiếp trong đường ra quyết định của mô hình.
>
> **Nguồn cảm hứng phương pháp:** RAD (Li et al., NeurIPS 2025): tri thức làm query của transformer decoder, mỗi query ra logit của chính nó, cộng thêm contrastive loss kéo đặc trưng ảnh về prototype tri thức. Đã đối chiếu với code chính thức `github.com/tdlhl/RAD`. Phương pháp của luận văn **không phải RAD**: tri thức có cấu trúc phân tầng, decoder phân tầng, loss phân tầng (xem §6).

---

# 1. Bài toán

## 1.1. Dataset

**BTXRD**: 3.746 ảnh raw. Loại hai nhóm composite `other bt` và `other mt`, còn **N = 3.586** ảnh. Ảnh có nhãn chồng lấn `IMG001276.jpeg` được gán `synovial_osteochondroma` ở bước preprocessing.

## 1.2. Các lớp

| Class | Số ảnh |
|---|---:|
| normal | 1879 |
| osteochondroma | 753 |
| multiple_osteochondromas | 263 |
| simple_bone_cyst | 206 |
| giant_cell_tumor | 93 |
| osteofibroma | 44 |
| synovial_osteochondroma | 51 |
| osteosarcoma | 297 |

## 1.3. Split

Split cố định, stratified 8 lớp: **train 2510 / val 538 / test 538**. Mọi experiment dùng cùng split.

**Quyết định kiến trúc và hyperparameter dựa trên validation.** Test chỉ dùng để báo cáo.

## 1.4. Metric

- **Chính:** Macro-F1 (8 lớp).
- **Phụ:** Accuracy, Balanced Accuracy, Macro Precision/Recall, per-class P/R/F1 (đọc kèm support), OvR AUROC, confusion matrix.
- **Phân tầng (mới):**
  - F1 normal ↔ u.
  - F1 lành ↔ ác, tính trên các ca u.
  - **Tỉ lệ lỗi nghiêm trọng:** dự đoán sai ở tầng normal/u hoặc tầng lành/ác.
  - Accuracy theo từng tầng.

---

# 2. Kết quả đã có (giai đoạn thăm dò, 1 seed, số test)

| Run | Mô tả | Acc | Bal. Acc | Macro-F1 | Macro AUROC |
|---|---|---:|---:|---:|---:|
| B0 | DenseNet-169, image-only | 74.54 | 60.35 | **61.78** | 90.78 |
| B1 | + metadata late fusion | 75.09 | 54.78 | 58.49 | 88.02 |
| B2 | + 65 proposition query, global pooling | 76.77 | 57.44 | 61.33 | 89.65 |
| B3 | B2 + contrastive trên z_kb | 76.77 | 58.66 | 62.64 | 91.31 |
| R1-D | class-conditioned decoder + residual logit | — | — | 62.48 (final), 64.04 (image head) | — |
| S0 | SKELEX image-only | 81.97 | 68.15 | 69.91 | 91.40 |

Ghi chú:
- Kết quả S0 cho thấy backbone là một nút thắt lớn. SKELEX giữ vai trò baseline mạnh và dùng cho ablation backbone (§8.3), không phải trục chính.
- Metadata (B1) không giúp, nên không dùng trong model chính.

---

# 3. Vì sao bỏ kiến trúc v1 (B2/B3/A1/R1)

Kiến trúc v1 dùng 65 proposition làm query cross-attention. Có hai biến thể:
- **B2/B3/A1:** gộp kết quả thành một vector z_kb, nối với z_img, rồi qua một linear.
- **R1:** prototype lớp làm query để attention lên các proposition thuộc K_c, sau đó cộng residual vào logit ảnh.

**Chẩn đoán trên checkpoint (val 538 ảnh):**

| Chỉ số | Kết quả | Ý nghĩa |
|---|---|---|
| Trọng số gộp của B2 (learned pooling) | entropy chuẩn hóa 1,0000; std giữa bệnh nhân 0,00017 | Gộp **đều tuyệt đối**. B2 thực chất bằng mean pooling, nên A1 mất ý nghĩa. |
| Trọng số class-attention của R1-D | entropy 1,0000; std giữa bệnh nhân 0,00000 | Decoder cũng sụp về trung bình cố định. |
| cos giữa 65 vector h_ij | 0,50 | Các proposition **không** bị trùng biểu diễn; vấn đề không nằm ở cross-attention. |
| cos giữa 65 bản đồ cross-attention | 0,87 | Các query nhìn vào vùng ảnh khá giống nhau. |
| F1 khi chỉ dùng logit knowledge (R1-D) | ~12% | Nhánh knowledge gần như không tự phân biệt được lớp. |
| R1-D image head → final | val 58,63 → 58,76; test 64,04 → 62,48 | Knowledge gần như không đổi quyết định. |

**Nguyên nhân gốc:**
- Ở v1, **không query tri thức nào có mục tiêu giám sát riêng**. 65 query bị gộp trước khi phân loại, nên trung bình đã đủ để giảm loss và mô hình không có lý do để chọn lọc.
- Điểm attention ban đầu chênh rất ít (query và các phép chiếu có thang đo nhỏ), nên mô hình kẹt ở trạng thái đều.
- Trên SkinCAP (repo `RAD-colab`, nhánh P1), hướng điểm theo từng proposition rồi gộp theo bệnh cũng không hơn RAD gốc. Điểm từng proposition khi không có nhãn ở mức proposition cũng không đúng ngữ nghĩa.

**So với RAD:** mỗi query của RAD gắn với một lớp và ra thẳng logit của lớp đó, nên loss buộc query tìm bằng chứng cho lớp của nó. v1 đã bỏ mất đúng mối nối này.

**Kết luận:**
1. Tri thức phải đi vào đường ra quyết định qua **query có mục tiêu giám sát riêng**.
2. Không dựa vào điểm hay attention của từng proposition riêng lẻ, vì không có nhãn ở mức proposition và không dùng nhãn yếu.

Code v1 (B2/B3/A1/R1) giữ trong repo để tái lập. Kết quả v1 được báo cáo như **phân tích thất bại có chẩn đoán**, không phải model chính.

---

# 4. Tri thức y khoa

## 4.1. KB proposition (không đổi)

- **BTXRD 8-class KB v1.0** (`02_kb/`, đã khóa): 68 proposition, có nguồn, mapping proposition → một hoặc nhiều lớp.
- Model dùng **P001–P065**. P066–P068 là proposition metadata tuổi/giới, bị loại.
- Text encoder: **MedCPT query encoder, đóng băng**. Embedding t_j ∈ ℝ⁷⁶⁸ được cache offline.
- Taxonomy limitation giữ như v1:
  - `osteofibroma` được biểu diễn bằng literature của osteofibrous dysplasia.
  - `synovial_osteochondroma` được biểu diễn bằng literature của synovial (osteo)chondromatosis.

## 4.2. Cây phân loại (mới)

Nhóm u theo S10 (WHO manual of diagnostic imaging, chương 9.6, đã có trong KB) kết hợp nhãn lành/ác của BTXRD. Nút chỉ có một con được gộp vào con của nó.

```text
gốc
├─ normal                                         [lá]
└─ u xương
   ├─ lành tính
   │   ├─ nhóm tạo sụn (S10 §9.6.2; §9.5)
   │   │   ├─ osteochondroma                      [lá]
   │   │   ├─ multiple_osteochondromas            [lá]
   │   │   └─ synovial_osteochondroma             [lá]
   │   ├─ osteofibroma   (nhóm xơ, S10 §9.6.3)    [lá, gộp nút một con]
   │   └─ nhóm chưa rõ nguồn gốc (S10 §9.6.5)
   │       ├─ simple_bone_cyst                    [lá]
   │       └─ giant_cell_tumor                    [lá]
   └─ osteosarcoma   (ác tính, tạo xương, S10 §9.6.6)  [lá, gộp nút một con]
```

- 5 nút quyết định: gốc, u xương, lành tính, nhóm tạo sụn, nhóm chưa rõ nguồn gốc.
- **12 nút con**: 4 nút trong và 8 lá.
- Cây lưu trong file cấu hình riêng (ví dụ `02_kb/taxonomy_8class.yaml`), mỗi nhóm kèm nguồn. Mỗi nút có thêm một tên ngắn, dùng cho run đối chứng H2-N.

## 4.3. Gán proposition vào nút (tất định, không gán tay)

Mỗi proposition được gán vào **nút thấp nhất bao trùm tất cả các lớp mà nó map tới** (lowest common ancestor). Việc gán do code tính từ mapping đã khóa, không có nhãn mới. Kết quả kiểm tra trên 65 proposition:

| Tầng | Nút | Số proposition |
|---|---|---:|
| Chung | u xương | 5 |
| | lành tính | 3 |
| | nhóm tạo sụn | 5 |
| | nhóm chưa rõ nguồn gốc | 3 |
| Riêng | normal | 12 |
| | osteosarcoma | 11 |
| | synovial_osteochondroma | 8 |
| | simple_bone_cyst | 6 |
| | multiple_osteochondromas | 4 |
| | osteofibroma | 4 |
| | giant_cell_tumor | 3 |
| | osteochondroma | 1 |
| | **Tổng** | **65** |

Định nghĩa:
- `own(n)`: proposition được gán cho chính nút n.
- `sub(n)`: proposition của n và mọi nút con cháu.
- Mọi nút trong cây hiện tại đều có `own(n)` khác rỗng. Quy tắc dự phòng khi rỗng là dùng `sub(n)`.

Proposition **không** bị diễn giải là nhãn hay xác suất xuất hiện. Chúng chỉ là đơn vị tri thức dùng để tạo query và prototype.

---

# 5. Kiến trúc

## 5.1. Image encoder (không đổi)

- DenseNet-169 ImageNet, input 224, ảnh xám lặp thành 3 kênh.
- F ∈ 7×7×1664, lấy sau ReLU cuối.
- Visual token: V = W_v·F_flat ∈ ℝ^(49×d).
- Global feature: z_img = GAP(F) ∈ ℝ¹⁶⁶⁴, chỉ dùng cho PECL.
- Không dùng positional embedding.
- Interface encoder dùng chung (EncoderOutput), để thay được SKELEX cho ablation backbone.

## 5.2. Knowledge query của nút

q̄_n = Normalize( mean_{j ∈ own(n)} Normalize(t_j) ), sau đó q_n = W_q · q̄_n ∈ ℝ^d. W_q trainable, MedCPT đóng băng.

- Flat decoder (F1): query lớp c dùng toàn bộ K_c, tức prototype g_c như v1.
- H2-N (đối chứng): q̄_n là embedding MedCPT của **tên nút**. Prototype PECL của H2-N cũng lấy từ tên nút (§6.2), để H2-N hoàn toàn không dùng nội dung proposition.

## 5.3. Knowledge decoder

Transformer decoder theo kiểu RAD:
- **Query:** các q_n. Flat dùng 8 query; phân tầng dùng 12 query.
- **Key/Value:** visual token V.
- **Mỗi layer, pre-norm:** self-attention giữa các query → cross-attention vào V → FFN, có residual + LayerNorm.
- **Mặc định:** d = 256, 4 layer, 4 head, FFN 1024, dropout 0,1. Khởi tạo Linear/MHA theo normal(0; 0,02) như RAD.
- **Đầu ra:** mỗi query qua **một Linear(d, 1) dùng chung** → 1 logit cho nút đó. Danh tính nút đến từ query, không đến từ head.
- **Không có head linear riêng cho ảnh và không có residual logit:** decoder thay hẳn classifier, giống RAD. B0 là baseline "classifier thường".

## 5.4. Dự đoán phân tầng

- Tại mỗi nút quyết định p: P(c | p) = softmax trên logit của các con của p.
- Xác suất lá: P(lá) = tích P(nút | cha) dọc đường đi từ gốc xuống lá.
- Dự đoán 8 lớp = argmax P(lá).
- Xác suất các tầng trên (u/normal, lành/ác, nhóm) cũng được xuất ra.
- Với decoder phẳng (F1): P(lá) = softmax trên 8 logit.

---

# 6. Loss

## 6.1. Loss phân loại

- **Phẳng:** weighted CE trên 8 lớp.
- **Phân tầng:** L_H = −w_y · log P(lá đúng) = −w_y · Σ_{p ∈ đường đi} log P(con đúng | p).
  - Tương đương CE có giám sát ở **mọi tầng** trên đường đi.
  - Nhãn mọi tầng suy tất định từ nhãn 8 lớp, **không có nhãn yếu**.
  - w_y = trọng số sqrt-inverse của lá đúng, giữ như protocol cũ.

## 6.2. PECL — Proposition-Enhanced Contrastive Loss

Đối chiếu với RAD: GECL nhánh ảnh của RAD kéo **đặc trưng ảnh pooled** về prototype cố định, là embedding đoạn mô tả X-quang của guideline. PECL giữ nguyên ý đó, chỉ đổi nguồn prototype:

- u = Normalize(φ(z_img)), với φ: D_global → 768. D_global lấy từ interface encoder (`global_dim`): DenseNet 1664, SKELEX `hidden_size`. Không hard-code.
- Prototype nút: P_n = Normalize( mean_{j ∈ sub(n)} Normalize(t_j) ). Cố định, không có gradient.
- Riêng H2-N: P_n = Normalize(embedding MedCPT của tên nút).
- **Phẳng:** softmax CE của cos(u, P_c)/τ trên 8 lá.
- **Phân tầng:** tại mỗi nút quyết định p trên đường đi, softmax CE trên cos(u, P_con)/τ giữa các con của p, rồi lấy tổng. Đây là dạng single-label tương ứng với SupCon multi-positive của RAD; trường hợp phẳng chỉ là cây một tầng.
- Mặc định: λ = 0,1 (giống β của RAD), τ = 0,07.
- Tổng loss: L = L_cls + λ · L_PECL.

---

# 7. Training protocol (không đổi)

- 224×224. RandomAffine(7, (0.03, 0.03), (0.95, 1.05)) chỉ áp khi train.
- Sqrt-inverse weighted loss. Không dùng sampler, không oversampling.
- Adam, lr 1e-4, wd 1e-4. StepLR(25, 0,1). 100 epoch. Batch 32. AMP. Seed 2026.
- Checkpoint chọn theo **val Macro-F1 8 lớp**.
- SKELEX (ablation backbone) dùng recipe riêng của S0, đã có trong package.

---

# 8. Thí nghiệm

## 8.1. Core runs (DenseNet-169)

| Run | Decoder | Query | PECL | Mục đích |
|---|---|---|---|---|
| **B0** | — (linear) | — | — | Baseline image-only (đã có) |
| **F1** | phẳng, 8 query | tri thức lớp (K_c) | **tắt** | Query tri thức có giám sát có giúp không |
| **H1** | phân tầng, 12 query | tri thức nút (own) | **tắt** | Decode phân tầng có giúp không |
| **H2** | phân tầng, 12 query | tri thức nút (own) | phân tầng, prototype proposition | **Model đề xuất đầy đủ** |
| **H2-N** | phân tầng, 12 query | **tên nút** | phân tầng, prototype **tên nút** | Đối chứng: nội dung proposition có giúp không |

## 8.2. Các phép so

Mỗi bước chỉ đổi đúng một thành phần:

- **F1 vs B0:** decoder với query tri thức có giám sát, so với classifier thường.
- **H1 vs F1:** cấu trúc phân tầng từ chung tới riêng.
- **H2 vs H1:** PECL phân tầng.
- **H2 vs H2-N:** nội dung proposition, so với chỉ dùng tên nút. Cả hai cùng cây phân loại, nên phép so này đo nội dung proposition chứ không đo cấu trúc cây. **Báo cáo trung thực kết quả, dù tốt hay xấu.**

Theo dõi riêng: lá osteochondroma chỉ có 1 proposition trong own(n). Nếu query lá này yếu thì ghi nhận là limitation, không sửa KB trước khi chạy.

## 8.3. Sau core runs

- **Ablation backbone:** H2 với SKELEX, so với S0. Chỉ đổi backbone.
- **Không làm nếu chưa có lý do thực nghiệm:**
  - query từng proposition hoặc điểm từng proposition;
  - nhãn yếu;
  - grid lớn cho λ/τ;
  - so sánh nhiều text encoder;
  - fine-tune MedCPT;
  - metadata;
  - multi-scale;
  - lesion supervision;
  - sửa KB.
- Seed bổ sung và CV: để giai đoạn chạy chính thức.

---

# 9. XAI

- **Grad-CAM:** B0 vs H2.
- **Lesion grounding** trên ảnh u có annotation BTXRD: Pointing Game và Energy Inside Lesion. Normal không tính.
- **Cross-attention** của query nút "u xương" và các nút lá: chỉ là visualization bổ sung, cho biết query đang tập trung vào đâu. Các query này có giám sát thật ở tầng của nó, nhưng vẫn **không claim** nhân quả.
- **Đường suy luận** từ gốc xuống lá kèm xác suất từng tầng: trình bày như output phân tầng của mô hình.

---

# 10. Phạm vi claim

**Được claim, nếu thực nghiệm hỗ trợ:**
- Tri thức X-quang được tổ chức thành proposition có nguồn, sắp theo cây từ chung tới riêng; việc gán vào cây là tất định từ mapping KB.
- Tri thức nút được dùng làm query có giám sát trong decoder phân tầng.
- Decode phân tầng hoặc PECL cải thiện Macro-F1 hoặc giảm lỗi nghiêm trọng so với baseline.
- Nội dung tri thức giúp hơn tên nút, **chỉ khi H2 > H2-N**.
- Mô hình tập trung vào vùng tổn thương tốt hơn, nếu metric grounding hỗ trợ.

**Không claim:**
- Proposition xuất hiện hay không, hoặc điểm của từng proposition.
- Mô hình là concept bottleneck.
- Attention là giải thích nhân quả.
- Mô hình suy luận giống bác sĩ.
- Giá trị lâm sàng ngoài phạm vi dataset.

---

# 11. Các quyết định đã khóa

| Thành phần | Quyết định |
|---|---|
| Dataset / task | BTXRD, 8 lớp, 3.586 ảnh, split 2510/538/538 |
| Metric chính | Macro-F1 8 lớp (chọn checkpoint trên val) |
| Backbone chính | DenseNet-169 ImageNet, 224 |
| KB | 8-class KB v1.0, P001–P065, MedCPT query encoder đóng băng |
| Cây phân loại | §4.2, nguồn S10 chương 9.6 + nhãn lành/ác của BTXRD |
| Gán proposition | Lowest common ancestor, tất định |
| Query nút | Trung bình embedding đã normalize của own(n), sau đó W_q |
| Decoder | Pre-norm transformer decoder: d 256, 4 layer, 4 head, FFN 1024 |
| Head | Linear(d, 1) dùng chung cho mọi query; không có head ảnh, không residual |
| Dự đoán | Softmax phân tầng, xác suất lá = tích dọc đường đi |
| Loss | L_cls (weighted, phẳng hoặc phân tầng) + 0,1 · PECL |
| PECL | φ(z_img): D_global → 768; prototype sub(n) cố định (H2-N: tên nút); τ 0,07; tắt ở F1 và H1 |
| Core runs | B0, F1, H1, H2, H2-N |
| Nhãn yếu / giám sát proposition | Không |
| Metadata trong model chính | Không |
| SKELEX | Baseline S0 và ablation backbone, sau core runs |

---

# 12. Câu chuyện paper

1. Phân loại 8 lớp u xương trên X-quang là bài toán khó: lớp mất cân bằng, hình thái chồng lấn.
2. Tri thức X-quang được tổ chức thành proposition có nguồn, sắp trên cây phân loại từ chung (u xương, lành tính, nhóm mô) tới riêng (từng loại u).
3. Tri thức của mỗi nút làm query cho transformer decoder trên visual token. Mỗi query ra logit của nút đó, được giám sát bằng nhãn tầng suy ra tất định từ nhãn lớp.
4. Dự đoán bằng softmax phân tầng. PECL phân tầng kéo đặc trưng ảnh về prototype tri thức của các nút trên đường đi.
5. Đánh giá: Macro-F1, lỗi nghiêm trọng, lesion grounding. Đối chứng với query tên nút. Ablation backbone với SKELEX.
6. Phân tích thất bại của kiến trúc gộp proposition (v1) có chẩn đoán định lượng, làm động cơ cho thiết kế mới.

---

# 13. Nguyên tắc implementation

- Giữ toàn bộ package hiện có: data, recipe, encoder interface, KB loader, MedCPT cache, trainer, predict, resume.
- **Thêm:**
  - file cây phân loại;
  - module gán proposition theo lowest common ancestor;
  - knowledge decoder (phẳng hoặc phân tầng dùng chung một code, khác nhau ở cây);
  - loss phân tầng;
  - PECL trên z_img;
  - cache MedCPT cho tên nút (dùng cho H2-N).
- Config-driven: F1, H1, H2, H2-N chỉ khác nhau ở config.
- Code v1 (B2/B3/A1/R1) giữ nguyên để tái lập, không phát triển thêm.
- Lưu logit từng nút và xác suất từng tầng khi predict. Ghi metric phân tầng trên val mỗi epoch.
- Notebook Colab chỉ là runner.
