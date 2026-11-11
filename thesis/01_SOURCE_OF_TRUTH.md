# SOURCE OF TRUTH — Luận văn K34, BTXRD 8 lớp

## Tích hợp tri thức y khoa dạng mệnh đề vào mô hình phân loại u xương trên X-quang

> **Phiên bản:** 11/11/2026, bản chốt.
> **Hiệu lực:** đây là tài liệu quyết định cao nhất. Mọi file SOT, coding guide và handoff khác trong project không còn hiệu lực.
> **Quy ước số liệu:** số trong mục 7 là **khảo sát sơ bộ** (1 split, 1 seed). Số chính thức lấy từ protocol CV ở mục 3.4.
> **Ký hiệu:** `[đã kiểm chứng]` là có số liệu hoặc code xác nhận; `[giả thuyết]` là lập luận chưa được thực nghiệm kiểm tra.

---

## Mục lục

1. Bài toán và động lực
2. Liên hệ với hai bài nền: RAD và BoneVisionNet
3. Dữ liệu và protocol thực nghiệm
4. Knowledge base (KB)
5. Kiến trúc mô hình đề xuất
6. Thiết kế thực nghiệm: baseline, ablation, câu hỏi
7. Kết quả khảo sát sơ bộ
8. Phạm vi claim
9. XAI (tùy chọn)
10. Hạn chế đã biết
11. Thuật ngữ

---

## 1. Bài toán và động lực

### 1.1. Bài toán

Phân loại ảnh X-quang xương vào **8 lớp**: 1 lớp bình thường và 7 loại u xương cụ thể. Chỉ có ảnh làm đầu vào; không có báo cáo chẩn đoán hình ảnh, không có metadata.

### 1.2. Vì sao khó

- **Lớp rất lệch.** Normal có 1.879 ảnh, trong khi osteofibroma chỉ có 44 và synovial osteochondroma 51. Mô hình dễ học tốt lớp lớn mà bỏ qua lớp hiếm.
- **Hình thái chồng lấn.** Nhiều loại u có chung dấu hiệu. Ví dụ, "tổn thương tiêu xương" gặp ở giant cell tumor, simple bone cyst và osteofibroma; "vùng hành xương của xương dài" gặp ở 4 lớp.
- **Dữ liệu nhỏ.** 3.586 ảnh, nên mạng sâu dễ học thuộc tập train.

### 1.3. Ý tưởng chính

Bác sĩ chẩn đoán dựa trên **dấu hiệu hình ảnh đã được mô tả trong y văn**: bờ tổn thương, phản ứng màng xương, chất nền, vị trí trên xương. Luận văn đưa các dấu hiệu này vào mô hình dưới dạng **mệnh đề tri thức** (proposition):
- mỗi mệnh đề là một câu ngắn có nguồn trích dẫn;
- mỗi mệnh đề được gán cho các lớp liên quan.

Mô hình dùng mệnh đề theo hai cách:
- **Làm query** để đọc ảnh (cross-attention), rồi tổng hợp theo từng lớp thành một nhánh dự đoán riêng: **knowledge decoder**.
- **Làm điểm neo** để tổ chức không gian đặc trưng ảnh: **PECL**.

**Mục tiêu đo được:** mô hình có tri thức phải đạt Macro-F1 cao hơn mô hình chỉ dùng ảnh. Phần tăng phải đến từ **nội dung** tri thức, kiểm chứng bằng đối chứng xáo trộn tri thức.

---

## 2. Liên hệ với hai bài nền

### 2.1. RAD (Li et al., NeurIPS 2025): nguồn ý tưởng phương pháp

| Thành phần RAD | Trong luận văn |
|---|---|
| Guideline của mỗi bệnh làm query trong decoder, cross-attention lên đặc trưng ảnh và văn bản | Mệnh đề làm query, cross-attention lên đặc trưng ảnh (chỉ có ảnh) |
| Dual decoder: nhánh guideline và nhánh nhãn, **mỗi nhánh có loss riêng** | Hai nhánh (image head và knowledge decoder), **mỗi nhánh có CE riêng**, cộng logit khi dự đoán |
| GECL: kéo đặc trưng ảnh/văn bản về prototype guideline (prototype cố định, không lấy gradient) | PECL: kéo đặc trưng ảnh về prototype lớp tạo từ mệnh đề |
| Table 5: ablation bật/tắt từng thành phần | Bảng ablation cùng khung (mục 6.3) |
| Bảng baseline: CNN, transformer, mô hình pretrain trong miền y tế | Bảng baseline cùng khung (mục 6.2) |

**Khác biệt cốt lõi.**
- Tri thức của RAD là guideline nguyên khối, do LLM tóm tắt. Tri thức ở đây là **các mệnh đề độc lập, có nguồn truy vết, được gán theo lớp**.
- RAD có modality văn bản (báo cáo, EHR). Chính RAD ghi nhận nhánh văn bản giúp nhiều hơn vì cùng modality với guideline. Luận văn **chỉ có ảnh**, nên mức tăng kỳ vọng nhỏ hơn RAD.

### 2.2. BoneVisionNet (Ajay et al., CMIG 2026): nguồn protocol và baseline

- **Kiến trúc của họ:** ba nhánh ảnh: CEiT kèm Global Context Block (toàn cục), một CNN tự thiết kế (mức giữa), DenseNet-169 kèm ECA (cục bộ). XAI bằng Grad-CAM, LIME, SHAP, đánh giá định tính.
- **Học theo:**
  - **DenseNet-169 là backbone** (họ báo đây là CNN tốt nhất, 72,68%);
  - siêu tham số giống hệt: batch 32, lr 1e-4, wd 1e-4, StepLR 25, Adam, CE, 100 epoch;
  - báo bảng chỉ số theo từng lớp (Table 9 của họ).
- **Không so trực tiếp.** Họ làm **3 lớp** (normal/benign/malignant), chia **ngẫu nhiên theo ảnh** 70/20/10, và oversample lớp malignant bằng lật/xoay 90–270° **ở cả train lẫn val**. Con số 84,35% của họ chỉ nhắc ở related work.

---

## 3. Dữ liệu và protocol thực nghiệm

### 3.1. Dataset

BTXRD (Yao et al., *Scientific Data* 2025): 3.746 ảnh. Loại hai nhóm gộp `other bt` và `other mt`. Ảnh `IMG001276` có hai nhãn, được gán synovial_osteochondroma. Còn lại **3.586 ảnh**.

| Lớp | Số ảnh | Ghi chú |
|---|---|---|
| normal | 1.879 | |
| osteochondroma | 753 | |
| multiple_osteochondromas | 263 | |
| simple_bone_cyst | 206 | |
| giant_cell_tumor | 93 | hiếm |
| osteofibroma | 44 | rất hiếm |
| synovial_osteochondroma | 51 | rất hiếm |
| osteosarcoma | 297 | ác tính |

### 3.2. Tiền xử lý

- Ảnh cache sẵn ở `03_cache/images_512` (PNG), resize về **224×224**.
- Ảnh xám nhân thành 3 kênh, chuẩn hóa theo mean/std của ImageNet. BiomedCLIP dùng mean/std của CLIP.

### 3.3. Augmentation và cân bằng lớp

**Protocol chính thức** (áp dụng sau khi chạy thử ở mục 6.5 xác nhận):

| Thành phần | Thiết lập |
|---|---|
| Phép biến đổi (chỉ train) | Lật ngang p = 0,5; RandomAffine (xoay ±7°, dịch ±3%, scale 0,95–1,05); ColorJitter độ sáng và tương phản ±10% |
| Cân bằng lớp | `WeightedRandomSampler` có thay thế; trọng số mỗi ảnh ∝ 1/√n_c (n_c = số ảnh train của lớp); mỗi epoch lấy đúng số ảnh train |
| Loss | CE **không trọng số** (sampler đã cân bằng, không bù hai lần) |
| Val / test | Không biến đổi gì |

Lý do:
- **Oversampling online** là bản online của cách BoneVisionNet tăng cường lớp hiếm: mỗi lần ảnh hiếm được lấy lại thì nhận một phép biến đổi ngẫu nhiên khác, không tạo bản sao giống hệt.
- **√-nghịch đảo thay vì nghịch đảo hoàn toàn**, vì osteofibroma chỉ khoảng 31 ảnh train; cân bằng tuyệt đối sẽ lặp mỗi ảnh khoảng 40 lần mỗi epoch, dễ học thuộc.
- **Lật ngang hợp lý** vì chi trái và chi phải đối xứng.
- **Không lật dọc, không xoay 90/180/270°**, vì ảnh test luôn ở hướng chụp chuẩn. Biến đổi tạo ra hướng không bao giờ gặp ở test không giúp tổng quát hóa vào đúng phân phối cần dự đoán.
- **Không tăng cường val/test.** BoneVisionNet làm điều này với val; luận văn không làm theo.

**Protocol khảo sát sơ bộ** (đã dùng cho các số ở mục 7): chỉ RandomAffine như trên, không lật, không jitter, không sampler; CE có trọng số 1/√n_c.

### 3.4. Đánh giá chính thức: CV 5 fold gom theo ca

| Hạng mục | Thiết lập |
|---|---|
| Chia fold | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`, phân tầng theo 8 lớp, **gom theo ca**: các view của cùng một ca luôn chung fold |
| Mỗi fold k | Test = fold k. Bốn fold còn lại chia tiếp train ≈ 90% / val nội bộ ≈ 10% (cũng phân tầng, gom theo ca) |
| Chọn checkpoint | Epoch có Macro-F1 cao nhất trên val nội bộ, tính trên logit cuối |
| Dùng chung | Mọi config dùng **cùng file fold**, cùng seed |
| Chỉ số chính | **Macro-F1 gộp ngoài fold**: ghép dự đoán test của 5 fold thành một tập phủ toàn bộ 3.586 ảnh |
| Chỉ số phụ | Accuracy, Balanced Accuracy, macro AUROC (one-vs-rest), precision/recall/F1 từng lớp, confusion matrix; mean ± sd Macro-F1 theo fold |
| Khoảng tin cậy | Bootstrap 2.000 lần **theo ca** (lấy mẫu lại ca, không lấy mẫu từng ảnh) trên dự đoán ngoài fold. So sánh hai mô hình bằng chênh lệch ghép cặp kèm 95% CI |
| Tiêu chí "khác biệt có ý nghĩa" | 95% CI của chênh lệch ghép cặp không chứa 0. Báo kèm số fold thắng (k/5) |

Vì sao cần CV (bằng chứng ở mục 7.4): trên một split 538 ảnh, các lớp nhỏ ở test chỉ có 7–14 ảnh. Chênh lệch Macro-F1 giữa hai mô hình vì thế có độ bất định khoảng ±5 điểm. Gộp dự đoán ngoài fold đưa tập đánh giá lên 3.586 ảnh, nên khoảng tin cậy hẹp lại.

Nhóm ca dùng file nhóm đã dựng ở bước chuẩn bị dữ liệu (ca heuristic, gộp cả ảnh gần trùng). BTXRD không có mã bệnh nhân, nên đây là nhóm heuristic. Phải ghi rõ điều này trong luận văn.

### 3.5. Recipe huấn luyện

| Họ mô hình | Optimizer | Lịch LR | Epoch | Batch | Khác |
|---|---|---|---|---|---|
| CNN (DenseNet-169, ResNet-50, EfficientNet-B3) và **mọi dòng ablation** | Adam, lr 1e-4, wd 1e-4 | StepLR(25, 0,1) | 100 | 32 | AMP; theo BoneVisionNet |
| Transformer (ViT-B/16, BiomedCLIP ViT-B/16) | AdamW, lr 5e-5, wd 0,05, LLRD 0,75 | warmup 5 + cosine | 50 | 16 × accum 2 | clip grad 1,0; recipe fine-tune ViT chuẩn |

Không tune siêu tham số cho từng mô hình. Seed 2026.

---

## 4. Knowledge base (KB)

### 4.1. Phiên bản và file

- **KB 8 lớp v1.0**, khóa ngày 08/11/2026.
- File: `02_kb/btxrd_kb_8class.json`, sha256 `51be885fb22b275c2e64f0d93c520cf94b37d54a1989c6239fff724a82da25fb`.
- Kèm `propositions_8class.csv`, `sources_8class.md`, `kb_8class_report.md`.

### 4.2. Mệnh đề là gì

Mỗi mệnh đề gồm:
- một câu tiếng Anh ngắn, một ý, mô tả một dấu hiệu quan sát được trên X-quang;
- **không chứa tên bệnh**;
- mã nguồn và đoạn trích nguyên văn;
- loại: `characteristic` (đặc trưng) hoặc `shared` (dùng chung);
- danh sách lớp được gán.

Ví dụ thật trong KB:

| ID | Mệnh đề | Loại | Lớp được gán | Nguồn |
|---|---|---|---|---|
| P001 | The cortex forms a dense compact layer along the bone surface. | characteristic | normal | S10 |
| P020 | The lesion is predominantly lytic. | shared | giant_cell_tumor, simple_bone_cyst, osteofibroma | S10, S07, S11 |
| P033 | The bony protrusion arises from the outer bone surface and points away from the joint. | characteristic | osteochondroma, multiple_osteochondromas | S10 |
| P038 | Divergent sunburst spicules radiate from the lesion. | characteristic | osteosarcoma | S10, S05 |
| P043 | Numerous calcified intra-articular bodies of similar size are present. | characteristic | synovial_osteochondroma | S08, S12 |
| P050 | The lesion involves the metaphyseal region of a long bone. | shared | osteochondroma, multiple_osteochondromas, simple_bone_cyst, osteosarcoma | S06, S07, S10 |
| P060 | The tibia shows anterior bowing. | characteristic | osteofibroma | S11 |

**Quan trọng:** nhãn lớp của mệnh đề là **ánh xạ tri thức ở mức lớp**, không phải nhãn của từng bệnh nhân. Mô hình **không** được giám sát để dự đoán mệnh đề nào có mặt trên một ảnh cụ thể. Không có nhãn yếu, không có giám sát ở mức mệnh đề.

### 4.3. Thống kê

- Tổng 68 mệnh đề:
  - P001–P065 là mệnh đề hình ảnh, **được dùng**;
  - P066–P068 mô tả tuổi/giới (metadata), **bị loại** vì mô hình chỉ dùng ảnh.
- Trong 68 mệnh đề: 38 `characteristic`, 30 `shared`. 49 mệnh đề thuộc 1 lớp, 13 thuộc 2 lớp, 4 thuộc 3 lớp, 2 thuộc 4 lớp.
- Số mệnh đề hình ảnh theo lớp (một mệnh đề có thể thuộc nhiều lớp):

| normal | osteochondroma | multiple_osteochondromas | simple_bone_cyst | giant_cell_tumor | osteofibroma | synovial_osteochondroma | osteosarcoma |
|---|---|---|---|---|---|---|---|
| 12 | 8 | 10 | 13 | 12 | 7 | 9 | 16 |

### 4.4. Nguồn

13 nguồn mở, đọc được toàn văn: review trên PMC/BMC/RSNA và sách *WHO Manual of Diagnostic Imaging* (S10). Danh sách đầy đủ kèm DOI và giấy phép ở `02_kb/sources_8class.md`. Nguồn bị loại (không đọc được toàn văn) cũng được ghi lại.

### 4.5. Mã hóa văn bản

- Encoder `ncbi/MedCPT-Query-Encoder`, **đóng băng**, lấy token [CLS], max_length 64 → mỗi mệnh đề là một vector **t_j ∈ ℝ⁷⁶⁸**.
- Tính một lần, lưu cache `03_cache/medcpt_query_kb_v1.pt` (sha256 `a7566bda…d91`). Code kiểm tra sha256 của KB với cache trước khi train.
- **Prototype lớp** c: g_c = normalize( mean_{j ∈ K_c} normalize(t_j) ) ∈ ℝ⁷⁶⁸, với K_c là tập mệnh đề hình ảnh của lớp c.

---

## 5. Kiến trúc mô hình đề xuất

### 5.1. Sơ đồ tổng quát

```text
Ảnh X-quang x (3×224×224)
        │
        ▼
 DenseNet-169 (ImageNet) ──► feature map F (1664×7×7)
        │                         │
        │ GAP                     │ flatten → 49 visual token V (49×1664)
        ▼                         ▼
   z_img (1664)          ┌────────────────────────────────────────┐
        │                │ (A) Proposition cross-attention          │
        │                │   query: 65 mệnh đề  t_j (MedCPT, đóng băng)│
        │                │   key/value: 49 visual token             │
        │                │   → h_j (65×256): "mệnh đề j đọc ảnh"     │
        │                ├────────────────────────────────────────┤
        │                │ (B) Knowledge decoder theo lớp           │
        │                │   query: prototype lớp g_c (8)            │
        │                │   chỉ attend lên mệnh đề của lớp c (mask) │
        │                │   → r_c (8×256) → scorer → logits_kb (8)  │
        │                └────────────────────────────────────────┘
        │                                   │
        ├─► (C) Image head: Linear → logits_img (8)
        │                                   │
        │            logits = logits_img + β · logits_kb      (dự đoán)
        │
        └─► (D) PECL: φ(z_img) so với prototype lớp g_c  (chỉ khi train)

Loss = CE(logits) + λ_img·CE(logits_img) + λ_kb·CE(logits_kb) + λ_pecl·L_PECL
```

### 5.2. Image encoder: DenseNet-169

- Pretrain ImageNet, fine-tune toàn bộ.
- Lấy feature map cuối sau ReLU: **F ∈ ℝ^{1664×7×7}**.
- **Visual token:** flatten 7×7 thành **49 token**, mỗi token V_i ∈ ℝ¹⁶⁶⁴ ứng với một vùng ảnh khoảng 32×32 pixel.
- **Đặc trưng toàn cục:** z_img = GAP(F) ∈ ℝ¹⁶⁶⁴.

### 5.3. (A) Proposition cross-attention: mệnh đề đọc ảnh

- Query: q_j = W_q t_j ∈ ℝ²⁵⁶ (Linear 768→256), j = 1…65.
- Key/value: k_i = v_i = W_v V_i ∈ ℝ²⁵⁶ (Linear 1664→256), i = 1…49.
- Multi-head attention, 4 head, d = 256, dropout 0,1:

  h_j = MHA(q_j, {k_i}, {v_i}) = Σ_i α_{ji} · v_i (theo từng head, rồi ghép lại và chiếu)

- Kết quả: **H ∈ ℝ^{65×256}**, mỗi hàng h_j là "ảnh này nhìn qua lăng kính mệnh đề j". Trọng số α ∈ ℝ^{65×49} cho biết mệnh đề j chú ý vùng nào.
- Vì t_j đóng băng, phần học được là W_q, W_v và attention.

### 5.4. (B) Knowledge decoder theo lớp

- Mỗi lớp c có một query là prototype của chính nó: p_c = W_g g_c ∈ ℝ²⁵⁶ (Linear 768→256).
- Attention 1 head, có **mask**: p_c chỉ được attend lên các h_j với j ∈ K_c (mệnh đề của lớp c):

  r_c = Σ_{j ∈ K_c} β_{cj} · h_j,  β_{cj} = softmax_{j ∈ K_c}(⟨p_c, h_j⟩ / √d)

- Điểm của lớp: **logits_kb,c = wᵀ r_c + b**, một `Linear(256,1)` dùng chung cho 8 lớp. Kết quả logits_kb ∈ ℝ⁸.
- **Ý nghĩa:** điểm của lớp c chỉ được tạo từ bằng chứng của các mệnh đề thuộc lớp c. Cấu trúc ánh xạ mệnh đề → lớp của KB được đưa thẳng vào kiến trúc qua mask.

### 5.5. (C) Image head và công thức dự đoán

- logits_img = W_img z_img + b_img (Linear 1664→8), giống hệt B0.
- **Dự đoán:** logits = logits_img + β · logits_kb, với β là một vô hướng học được, khởi tạo 0,1.
- Vì logits_img giữ nguyên dạng của B0, mô hình đề xuất **chứa B0 như một trường hợp riêng** (β = 0). Nhánh tri thức là phần cộng thêm, đóng góp của nó đo được trực tiếp.

### 5.6. Giám sát riêng hai nhánh

- L_cls = CE(logits) + λ_img · CE(logits_img) + λ_kb · CE(logits_kb), với λ_img = λ_kb = 1,0, cố định, không tune.
- **Lý do** (theo RAD, mỗi nhánh có loss riêng): nếu chỉ giám sát logit tổng, nhánh tri thức không bị buộc phải tự phân biệt 8 lớp. Nó có thể chỉ học một độ lệch cố định theo lớp, hoặc lặp lại thông tin image head đã có. Khi có CE riêng, logits_kb phải tự phân loại được. `[đã kiểm chứng]` Trong khảo sát sơ bộ, nhánh tri thức đứng riêng đạt Macro-F1 val 63,48% (mục 7.2).

### 5.7. (D) PECL: Proposition-Enhanced Contrastive Loss

- u = normalize( φ(z_img) ) ∈ ℝ⁷⁶⁸, với φ là Linear 1664→768.
- Prototype g_c cố định (đóng băng, không lấy gradient), như prototype của GECL trong RAD.
- s_c = ⟨u, g_c⟩ / τ, τ = 0,07; L_PECL = CE(s, y).
- Hệ số λ_pecl = 0,1, theo hệ số β = 0,1 của GECL trong RAD.
- **Chỉ tác động lúc train:** kéo đặc trưng ảnh của ca thuộc lớp c về gần prototype tri thức của lớp c, và xa prototype các lớp khác. Không đổi công thức dự đoán.
- Vì g_c dựng từ nội dung mệnh đề, khoảng cách giữa các prototype phản ánh mức giống nhau về dấu hiệu hình ảnh giữa các lớp. Ví dụ, osteochondroma và multiple_osteochondromas chia nhau nhiều mệnh đề nên prototype gần nhau. `[giả thuyết]` Cấu trúc này giúp lớp hiếm "mượn" thông tin từ lớp gần nó.

### 5.8. Tổng loss và số tham số

L = CE(logits) + 1,0·CE(logits_img) + 1,0·CE(logits_kb) + 0,1·L_PECL

| Khối | Tham số |
|---|---|
| DenseNet-169 encoder | 12.484.480 |
| Image head (1664→8) | 13.320 |
| Proposition cross-attention (W_v, W_q, MHA 4 head) | 886.272 |
| Knowledge decoder (W_g, MHA 1 head, scorer) | 460.289 |
| β | 1 |
| **Tổng, chưa có PECL** | **13.844.362** (B0: 12.497.800) |
| φ của PECL (1664→768) | khoảng 1,28 triệu (thêm khi bật PECL) |

### 5.9. Biến thể đối chứng: tri thức xáo trộn

- Giữ đúng 65 vector MedCPT thật, nhưng **hoán vị ánh xạ mệnh đề ↔ vector** (seed 0). Ví dụ, P001 nhận vector của một mệnh đề khác.
- Hoán vị áp dụng trước khi dựng query, prototype của decoder và prototype của PECL, nên **mọi nơi dùng tri thức đều bị xáo trộn**.
- Mask lớp và số mệnh đề mỗi lớp giữ nguyên. Kiến trúc, số tham số, optimizer giữ nguyên.
- Kết quả hoán vị: 0/65 mệnh đề giữ vector của chính nó; 11/65 nhận vector của một mệnh đề cùng lớp (ngẫu nhiên).
- Chỉ phá **sự tương ứng ngữ nghĩa** giữa tri thức và lớp. Nếu mô hình thật hơn bản xáo trộn, phần tăng đến từ nội dung tri thức chứ không phải từ kiến trúc thêm vào.

---

## 6. Thiết kế thực nghiệm

### 6.1. Nguyên tắc

- Chỉ dùng **DenseNet-169** làm backbone cho mô hình đề xuất và ablation.
- Mọi run chính thức: CV 5 fold (mục 3.4), cùng augmentation và cân bằng lớp (mục 3.3), cùng file fold.
- Không tune siêu tham số theo từng mô hình. Không thêm run ngoài danh sách dưới đây (trừ XAI).

### 6.2. Bảng baseline (chỉ ảnh), theo bảng baseline của RAD

| ID | Mô hình | Pretrain | Họ |
|---|---|---|---|
| B0 | DenseNet-169 | ImageNet | CNN; CNN tốt nhất trong BoneVisionNet |
| B-R50 | ResNet-50 | ImageNet | CNN; backbone chính của RAD |
| B-EB3 | EfficientNet-B3 | ImageNet | CNN; hay dùng trên BTXRD |
| B-ViT | ViT-B/16 | ImageNet | Transformer; RAD dùng trong ablation |
| B-BMC | BiomedCLIP ViT-B/16 (image encoder) | PMC-15M (cặp ảnh–chú thích y sinh) | Pretrain trong miền y tế |

Tất cả fine-tune toàn bộ mạng, head tuyến tính trên đặc trưng toàn cục, input 224. EfficientNet-B3 cũng dùng 224 (thay vì 300 gốc) để đồng nhất.

### 6.3. Bảng ablation trên DenseNet-169, theo Table 5 của RAD

| ID | Knowledge decoder (+ giám sát hai nhánh) | PECL | Nội dung tri thức |
|---|---|---|---|
| A0 = B0 | × | × | — |
| A1 | × | ✓ | thật |
| A2 | ✓ | × | thật |
| **A3 (mô hình đề xuất)** | ✓ | ✓ | thật |
| A4 (đối chứng) | ✓ | ✓ | **xáo trộn** (seed 0) |

### 6.4. Câu hỏi thực nghiệm và cách trả lời

| Câu hỏi | So sánh | Trả lời "có" khi |
|---|---|---|
| **Q1.** Knowledge decoder có cải thiện so với chỉ dùng ảnh không? | A2 so với A0 | Chênh Macro-F1 gộp ngoài fold > 0, 95% CI không chứa 0 |
| **Q2.** PECL có cải thiện không? | A1 so với A0; A3 so với A2 | Cả hai chênh > 0; nói "có ý nghĩa" chỉ khi CI không chứa 0 |
| **Q3.** Kết hợp hai thành phần có tốt nhất không? | A3 so với A0, A1, A2 | A3 cao nhất; CI so với A0 không chứa 0 |
| **Q4.** Phần cải thiện có đến từ nội dung tri thức không? | A3 so với A4 | Chênh > 0, CI không chứa 0 |
| **Q5.** Mô hình đề xuất đứng ở đâu so với baseline chuẩn? | A3 so với B0, B-R50, B-EB3, B-ViT, B-BMC | Mô tả; chỉ nói "tốt hơn X" khi CI so với X không chứa 0 |

Kèm theo mỗi so sánh: F1 từng lớp, đặc biệt các lớp hiếm (giant_cell_tumor, osteofibroma, synovial_osteochondroma).

### 6.5. Chạy thử augmentation (trước CV)

- 1 run B0 với augmentation và sampler mới (mục 3.3), trên split cố định 2510/538/538.
- So với val Macro-F1 cao nhất của B0 cũ: **59,51**.
- **Quy tắc:** nếu không thấp hơn quá 1 điểm thì dùng protocol mới cho mọi run CV; nếu thấp hơn quá 1 điểm thì giữ protocol sơ bộ (chỉ affine, CE có trọng số; class weight tính lại theo train của từng fold).

### 6.6. Danh sách run chính thức

| Nhóm | Config | Số lần train |
|---|---|---|
| Chạy thử | B0, augmentation mới, split cố định | 1 |
| Baseline | B0, B-R50, B-EB3, B-ViT, B-BMC | 5 × 5 fold = 25 |
| Ablation (A0 = B0, dùng chung) | A1, A2, A3, A4 | 4 × 5 fold = 20 |
| **Tổng** | | **46** |

Ước lượng thời gian: một run DenseNet 100 epoch trên Colab T4 mất khoảng 1,1–1,9 giờ (đo từ khảo sát sơ bộ), nên tổng cộng khoảng 50–90 giờ GPU.

---

## 7. Kết quả khảo sát sơ bộ

> Split cố định 2510/538/538, 1 seed, protocol sơ bộ (mục 3.3). **Không phải số chính thức.** Dùng để chọn hướng và để giải thích vì sao cần CV.

### 7.1. Tổng quan

| Run (DenseNet-169) | Val Macro-F1 (best) | Test Acc | Test BA | Test Macro-F1 | Test AUROC |
|---|---|---|---|---|---|
| B0 — chỉ ảnh | 59,51 (epoch 14) | 74,54 | 60,35 | 61,78 | 90,78 |
| KD-real — decoder + giám sát hai nhánh, tri thức thật (chưa có PECL) | **65,70** (epoch 98) | 77,70 | 58,70 | 62,34 | 91,36 |
| KD-shuffled — như trên, tri thức xáo trộn | 61,82 (epoch 86) | 79,55 | 63,39 | 66,02 | 92,60 |

KD-real tương ứng dòng A2 của ablation; KD-shuffled là A2 với tri thức xáo trộn.

### 7.2. Theo từng nhánh (val, checkpoint best)

| Macro-F1 val | Image head | Nhánh tri thức đứng riêng | Dự đoán cuối | Trung bình epoch 30–99 (cuối) |
|---|---|---|---|---|
| KD-real | 63,36 | **63,48** | **65,70** | 63,5 |
| KD-shuffled | 62,00 | 56,76 | 61,82 | 57,7 |

- `[đã kiểm chứng]` Với giám sát hai nhánh, nhánh tri thức tự phân loại được (63,48%). β học tăng từ 0,10 lên 0,164, tức mô hình tự tăng trọng số cho nhánh tri thức.
- `[đã kiểm chứng]` Trên val, tri thức thật hơn tri thức xáo trộn ổn định suốt quá trình train (+5,9 điểm trung bình epoch 30–99), không chỉ ở một checkpoint.

### 7.3. F1 từng lớp (test)

| Lớp (support test) | B0 | KD-real | KD-shuffled |
|---|---|---|---|
| normal (282) | 84,2 | 86,7 | 89,0 |
| osteochondroma (113) | 61,7 | 67,0 | 66,7 |
| multiple_osteochondromas (39) | 77,1 | 70,4 | 75,0 |
| simple_bone_cyst (31) | 45,2 | 52,0 | 56,7 |
| giant_cell_tumor (14) | 36,4 | 41,7 | 48,0 |
| osteofibroma (7) | 42,9 | 30,8 | 33,3 |
| synovial_osteochondroma (7) | 61,5 | 66,7 | 76,9 |
| osteosarcoma (45) | 85,4 | 83,5 | 82,6 |

### 7.4. Độ bất định: vì sao cần CV

Bootstrap ghép cặp 2.000 lần (theo ảnh) trên chênh lệch Macro-F1:

| So sánh | Val | Test |
|---|---|---|
| KD-real − KD-shuffled | +3,88 [−1,37; +9,11] | −3,68 [−9,31; +1,42] |
| KD-real − B0 | (không có dự đoán val của B0) | +0,56 [−7,31; +7,67] |

- Val và test **cho hướng ngược nhau**; mọi khoảng tin cậy đều chứa 0.
- Nguyên nhân: các lớp ở test có 7–14 ảnh, đoán đúng thêm hay sai thêm 1–2 ảnh làm F1 của lớp đó lệch 10–15 điểm.
- **Quyết định đã chốt:** theo quy tắc quyết định trên val, giữ hướng tri thức thật > xáo trộn. **Claim cuối cùng lấy từ CV** (mục 3.4).

### 7.5. Ghi chú về loss khi train

- `val_loss_total` là tổng của 3 CE (hoặc 4 loss khi có PECL), nên lớn hơn khoảng 3 lần một CE đơn. Muốn so với mô hình khác thì đọc `val_loss_ce_final`.
- CE val tăng dần trong khi F1 vẫn tăng: mô hình học thuộc train (train CE khoảng 0,003), các ca đoán sai trên val bị đoán sai với độ tự tin cao. Nhánh tri thức bị nặng nhất (CE val 1,34 → 2,34).
- Nếu hiện tượng này làm hỏng kết quả CV thì dùng label smoothing 0,1 cho **mọi** config, không áp riêng cho một config.

---

## 8. Phạm vi claim

**Được claim, nếu CV ủng hộ (CI không chứa 0):**
- Tri thức y khoa dạng mệnh đề, tích hợp qua knowledge decoder theo lớp với giám sát hai nhánh và PECL, cải thiện Macro-F1 so với mô hình chỉ dùng ảnh trên BTXRD 8 lớp (Q1–Q3).
- Phần cải thiện phụ thuộc vào nội dung tri thức (Q4, so với đối chứng xáo trộn).
- So sánh với các baseline chuẩn trong cùng protocol (Q5).

**Nếu CI chứa 0:** viết là "không có khác biệt có ý nghĩa thống kê", kèm số fold thắng. Không làm tròn thành "cải thiện".

**Không claim, trong mọi trường hợp:**
- hơn BoneVisionNet (khác bài toán và protocol);
- mô hình nhận ra từng dấu hiệu, hay xác suất của một mệnh đề có ý nghĩa lâm sàng;
- mô hình "lập luận như bác sĩ";
- attention là lời giải thích nhân quả;
- hiệu lực lâm sàng ngoài phạm vi dataset.

---

## 9. XAI (tùy chọn, không thuộc phần lõi)

- **Grad-CAM post-hoc** trên logit cuối, lớp feature map 7×7 cuối của DenseNet. Không cần train thêm. Vì logits = logits_img + β·logits_kb, có thể vẽ riêng heatmap của từng nhánh.
- **Đánh giá định lượng** bằng bounding box/mask chuyên gia của BTXRD, trên ảnh u (bỏ lớp normal):
  - pointing game: điểm nóng nhất của heatmap có nằm trong box không;
  - energy inside lesion: tỷ lệ năng lượng heatmap nằm trong box.

  So B0 với A3, tương tự phân tích grounding ở Table 3 của RAD.
- LIME và SHAP chỉ để minh họa định tính, như BoneVisionNet.
- **Không** dùng attention mệnh đề → vùng ảnh làm lời giải thích chính: chưa có cách kiểm chứng một mệnh đề cụ thể nhìn đúng chỗ.

---

## 10. Hạn chế đã biết

1. **Không có cầu nối ảnh–chữ.** Vector mệnh đề đến từ encoder chỉ học văn bản (MedCPT), chưa từng gióng hàng với ảnh. Mô hình không được dạy "mệnh đề j trông thế nào trên ảnh" (không có nhãn mệnh đề). Tri thức tác động qua cấu trúc lớp–mệnh đề và hình học prototype, không qua việc nhận diện từng dấu hiệu.
2. **Lớp rất nhỏ** (44–93 ảnh), nên kể cả với CV, kết luận trên các lớp này vẫn kém chắc chắn.
3. **Nhóm ca là heuristic**, vì BTXRD không có mã bệnh nhân.
4. **Một dataset, một backbone cho mô hình đề xuất.** Chưa kiểm tra trên dữ liệu ngoài.
5. **KB do người làm luận văn biên soạn** từ y văn, chưa được bác sĩ chẩn đoán hình ảnh thẩm định.

---

## 11. Thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Mệnh đề (proposition) | Một câu mô tả một dấu hiệu X-quang, có nguồn, gán cho các lớp liên quan |
| Query / key / value | Trong attention: query "hỏi", key/value là nơi được "đọc". Ở đây mệnh đề hỏi, vùng ảnh được đọc |
| Cross-attention | Attention giữa hai nguồn khác nhau (văn bản → ảnh) |
| Visual token | Vector đặc trưng của một vùng ảnh (49 vùng trên lưới 7×7) |
| Prototype lớp | Vector đại diện tri thức của một lớp: trung bình vector các mệnh đề của lớp |
| Logit | Điểm chưa chuẩn hóa của mỗi lớp; softmax của logit cho ra xác suất |
| β | Trọng số học được, quyết định nhánh tri thức đóng góp bao nhiêu vào dự đoán |
| Giám sát hai nhánh | Mỗi nhánh (ảnh, tri thức) có loss CE riêng, ngoài loss của dự đoán cuối |
| PECL | Loss contrastive kéo đặc trưng ảnh về prototype tri thức đúng lớp |
| Contrastive | Học bằng cách kéo cặp đúng lại gần và đẩy cặp sai ra xa |
| GAP | Global average pooling: trung bình feature map theo không gian |
| Macro-F1 | Trung bình F1 của 8 lớp, mỗi lớp trọng số như nhau, nên lớp hiếm được tính ngang lớp lớn |
| Dự đoán ngoài fold (out-of-fold) | Dự đoán của mỗi ảnh lấy từ mô hình không train trên fold chứa ảnh đó |
| Bootstrap theo ca | Lấy mẫu lại các ca (có hoàn lại) nhiều lần để ước khoảng tin cậy |
| Đối chứng xáo trộn | Giữ nguyên vector tri thức nhưng hoán vị gán mệnh đề ↔ vector, để tách hiệu ứng nội dung khỏi hiệu ứng kiến trúc |
