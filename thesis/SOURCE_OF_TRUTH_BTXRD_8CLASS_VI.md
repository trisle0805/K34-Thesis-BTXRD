# SOURCE OF TRUTH — BTXRD 8 lớp

## Phân loại u xương có tích hợp tri thức y khoa theo proposition

> **Trạng thái:** Đã cập nhật sau khi chuyển từ bài toán 3 lớp sang 8 lớp và sau khi hoàn thành các baseline B0/B1.
>
> **Mục đích:** Đây là tài liệu chuẩn duy nhất cho hướng nghiên cứu BTXRD hiện tại, bao gồm task, kiến trúc mô hình, tổ chức code, ablation, đánh giá và phạm vi claim.
>
> **Nguyên tắc cốt lõi:** Giữ câu chuyện nghiên cứu tập trung: một ý tưởng chính là proposition-structured medical knowledge, một kiến trúc knowledge-guided rõ ràng, một auxiliary Contrastive Learning hợp lý, số lượng ablation vừa đủ và phần XAI dựa trên lesion grounding.

---

# 1. Bài toán nghiên cứu

## 1.1. Dataset

Dataset sử dụng: **BTXRD**

Tổng số ảnh raw:

\[
3746
\]

Hai nhóm composite sau không được dùng trong task chính:

- `other bt`
- `other mt`

Một ảnh có nhãn chồng lấn được xử lý ở bước preprocessing:

- `IMG001276.jpeg` → `synovial_osteochondroma`

Sau khi loại hai nhóm composite, dataset chính còn:

\[
N=3586
\]

ảnh.

## 1.2. Các lớp

Task chính là phân loại **8 lớp**:

1. `normal`
2. `osteochondroma`
3. `multiple_osteochondromas`
4. `simple_bone_cyst`
5. `giant_cell_tumor`
6. `osteofibroma`
7. `synovial_osteochondroma`
8. `osteosarcoma`

Phân bố:

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

## 1.3. Data split

Dùng một split cố định, stratified theo 8 lớp:

- Train: **2510**
- Validation: **538**
- Test: **538**

Tất cả experiment nội bộ phải dùng cùng split này.

Test set không được dùng để chọn architecture hoặc hyperparameter.

## 1.4. Chỉ số chính

Metric chính:

\[
\boxed{\text{Macro-F1}}
\]

Các metric bổ sung:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Precision / Recall / F1 theo từng class
- One-vs-Rest AUROC
- confusion matrix

Với các class có số mẫu nhỏ, per-class F1 phải được diễn giải cùng với support.

---

# 2. Baseline đã hoàn thành

## B0 — Image-only

Backbone:

> DenseNet-169, ImageNet-pretrained

Kết quả test chính:

- Accuracy ≈ **74.54%**
- Balanced Accuracy ≈ **60.35%**
- Macro-F1 ≈ **61.78%**
- Macro AUROC ≈ **90.78%**

B0 là **baseline nội bộ chính**.

## B1 — Image + Metadata

Metadata được dùng:

- age
- gender
- coarse anatomical region
- view

Naive late fusion không cải thiện so với B0.

Kết quả test chính:

- Accuracy ≈ **75.09%**
- Balanced Accuracy ≈ **54.78%**
- Macro-F1 ≈ **58.49%**
- Macro AUROC ≈ **88.02%**

Kết luận:

> Naive metadata fusion không được dùng trong B2/B3 chính.

B1 vẫn được giữ như một auxiliary baseline.

---

# 3. Medical Knowledge Base

## 3.1. Phiên bản KB

Dùng:

> **BTXRD 8-class KB v1.0**

Các file:

```text
02_kb/
├── btxrd_kb_8class.json
├── propositions_8class.csv
├── sources_8class.md
└── kb_8class_report.md
```

## 3.2. Quy mô KB

Tổng số proposition:

\[
M=68
\]

Theo `role`:

- characteristic: 38
- shared: 30

Một proposition có thể được map với nhiều class nếu nguồn y khoa hỗ trợ.

## 3.3. Proposition hình ảnh và proposition metadata

Các proposition dùng trong main visual model:

\[
P001\text{–}P065
\]

Do đó:

\[
M_{visual}=65
\]

Ba proposition:

- P066 — tuổi trẻ
- P067 — tuổi 20–50 / skeletal maturity
- P068 — male

là **metadata propositions**, không phải dấu hiệu nhìn trực tiếp trên X-ray.

Vì vậy:

> **B2/B3 chính chỉ dùng P001–P065.**

P066–P068 vẫn được giữ trong KB nhưng chỉ bật nếu một experiment sau này thực sự đưa metadata vào patient memory.

## 3.4. Vai trò của proposition

Một proposition là:

> **một đơn vị tri thức y khoa fine-grained được dùng làm semantic query.**

Proposition không phải:

- patient-level label
- PRESENT / ABSENT target
- concept bottleneck
- clinical probability
- symbolic rule

Không có proposition-level supervision.

## 3.5. Các lưu ý về taxonomy

Hai giới hạn cần ghi rõ trong paper:

1. `osteofibroma` được biểu diễn bằng literature của **osteofibrous dysplasia (OFD)** theo quy ước của KB; chưa có xác nhận từ tác giả BTXRD rằng hai nhãn là hoàn toàn tương đương.
2. `synovial_osteochondroma` được diễn giải theo literature về **synovial (osteo)chondromatosis**.

Đây là limitation của taxonomy dataset/KB, không phải claim của mô hình.

---

# 4. Tổ chức code

Từ giai đoạn này, toàn bộ logic mô hình không được đặt chủ yếu trong Colab notebook.

## 4.1. GitHub-first workflow

Toàn bộ:

- architecture
- data loading
- training
- losses
- metrics
- evaluation
- ablation switches

phải nằm trong GitHub repository.

Colab chỉ đóng vai trò GPU runner.

Workflow:

```text
GitHub repository
        ↓
clone / pull
        ↓
pip install -e .
        ↓
mount Google Drive
        ↓
chọn YAML config
        ↓
train / evaluate trên GPU
```

## 4.2. Cấu trúc repo đề xuất

```text
src/btxrd/
├── data/
│   ├── dataset.py
│   └── transforms.py
│
├── knowledge/
│   ├── kb.py
│   └── text_encoder.py
│
├── models/
│   ├── image_encoder.py
│   ├── cross_attention.py
│   ├── aggregation.py
│   ├── fusion.py
│   └── model.py
│
├── losses/
│   ├── classification.py
│   └── contrastive.py
│
├── training/
│   ├── trainer.py
│   ├── metrics.py
│   └── checkpoint.py
│
└── utils/
    └── seed.py

configs/
├── b0_image.yaml
├── b1_metadata.yaml
├── b2_knowledge.yaml
├── b3_contrastive.yaml
└── ablations/
    └── a1_mean_aggregation.yaml

scripts/
├── train.py
├── evaluate.py
└── cache_text_embeddings.py
```

Cùng một codebase phải chạy được toàn bộ experiment thông qua config.

---

# 5. Kiến trúc chính

## 5.1. Image encoder

Input:

\[
I_i\in\mathbb{R}^{3\times224\times224}
\]

Ảnh grayscale được lặp thành 3 channels để dùng ImageNet-pretrained DenseNet.

Backbone:

> **DenseNet-169, ImageNet-pretrained**

Lấy feature map cuối:

\[
F_i\in\mathbb{R}^{7\times7\times1664}
\]

sau activation cuối của DenseNet.

Global image feature:

\[
z_{img,i}
=
GAP(F_i)
\in\mathbb{R}^{1664}
\]

để giữ tương thích với representation của B0.

## 5.2. Visual tokens

Flatten feature map:

\[
F_i^{flat}
\in
\mathbb{R}^{49\times1664}
\]

sau đó chiếu về 256 chiều:

\[
V_i
=
W_vF_i^{flat}
\in
\mathbb{R}^{49\times256}
\]

Có thể triển khai bằng Linear layer hoặc \(1\times1\) convolution.

### Positional embedding

Cấu hình chính:

```text
use_positional_embedding = false
```

Không dùng learned 2D positional embedding trong main model.

Lý do:

- DenseNet feature đã giữ local spatial context;
- absolute position trên X-ray thay đổi theo anatomy, view và crop;
- attention map vẫn có thể reshape về grid \(7\times7\) để visualization.

Nếu sau này thật sự cần, có thể hỗ trợ fixed 2D sine/cosine encoding qua config switch, nhưng đây không phải core ablation.

---

# 6. Frozen proposition embeddings

## 6.1. Text encoder

Dùng một biomedical sentence encoder đã đóng băng.

Mặc định:

> **MedCPT query encoder**

Lý do:

- proposition là các biomedical sentence ngắn;
- MedCPT được huấn luyện contrastively cho biomedical semantic representation;
- cùng text space sẽ được dùng cho semantic query và class prototype.

Không thêm text-encoder comparison vào core experiment.

## 6.2. Cache offline

Với proposition \(p_j\):

\[
t_j
=
E_T(p_j)
\in
\mathbb{R}^{768}
\]

Text encoder được freeze hoàn toàn.

Tất cả \(t_j\) được tính offline và cache trước training.

Không backpropagate vào text encoder.

## 6.3. Query projection

Với 65 visual propositions:

\[
q_j
=
W_q t_j
\in
\mathbb{R}^{256}
\]

Trong đó \(W_q\) là trainable projection.

Lưu ý khi implementation:

> Không tự tạo hai lớp query projection dư thừa.

Nếu dùng PyTorch `MultiheadAttention`, \(W_q\) chỉ có nhiệm vụ map text embedding 768→256; các Q/K/V projection bên trong attention module sẽ được xử lý nội bộ.

---

# 7. Proposition-guided cross-attention

Tập query:

\[
Q
=
[q_1,\ldots,q_{65}]
\in
\mathbb{R}^{65\times256}
\]

Với ca bệnh \(i\):

- query = proposition embeddings \(Q\)
- key = visual tokens \(V_i\)
- value = visual tokens \(V_i\)

Cross-attention:

\[
H_i
=
CrossAttn(Q,V_i,V_i)
\in
\mathbb{R}^{65\times256}
\]

Với mỗi proposition:

\[
h_{ij}\in\mathbb{R}^{256}
\]

được hiểu là:

> **patient representation conditioned on proposition \(p_j\)**

Không được diễn giải \(h_{ij}\) thành:

- xác suất proposition xuất hiện;
- proposition verification;
- clinical evidence score.

Các hyperparameter mặc định có thể cấu hình:

```text
knowledge_dim = 256
num_heads = 4
attention_dropout = 0.1
```

Đây chỉ là implementation defaults, không phải research claims.

---

# 8. Tổng hợp proposition

## 8.1. Cấu hình chính — learned attention pooling

Với mỗi proposition-conditioned feature:

\[
e_{ij}
=
w^T\tanh(W_hh_{ij})
\]

sau đó:

\[
\alpha_{ij}
=
softmax_j(e_{ij})
\]

Knowledge representation:

\[
z_{kb,i}
=
\sum_{j=1}^{65}
\alpha_{ij}h_{ij}
\in
\mathbb{R}^{256}
\]

\(\alpha_{ij}\) là learned aggregation weight.

Không diễn giải chúng như clinical probability.

## 8.2. A1 — mean aggregation

A1 thay learned pooling bằng:

\[
z_{kb,i}
=
\frac{1}{65}
\sum_{j=1}^{65}
h_{ij}
\]

Các thành phần khác giữ nguyên.

A1 trả lời câu hỏi:

> Learned aggregation có mang lại lợi ích so với mean pooling đơn giản hay không?

---

# 9. Classification fusion

Knowledge branch bổ sung cho image branch.

Không dùng proposition bottleneck.

Representation cuối:

\[
z_i
=
[z_{img,i};z_{kb,i}]
\]

sau đó:

\[
z_i
\rightarrow
Linear
\rightarrow
\ell_i\in\mathbb{R}^{8}
\]

Classifier head được khóa là một lớp tuyến tính:

\[
\ell_i=W_cz_i+b_c
\]

B0, B1, B2, B3 và A1 phải dùng classifier linear tương ứng với input dimension của từng nhánh. Code có thể hỗ trợ MLP qua config cho experiment ngoài core, nhưng không main config nào dùng MLP.

---

# 10. Knowledge-guided Contrastive Learning

## 10.1. Fixed class prototypes

Class prototype được giữ trong **frozen biomedical text space**.

Với class \(c\), gọi \(K_c\) là tập proposition được map với class đó trong KB.

Chuẩn hóa từng frozen text embedding:

\[
\bar t_j
=
\frac{t_j}{\|t_j\|}
\]

Sau đó:

\[
g_c
=
Normalize
\left(
\frac{1}{|K_c|}
\sum_{j\in K_c}
\bar t_j
\right)
\in
\mathbb{R}^{768}
\]

Prototype được cố định và không nhận gradient.

Shared propositions có thể xuất hiện trong nhiều class prototype theo mapping của KB.

## 10.2. Patient projection

Contrastive alignment áp dụng trên **knowledge-derived patient representation**:

\[
u_i
=
\phi(z_{kb,i})
\in
\mathbb{R}^{768}
\]

sau đó normalize:

\[
\bar u_i
=
\frac{u_i}{\|u_i\|}
\]

Cách này giúp auxiliary contrastive objective tác động trực tiếp lên nhánh knowledge.

## 10.3. Contrastive loss

\[
\mathcal{L}_{con}^{(i)}
=
-\log
\frac{
\exp(
\cos(\bar u_i,g_{y_i})/\tau
)
}{
\sum_{c\in\mathcal Y}
\exp(
\cos(\bar u_i,g_c)/\tau
)
}
\]

Total loss:

\[
\boxed{
\mathcal L
=
\mathcal L_{CE}
+
\lambda\mathcal L_{con}
}
\]

B2:

\[
\lambda=0
\]

B3:

\[
\lambda>0
\]

Không dùng proposition-level contrastive loss.

---

# 11. Metadata

Metadata **không thuộc main B2/B3 configuration** vì B1 không cải thiện B0.

Codebase vẫn nên hỗ trợ:

```text
use_metadata = true | false
```

Nếu một experiment sau này bật metadata, chỉ được dùng:

- age
- gender
- coarse region
- view

Khi đó P066–P068 mới có thể được bật.

Metadata experiment là optional, không phải core experiment nếu chưa có kết quả thực nghiệm hỗ trợ.

---

# 12. Training protocol

Giữ protocol tương thích với B0.

## 12.1. Image preprocessing

Input size:

\[
224\times224
\]

Train-only augmentation:

```text
RandomAffine(
    degrees = 7,
    translate = (0.03, 0.03),
    scale = (0.95, 1.05)
)
```

Validation/test:

- không augmentation
- deterministic

## 12.2. Class imbalance

Dùng sqrt-inverse-frequency weighted Cross-Entropy đã tính từ training split.

Không dùng:

- weighted sampler
- oversampling

## 12.3. Optimizer

```text
Adam
lr = 1e-4
weight_decay = 1e-4
```

## 12.4. Learning-rate schedule

```text
max_epochs = 100
StepLR(step_size=25, gamma=0.1)
```

Checkpoint được chọn theo:

\[
\boxed{\text{Validation Macro-F1}}
\]

Có thể dùng early stopping chỉ để tiết kiệm compute.

Patience mặc định:

```text
15 epochs
```

---

# 13. Core experiments và ablation

Giữ số experiment vừa đủ.

| Run | Image | Metadata | Proposition knowledge | Aggregation | Contrastive | Mục đích |
|---|:---:|:---:|:---:|:---:|:---:|---|
| B0 | ✓ | – | – | – | – | Image-only baseline |
| B1 | ✓ | ✓ | – | – | – | Metadata baseline, đã hoàn thành |
| B2 | ✓ | – | ✓ | learned | – | Đo đóng góp của knowledge |
| B3 | ✓ | – | ✓ | learned | ✓ | Đo đóng góp thêm của Contrastive Learning |
| A1 | ✓ | – | ✓ | mean | – | Ablation cho aggregation |

## 13.1. B2

Main knowledge model:

\[
\mathcal L=\mathcal L_{CE}
\]

Câu hỏi:

> Proposition-guided medical knowledge có cải thiện 8-class classification so với B0 hay không?

## 13.2. B3

Giữ nguyên B2 và thêm:

\[
\lambda\mathcal L_{con}
\]

Câu hỏi:

> Class-level knowledge alignment có bổ sung lợi ích ngoài B2 hay không?

## 13.3. A1

Giữ nguyên B2 nhưng dùng mean aggregation.

Câu hỏi:

> Learned proposition aggregation có tốt hơn uniform averaging hay không?

## 13.4. Không phải core experiments

Không thêm nếu chưa có lý do thực nghiệm rõ ràng:

- matched / irrelevant knowledge control
- random propositions
- shuffled propositions
- random vectors
- learned free queries
- multiple text encoder comparison
- fine-tuned vs frozen text encoder
- shared-only vs characteristic-only
- positional-embedding ablation
- hyperparameter grid lớn

---

# 14. XAI và lesion grounding

## 14.1. XAI chính

Dùng:

> **Grad-CAM**

So sánh tối thiểu:

- B0 image-only
- final proposed model

## 14.2. Ground truth

Dùng lesion annotations của BTXRD trên các tumour images.

Normal không có lesion target nên không đưa vào lesion-grounding metrics.

## 14.3. Quantitative grounding

### Pointing Game

\[
PG
=
\frac{
\#\text{cases có peak activation nằm trong lesion}
}{
\#\text{annotated tumour cases}
}
\]

### Energy Inside Lesion

\[
E_{lesion}
=
\frac{
\sum_{(x,y)\in R}H(x,y)
}{
\sum_{x,y}H(x,y)
}
\]

## 14.4. Cross-attention visualization

Attention map từ proposition-guided cross-attention có thể dùng làm supplementary visualization.

Chỉ diễn giải là:

> nơi semantic query đang tập trung.

Không dùng attention map để claim:

- causal explanation
- proposition presence
- proposition-level diagnosis

SHAP là optional.

---

# 15. Phạm vi claim

Nếu experiment hỗ trợ, có thể claim:

- medical knowledge được tổ chức thành các radiographic propositions fine-grained;
- propositions được tích hợp như semantic queries vào image representation learning;
- proposition-guided interaction cải thiện hoặc hỗ trợ fine-grained 8-class classification;
- class-level contrastive alignment giúp representation nếu B3 tốt hơn B2;
- final model tập trung tốt hơn vào lesion region nếu annotation-based grounding metrics hỗ trợ.

Không claim:

- proposition weight là clinical probability;
- model dự đoán proposition có xuất hiện hay không;
- model là concept bottleneck;
- attention là causal explanation;
- model reasoning giống radiologist;
- clinical validity ngoài phạm vi dataset và experiment.

---

# 16. Các quyết định đã khóa

| Thành phần | Quyết định |
|---|---|
| Dataset | BTXRD |
| Task chính | 8-class classification |
| Số ảnh sử dụng | 3,586 |
| Classes | Normal + 7 tumour classes cụ thể |
| Class bị loại | other bt, other mt |
| Split | fixed stratified 2510 / 538 / 538 |
| Metric chính | Macro-F1 |
| Image backbone | DenseNet-169, ImageNet-pretrained |
| Input ảnh | 224×224 |
| KB | 8-class KB v1.0 |
| Tổng proposition | 68 |
| Main visual queries | P001–P065 |
| Metadata propositions | P066–P068, tắt trong B2/B3 chính |
| Text encoder | frozen MedCPT query encoder |
| Text embeddings | cache offline |
| Knowledge dimension | 256 |
| Knowledge interaction | proposition-guided cross-attention |
| Main aggregation | learned attention pooling |
| Aggregation ablation | mean pooling |
| Classification fusion | global image feature + knowledge feature |
| Classifier head | Linear cho B0/B1/B2/B3/A1 |
| Main loss | sqrt-inverse weighted Cross-Entropy |
| Auxiliary loss | class-level knowledge contrastive loss |
| Contrastive prototype | fixed prototype trong frozen text space |
| Contrastive patient feature | projected \(z_{kb}\) |
| Metadata trong B2/B3 chính | Không |
| Main XAI | Grad-CAM |
| Grounding metrics | Pointing Game + Energy Inside Lesion |
| Core runs | B0, B1, B2, B3, A1 |
| Matched control | Không bắt buộc |
| Weak labels | Không |
| Proposition supervision | Không |
| Proposition bottleneck | Không |

---

# 17. Câu chuyện paper tối giản

1. **Bài toán:** Fine-grained classification giữa normal bone và 7 nhóm bone tumour trên radiograph là khó, đặc biệt với class imbalance và morphology overlap.
2. **Biểu diễn tri thức:** Tri thức X-quang được tổ chức thành các fine-grained radiographic propositions có provenance và class mapping.
3. **Phương pháp:** Frozen proposition embeddings đóng vai trò semantic queries để tương tác với DenseNet visual tokens bằng cross-attention.
4. **Aggregation:** Các proposition-conditioned patient features được tổng hợp thành knowledge representation và fusion với global image feature.
5. **Learning:** Weighted Cross-Entropy thực hiện classification; class-level Contrastive Learning là auxiliary objective tùy chọn.
6. **Đánh giá:** So sánh B0, B1, B2, B3 và A1; đánh giá classification và lesion grounding dựa trên annotation.
7. **Giới hạn:** Không weak label, không proposition presence prediction, không concept bottleneck, không mở rộng quá nhiều control/ablation.

---

# 18. Nguyên tắc implementation cuối cùng

Research code phải theo hướng **config-driven**.

Các module phải bật/tắt được mà không copy model code.

Ví dụ:

```yaml
model:
  use_knowledge: true
  use_metadata: false
  use_contrastive: false
  aggregation: learned
  use_positional_embedding: false

knowledge:
  exclude_ids:
    - P066
    - P067
    - P068
  text_encoder_frozen: true

loss:
  contrastive_weight: 0.0
```

B2/B3/A1 chỉ khác nhau bằng config.

Colab notebook chỉ là execution wrapper, không chứa logic kiến trúc chính.
