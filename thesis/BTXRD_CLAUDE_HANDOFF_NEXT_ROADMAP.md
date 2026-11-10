# BTXRD — HANDOFF CHO CLAUDE
## Trạng thái nghiên cứu hiện tại, Next Research Roadmap và hướng code tiếp theo

> **Mục đích của file này:** Đây là handoff đầy đủ để Claude nắm đúng trạng thái mới nhất của dự án trước khi hỗ trợ viết prompt cho coding agent.
>
> **Không tự mở rộng scope ngoài tài liệu này.** Nếu có điểm nào chưa khóa, cần ghi rõ là chưa khóa thay vì tự quyết định.

---

# 1. Bối cảnh nghiên cứu

Bài toán hiện tại:

> **Phân loại X-quang xương BTXRD thành 8 lớp**, có tích hợp tri thức y khoa được tổ chức dưới dạng proposition.

8 lớp:

1. `normal`
2. `osteochondroma`
3. `multiple_osteochondromas`
4. `simple_bone_cyst`
5. `giant_cell_tumor`
6. `osteofibroma`
7. `synovial_osteochondroma`
8. `osteosarcoma`

Dataset sau preprocessing:

- Train: **2510**
- Validation: **538**
- Test: **538**
- Tổng: **3586 ảnh**

Metric chính:

`Macro-F1`

Các metric phụ:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- per-class Precision / Recall / F1
- Macro AUROC
- confusion matrix

---

# 2. Knowledge Base hiện tại

KB đã khóa ở phiên bản 8-class.

Tổng số proposition: **68**

Trong đó:

- P001–P065: visual/radiographic propositions
- P066–P068: metadata propositions về tuổi/giới

Main visual model hiện tại chỉ dùng:

`P001–P065`

Không dùng P066–P068 trong main image-only / knowledge model.

Một proposition có thể thuộc nhiều class.

KB có mapping:

`proposition → one or more classes`

và mỗi class có một tập `K_c` gồm các proposition liên quan tới class đó.

Tri thức được dùng như:

> **semantic query**

Không dùng:

- weak label
- PRESENT / ABSENT supervision
- proposition bottleneck
- concept bottleneck
- proposition-level BCE

---

# 3. Các experiment đã chạy

## 3.1. B0 — DenseNet-169 Image-only

Kết quả:

- Accuracy ≈ **74.54%**
- Balanced Accuracy ≈ **60.35%**
- Macro-F1 ≈ **61.78%**
- Macro AUROC ≈ **90.78%**

Đây là baseline DenseNet chính.

## 3.2. B1 — DenseNet + Metadata

Metadata:

- age
- gender
- coarse anatomical region
- view

Kết quả:

- Accuracy ≈ **75.09%**
- Balanced Accuracy ≈ **54.78%**
- Macro-F1 ≈ **58.49%**
- Macro AUROC ≈ **88.02%**

Kết luận:

> Naive metadata fusion không giúp.

Metadata không thuộc main model hiện tại.

## 3.3. B2 — DenseNet + Proposition Knowledge

Kiến trúc:

```text
DenseNet feature map
    ↓
visual tokens
    ↓
65 proposition queries
    ↓
cross-attention
    ↓
65 proposition-conditioned patient features
    ↓
learned global aggregation
    ↓
z_kb
```

Sau đó:

```text
[z_img ; z_kb]
    ↓
Linear classifier
    ↓
8 logits
```

Loss:

`L = L_CE`

Kết quả:

- Accuracy ≈ **76.77%**
- Balanced Accuracy ≈ **57.44%**
- Macro-F1 ≈ **61.33%**
- Macro AUROC ≈ **89.65%**

Kết luận:

> Knowledge branch dạng global aggregation chưa cải thiện rõ so với B0.

## 3.4. B3 — B2 + Class-level Contrastive Learning

Thêm:

`L = L_CE + lambda * L_con`

với:

- `lambda = 0.1`
- `tau = 0.07`

Contrastive alignment:

```text
z_kb
  ↓
phi
  ↓
frozen MedCPT text space
  ↓
align với 8 fixed class prototypes
```

Kết quả:

- Accuracy ≈ **76.77%**
- Balanced Accuracy ≈ **58.66%**
- Macro-F1 ≈ **62.64%**
- Macro AUROC ≈ **91.31%**

Kết luận:

> Contrastive giúp nhẹ, nhưng mức tăng chưa đủ lớn.

---

# 4. S0 — SKELEX Image-only

Đã chạy thử SKELEX như một MSK-specific backbone.

Kết quả:

- Accuracy ≈ **81.97%**
- Balanced Accuracy ≈ **68.15%**
- Macro-F1 ≈ **69.91%**
- Macro AUROC ≈ **91.40%**

Best validation Macro-F1:

- **68.56%**
- best epoch: **27**

SKELEX cải thiện rất mạnh so với DenseNet B0:

`61.78% → 69.91% Macro-F1`

Kết luận quan trọng:

> Visual representation/backbone là một bottleneck lớn trong pipeline hiện tại.

Tuy nhiên:

> **Không chuyển hẳn sang SKELEX ngay.**

Mục tiêu mới là xây phương pháp **backbone-agnostic**, để cùng một knowledge architecture có thể chạy với:

```text
encoder = densenet169
```

hoặc:

```text
encoder = skelex
```

---

# 5. Tư duy nghiên cứu mới

Không muốn câu chuyện trở thành:

> “SKELEX mạnh nên model mạnh.”

Mục tiêu là:

> **Phương pháp knowledge-guided của chúng ta phải có khả năng cải thiện trên nhiều encoder khác nhau.**

Do đó cần tách hai yếu tố:

1. sức mạnh của image encoder;
2. contribution của phương pháp proposition-guided.

So sánh đúng phải là:

`R1-D vs B0`

và:

`R1-S vs S0`

Không dùng trực tiếp:

`R1-D vs R1-S`

để chứng minh contribution knowledge.

---

# 6. Hạn chế kiến trúc B2/B3 hiện tại

Hiện tại 65 proposition được xử lý:

```text
65 propositions
    ↓
cross-attention
    ↓
65 proposition-conditioned features
    ↓
GLOBAL aggregation
    ↓
1 z_kb
```

Vấn đề:

> Mapping `proposition → class` trong KB gần như bị bỏ qua trong đường classification chính.

Tức là:

- KB đã biết proposition nào liên quan class nào;
- nhưng model lại gộp tất cả proposition thành một representation duy nhất;
- class structure của KB không được sử dụng trực tiếp để tạo final logits.

Đây được xem là hạn chế kiến trúc quan trọng nhất hiện nay.

---

# 7. Ý tưởng mới: Class-Conditioned Proposition Decoder

Đây là hướng kiến trúc ưu tiên tiếp theo.

Tên tạm:

> **Class-Conditioned Proposition Decoder**

Tinh thần có thể học từ Label Decoder của RAD, nhưng **không copy nguyên architecture của RAD**.

Điểm khác:

- class query được xây từ proposition prototype của chính KB;
- decoder hoạt động trên proposition-conditioned patient features;
- không dùng disease-name query đơn thuần;
- không dùng proposition-level supervision.

---

# 8. Kiến trúc R1 đề xuất

## 8.1. Image encoder

Image encoder phải là module thay được bằng config.

Ví dụ:

```yaml
model:
  backbone: densenet169
```

hoặc:

```yaml
model:
  backbone: skelex
```

## 8.2. Interface encoder chung

Mọi image encoder nên trả về cùng một interface:

```python
EncoderOutput(
    visual_tokens,
    global_feature,
    spatial_shape
)
```

Trong đó:

```text
visual_tokens:  [B, L, D_encoder]
global_feature: [B, D_global]
spatial_shape:  optional
```

---

# 9. DenseNet-169 encoder

DenseNet:

```text
X-ray
 ↓
DenseNet-169
 ↓
feature map 7×7×1664
```

Visual tokens:

`49 × 1664`

Global feature:

`GAP(feature map) -> 1664`

Sau đó:

```text
visual tokens
    ↓
Linear projection
    ↓
49 × 256
```

---

# 10. SKELEX encoder

SKELEX là ViT/MAE-style encoder.

Input 224×224, patch size 16:

`14 × 14 = 196 patch tokens`

Dùng:

```text
patch tokens
```

làm visual memory.

Không dùng CLS token trong cross-attention visual memory.

Global image representation lấy từ:

```text
CLS token
```

hoặc representation global chuẩn của SKELEX.

Output:

```text
visual_tokens:
    [B, 196, hidden_dim]

global_feature:
    [B, hidden_dim]
```

Không hard-code hidden dimension.

Đọc từ:

```python
model.config.hidden_size
```

Sau đó:

```text
visual tokens
    ↓
Linear projection
    ↓
196 × 256
```

---

# 11. Proposition cross-attention

Giữ tinh thần hiện tại.

Frozen MedCPT embeddings:

`t_j ∈ R^768`

Query projection:

`q_j = W_q t_j ∈ R^256`

Visual tokens được chiếu về 256 chiều.

Sau đó:

`H_i = CrossAttention(Q, V_i, V_i)`

Output:

`H_i ∈ R^(65×256)`

Mỗi `h_ij` là:

> patient representation conditioned on proposition j.

Không diễn giải thành probability PRESENT/ABSENT.

---

# 12. Class-Conditioned Proposition Decoder

Với mỗi class `c`, KB có tập proposition:

`K_c`

Ví dụ:

```text
K_normal
K_osteochondroma
K_multiple_osteochondromas
...
K_osteosarcoma
```

Dùng các proposition features thuộc class:

`H_i,Kc`

để tạo class-specific patient representation:

`r_ic`

## 12.1. Class prototype query

Prototype class:

`g_c = Normalize(mean_j∈Kc Normalize(t_j))`

Prototype nằm trong frozen MedCPT text space.

Có thể chiếu prototype về decoder dimension:

`g_hat_c = W_g g_c ∈ R^256`

Sau đó dùng prototype làm query:

`r_ic = Attn(g_hat_c, H_i,Kc, H_i,Kc)`

Output:

`r_ic ∈ R^256`

Mỗi class có một class-conditioned patient representation.

---

# 13. Knowledge logits

Dùng một shared scorer:

`s_ic = f(r_ic)`

để tạo một scalar logit cho class `c`.

Sau 8 class:

`l_kb = [s_i1, ..., s_i8]`

---

# 14. Image logits

Image branch vẫn phải có khả năng classify độc lập.

Global feature:

`z_img`

đi qua image classifier:

`l_img ∈ R^8`

---

# 15. Residual logit fusion

Final logits:

`l_final = l_img + beta * l_kb`

Trong đó:

- `beta` là scalar trainable hoặc scalar parameter có initialization nhỏ;
- knowledge branch đóng vai trò **residual correction**;
- image classifier vẫn là nền chính;
- knowledge không trở thành bottleneck.

Đây là điểm thiết kế quan trọng.

---

# 16. Tại sao residual logits?

Ưu điểm:

1. không làm mất khả năng image-only của backbone;
2. knowledge chỉ hiệu chỉnh decision;
3. dễ so với image baseline;
4. nếu knowledge chưa tốt, model vẫn có thể dựa nhiều vào image logits;
5. phù hợp để chạy trên cả DenseNet và SKELEX.

---

# 17. R1 — hai run tiếp theo

## R1-D

```text
DenseNet-169
+
Class-Conditioned Proposition Decoder
```

Không Contrastive Learning.

Không metadata.

Không multi-scale.

Không sửa KB.

So sánh:

`R1-D vs B0`

Mục tiêu:

> Kiểm tra architecture knowledge mới có cải thiện DenseNet baseline hay không.

## R1-S

```text
SKELEX
+
Class-Conditioned Proposition Decoder
```

Architecture knowledge phải **y hệt R1-D**.

Chỉ đổi image encoder.

Không Contrastive Learning.

So sánh:

`R1-S vs S0`

Mục tiêu:

> Kiểm tra architecture knowledge mới có cải thiện một MSK-specific foundation model mạnh hay không.

---

# 18. Vì sao chưa bật Contrastive Learning ở R1?

B3 cho thấy contrastive chỉ mang lại gain nhỏ.

Nếu R1 vừa đổi decoder vừa bật contrastive thì không xác định được contribution của decoder mới.

Vì vậy R1 dùng:

`L = L_CE`

Nếu R1 thành công, bước sau mới thử:

`R2 = R1 + L_con`

---

# 19. Training recipe theo encoder

Không ép DenseNet và SKELEX dùng cùng optimizer.

Phép so công bằng là **trong từng backbone**.

## 19.1. DenseNet recipe

Giữ recipe của B0/B2/B3:

```text
Adam
lr = 1e-4
weight_decay = 1e-4
StepLR(step=25, gamma=0.1)
100 epochs
sqrt-inverse weighted CE
batch size = 32
```

Checkpoint:

`best validation Macro-F1`

## 19.2. SKELEX recipe

Giữ recipe đã cho kết quả S0 mạnh:

```text
AdamW
base lr = 5e-5
weight_decay = 0.05
layer-wise LR decay = 0.75
warmup = 5 epochs
cosine schedule
50 epochs
physical batch = 16
gradient accumulation = 2
effective batch = 32
weighted CE
per-image min-max normalization
```

Checkpoint:

`best validation Macro-F1`

---

# 20. Codebase phải backbone-agnostic

Không tạo hai model riêng:

```text
BTXRDModelDenseNet
BTXRDModelSKELEX
```

Thay vào đó dùng factory:

```python
build_image_encoder(cfg)
```

Ví dụ:

```python
if cfg.model.backbone == "densenet169":
    return DenseNet169Encoder(...)

if cfg.model.backbone == "skelex":
    return SKELEXEncoder(...)
```

Từ `EncoderOutput` trở xuống, toàn bộ knowledge module phải dùng chung.

---

# 21. Config mong muốn

Ví dụ:

```yaml
model:
  backbone: densenet169
  use_knowledge: true
  knowledge_decoder: class_conditioned
  residual_knowledge_logits: true
  use_contrastive: false
```

R1-S chỉ thay:

```yaml
model:
  backbone: skelex
```

Không copy model code.

---

# 22. Các config nên có

Ví dụ:

```text
configs/
├── b0_image.yaml
├── b2_knowledge.yaml
├── b3_contrastive.yaml
├── s0_skelex_image.yaml
├── r1_densenet_class_decoder.yaml
└── r1_skelex_class_decoder.yaml
```

Có thể refactor naming nếu coding agent thấy cần, nhưng semantic phải giữ như trên.

---

# 23. Tiêu chí thành công của R1

## Trường hợp lý tưởng nhất

`R1-D > B0`

và:

`R1-S > S0`

Ví dụ:

```text
B0      61.8
R1-D    67.0

S0      69.9
R1-S    74.0
```

Có thể claim:

> proposed knowledge-guided method improves both a conventional ImageNet-pretrained CNN and an MSK-specific foundation model.

## Trường hợp 2

DenseNet tăng mạnh nhưng SKELEX gần như không tăng:

```text
B0      61.8
R1-D    67+

S0      69.9
R1-S    ~70
```

Diễn giải:

> proposition knowledge giúp backbone yếu nhiều hơn; SKELEX đã encode sẵn nhiều domain knowledge.

## Trường hợp 3

DenseNet không tăng nhưng SKELEX tăng mạnh:

```text
R1-D ≈ B0
R1-S >> S0
```

Diễn giải:

> knowledge reasoning chỉ phát huy khi visual representation đủ mạnh.

## Trường hợp 4

Cả hai đều không tăng:

> Class-Conditioned Proposition Decoder không đủ.

Lúc đó mới audit KB / attention / differential propositions.

---

# 24. Chiến lược paper về backbone

Không được selective reporting.

Nếu SKELEX là strong relevant baseline và đã được thử, không nên bỏ chỉ vì proposed model không vượt nó.

## Hướng A — backbone-agnostic method

Baseline:

```text
DenseNet-169
EfficientNet
BioMedCLIP
SKELEX
...
```

Proposed:

```text
DenseNet + ours
SKELEX + ours
```

Nếu gain trên cả hai:

> rất mạnh.

Đây là hướng ưu tiên.

## Hướng B — conventional backbone setting

Nếu phương pháp tăng tốt trên DenseNet nhưng không tăng rõ trên SKELEX:

- DenseNet có thể vẫn là backbone chính để phân tích contribution;
- SKELEX được giữ như strong external/domain-specific baseline;
- không claim vượt mọi foundation model.

Contribution là gain của proposed method trong cùng backbone.

---

# 25. NEXT RESEARCH ROADMAP — đầy đủ

## Ưu tiên 1 — SKELEX image-only baseline

Đã hoàn thành.

Kết quả:

`Macro-F1 = 69.91%`

Cho thấy backbone là bottleneck đáng kể.

## Ưu tiên 2 — Class-Conditioned Proposition Decoder

Đây là bước hiện tại.

Mục tiêu:

> dùng trực tiếp mapping proposition→class trong final decision.

Hai run:

```text
R1-D
R1-S
```

## Ưu tiên 3 — SKELEX + Class-Conditioned Proposition Decoder

Chính là R1-S.

Nếu R1-S cải thiện rõ S0:

> đây là ứng viên final model mạnh nhất.

---

# 26. Chưa điều chỉnh KB ngay

Không sửa 68 propositions chỉ vì B2/B3 chưa tăng mạnh.

Lý do:

B2/B3 chưa dùng trực tiếp proposition→class mapping trong classification path.

Trước khi sửa KB cần audit:

1. cosine similarity giữa 8 class prototypes;
2. overlap giữa `K_c`;
3. shared / characteristic ratio từng class;
4. learned proposition weights theo true class;
5. cross-attention trên ca đúng/sai;
6. confusion pairs.

---

# 27. Nếu cần chỉnh KB — ưu tiên differential propositions

Không tăng số lượng câu một cách chung chung.

Chỉ bổ sung proposition giúp phân biệt các cặp hay nhầm:

```text
normal ↔ subtle benign lesion

osteochondroma
↔
multiple osteochondromas

osteochondroma
↔
synovial osteochondroma

simple bone cyst
↔
giant cell tumor

giant cell tumor
↔
osteosarcoma
```

Mục tiêu:

> differential diagnosis knowledge.

---

# 28. Multi-scale visual memory — phương án sau R1

Nếu vẫn dùng DenseNet và cần cải thiện thêm:

Hiện DenseNet chỉ dùng feature map:

`7×7`

Có thể quá coarse cho:

- margin
- transition zone
- cortical thinning
- matrix
- subtle morphology

Có thể thêm:

```text
mid-level 14×14
+
deep 7×7
```

project về cùng dimension rồi tạo multi-scale visual memory.

Nhưng:

> **Không làm trước R1.**

---

# 29. Lesion-guided supervision — phương án dự phòng

BTXRD có lesion annotations.

Có thể dùng:

`L = L_CE + lambda * L_con + gamma * L_loc`

Ví dụ:

`L_loc = 1 - attention_energy_inside_lesion`

Có thể giúp:

- localization;
- giảm shortcut;
- classification;
- XAI.

Nhưng:

- mở rộng scope;
- annotation không còn chỉ dùng để evaluate;
- phát sinh thêm contribution cần chứng minh.

Do đó:

> chỉ dùng nếu R1/R2 và KB audit vẫn chưa đủ.

---

# 30. Contrastive Learning — bước R2, không phải R1

Nếu Class-Conditioned Decoder thành công:

`R2 = R1 + L_con`

Lúc đó mới kiểm contrastive còn bổ sung gì trên decoder mới hay không.

Không bật trong R1.

---

# 31. Những việc chưa nên làm

Không ưu tiên:

- grid search lớn cho lambda/tau;
- matched knowledge control;
- random propositions;
- shuffled propositions;
- random vectors;
- free learned queries;
- multiple text encoder comparison;
- fine-tune MedCPT;
- sửa KB hàng loạt;
- thêm metadata trở lại;
- positional encoding ablation;
- nhiều auxiliary loss;
- multi-scale trước khi R1 có kết quả;
- lesion supervision trước khi R1 có kết quả.

Nguyên tắc:

> Mỗi module mới phải giải quyết một hạn chế cụ thể đã quan sát được.

---

# 32. Roadmap thực nghiệm tổng thể

```text
B0 DenseNet image-only
        ↓
B2 DenseNet + proposition knowledge
        ↓
B3 DenseNet + knowledge + contrastive
        ↓
S0 SKELEX image-only
        ↓
[HIỆN TẠI]
        ↓
Refactor backbone-agnostic code
        ↓
Class-Conditioned Proposition Decoder
        ↓
R1-D = DenseNet + decoder mới
        ↓
R1-S = SKELEX + decoder mới
        ↓
đánh giá gain trong từng backbone
        ↓
nếu tốt:
    R2 = R1 + contrastive
        ↓
nếu cần:
    audit KB
        ↓
    differential propositions
        ↓
nếu vẫn cần:
    multi-scale visual memory
        ↓
nếu vẫn cần:
    lesion-guided supervision
```

---

# 33. Hướng final model hiện được ưu tiên

Ứng viên final model về mặt kiến trúc:

```text
Configurable Image Encoder
(DenseNet-169 hoặc SKELEX)
        ↓
visual tokens
        ↓
65 proposition semantic queries
        ↓
proposition-guided cross-attention
        ↓
65 proposition-conditioned patient features
        ↓
8 Class-Conditioned Proposition Decoders
        ↓
8 knowledge logits
        +
8 image logits
        ↓
Residual Logit Fusion
        ↓
8-class prediction
```

Điểm cốt lõi:

> **Kiến trúc knowledge không phụ thuộc backbone.**

---

# 34. Yêu cầu dành cho coding agent sau này

Claude sau khi đọc file này sẽ viết prompt cho coding agent.

Prompt cho coding agent cần yêu cầu:

1. đọc code hiện tại trước khi sửa;
2. không phá B0/B2/B3 hiện có;
3. refactor image encoder theo interface chung;
4. thêm `SKELEXEncoder`;
5. giữ `DenseNet169Encoder`;
6. tạo encoder factory;
7. implement Class-Conditioned Proposition Decoder;
8. implement residual knowledge logits;
9. config-driven hoàn toàn;
10. R1-D và R1-S chỉ khác `backbone`;
11. training recipe tự chọn theo backbone;
12. thêm unit tests;
13. thêm smoke tests;
14. không thay KB;
15. không bật contrastive trong R1;
16. không thêm metadata;
17. không thêm multi-scale;
18. không thêm lesion supervision;
19. đảm bảo checkpoint/predict/resume vẫn hoạt động;
20. Colab notebook chỉ là runner.

---

# 35. Các invariants phải giữ

## Dataset

```text
3586 ảnh
2510 / 538 / 538
8 classes
```

## Knowledge

```text
68 propositions tổng
P001–P065 dùng main visual model
P066–P068 excluded
MedCPT frozen
```

## Main metric

`Macro-F1`

## Classification

`8 logits`

## R1 loss

`L = L_CE`

Không contrastive trong R1.

---

# 36. Không được thay đổi câu chuyện nghiên cứu

Contribution chính cần hướng tới:

> **Tổ chức tri thức y khoa dưới dạng fine-grained propositions và khai thác cấu trúc proposition→class thông qua class-conditioned decoding để cải thiện fine-grained bone tumour classification.**

Không biến câu chuyện thành:

> “SKELEX tốt hơn DenseNet.”

SKELEX chỉ là một backbone mạnh để kiểm tra tính tổng quát của phương pháp.

---

# 37. Việc Claude cần làm tiếp theo

Sau khi đọc file này:

1. Không viết code trực tiếp ngay.
2. Kiểm tra xem có hiểu đúng architecture hay không.
3. Nếu hiểu đúng, viết **một prompt đầy đủ cho coding agent**.
4. Prompt phải yêu cầu coding agent:
   - review repo hiện tại;
   - đưa ra implementation plan ngắn;
   - sau đó refactor và implement;
   - chạy tests;
   - báo rõ file nào thay đổi;
   - không mở rộng ngoài scope đã khóa.

---

# 38. Tóm tắt cực ngắn

Hiện tại:

```text
B0 DenseNet image-only: Macro-F1 ≈ 61.78
B3 DenseNet + knowledge + contrastive: ≈ 62.64
S0 SKELEX image-only: ≈ 69.91
```

Không chuyển hẳn sang SKELEX.

Tiếp theo:

```text
R1-D = DenseNet + Class-Conditioned Proposition Decoder

R1-S = SKELEX + cùng Class-Conditioned Proposition Decoder
```

So sánh:

```text
R1-D vs B0
R1-S vs S0
```

Nếu cả hai tăng:

> phương pháp knowledge-guided có tính backbone-agnostic và câu chuyện paper rất mạnh.

Đây là hướng hiện tại cần code.
