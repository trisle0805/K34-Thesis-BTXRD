# PROMPT CHO CODING AGENT — R1: Backbone-agnostic encoder + Class-Conditioned Proposition Decoder

> Repo: `trisle0805/K34-Thesis-BTXRD`, nhánh `main`, commit gốc **`80db443`**.
> Đây là giai đoạn **chạy thử để xem hướng nào có triển vọng**. Ưu tiên code chạy đúng, gọn, config-driven. Không thêm gì ngoài phạm vi dưới đây.

Tài liệu tham khảo (tôi sẽ thêm vào repo trước khi bạn bắt đầu):

- `thesis/BTXRD_CLAUDE_HANDOFF_NEXT_ROADMAP.md`: roadmap nghiên cứu, đọc §6–§22 và §34–§35.
- `reference/07_BTXRD_S0_SKELEX_IMAGE_ONLY.ipynb`: notebook S0 đã chạy (test Macro-F1 ≈ 69.91%, best val Macro-F1 68.56% ở epoch 27, đếm từ 1). Mục 3 và 4 bên dưới phải tái tạo đúng recipe của notebook này.

---

## 0. Cách làm việc

1. **Đọc code hiện tại trước**, đặc biệt các file `models/model.py`, `models/image_encoder.py`, `models/cross_attention.py`, `config.py`, `training/trainer.py`, `training/predict.py`, `data/transforms.py`, `data/dataset.py`.
2. **Gửi implementation plan ngắn** (tối đa khoảng 1 trang): file nào thêm, file nào sửa, config schema mới. Chờ tôi xác nhận rồi mới code.
3. Code, chạy `pytest -q` trên clone sạch, commit + push.
4. Báo lại: commit hash, danh sách file thay đổi, kết quả pytest, và những chỗ bạn phải tự quyết.

### Không được làm

- Không phá B0/B1/B2/B3/A1. Các config hiện có phải cho ra **cùng model, cùng `state_dict` keys/shapes, cùng optimizer/scheduler** như commit `80db443`. Checkpoint B2/B3 đã train bằng commit này phải còn load và predict được.
- Không sửa KB, không đổi text encoder, không bật contrastive trong R1.
- Không thêm metadata, multi-scale, lesion supervision.
- Không thêm thư viện ngoài `transformers` (đã có trong extra `[text]`).
- Không hard-code đường dẫn.
- Đặt tên trung lập, không dùng `presence`, `prob_proposition`, `evidence`.

---

## 1. Interface encoder chung

```python
@dataclass
class EncoderOutput:
    visual_tokens: Tensor      # [B, L, D_tok]
    global_feature: Tensor     # [B, D_glob]
    spatial_shape: tuple[int, int] | None
```

Mỗi encoder công bố `token_dim` và `global_dim`. Thêm factory `build_image_encoder(cfg, pretrained)`. Từ `EncoderOutput` trở xuống, mọi module knowledge dùng chung, không có nhánh `if backbone == ...` trong model.

### 1.1. `DenseNet169Encoder` (giữ nguyên hành vi)

- `f = relu(features(x))` → `[B, 1664, 7, 7]`.
- `visual_tokens = f.flatten(2).transpose(1, 2)` → `[B, 49, 1664]`.
- `global_feature = f.mean((2, 3))` → `[B, 1664]`.
- `spatial_shape = (7, 7)`.
- **Giữ nguyên tên tham số** (`image_encoder.features.*`). Phần chiếu token hiện nằm trong `PropositionCrossAttention` và nhận feature map. Hãy sửa để nó nhận `visual_tokens [B, L, D_tok]`, nhưng **giữ nguyên tên `cross_attention.token_proj.*`** và các key khác, để checkpoint cũ còn load được.
- `use_positional_embedding` (sincos): nếu bật thì dùng `spatial_shape`. Mặc định vẫn tắt.

### 1.2. `SKELEXEncoder` (mới)

- `transformers.ViTModel.from_pretrained(cfg.model.skelex.model_name, add_pooling_layer=False)`, mặc định `"skhoha/SKELEX"`. Import `transformers` lazy, bên trong class.
- `h = model(pixel_values=x).last_hidden_state`: `[B, 1 + N, H]`, đã qua layernorm cuối.
- `global_feature = h[:, 0]` (CLS). Đây đúng là input của classifier trong `ViTForImageClassification` mà S0 đã dùng.
- `visual_tokens = h[:, 1:]` → `[B, N, H]`. **Không đưa CLS vào visual memory.**
- `H = model.config.hidden_size`, `spatial_shape = (image_size // patch_size,) * 2`. **Không hard-code** 768, 196 hay 14.
- `pretrained=False` (dùng cho test, không cần internet): dựng `ViTModel(ViTConfig(...))` nhỏ từ `cfg.model.skelex.test_config` (ví dụ hidden 32, 2 layers, 2 heads, patch 16, image 224).
- `transformers` vẫn là optional: lỗi import phải báo rõ cần `pip install -e .[text]`.

### 1.3. Config validation

- Nới `model.backbone ∈ {densenet169, skelex}`.
- Thay `IMAGE_FEATURE_DIM` cố định bằng `encoder.token_dim` / `encoder.global_dim`.

---

## 2. Preprocessing theo backbone

Thêm `data.preprocessing ∈ {imagenet_gray3, skelex_minmax_rgb}`, mặc định suy ra từ backbone.

- `imagenet_gray3`: giữ y hệt pipeline hiện tại.
- `skelex_minmax_rgb`: y hệt `load_skelex_rgb` và các transform trong notebook S0:
  1. `convert("L")` → float32.
  2. Min-max theo từng ảnh về 0–255. Nếu `hi <= lo` hoặc không hữu hạn thì trả mảng 0.
  3. Clip, ép kiểu `uint8`, `convert("RGB")`.
  4. `Resize((224, 224))`.
  5. `RandomAffine(7, (0.03, 0.03), (0.95, 1.05))`, **chỉ khi train**.
  6. `ToTensor()`.
  7. `Normalize(ImageNet mean/std)`.

`data.preprocessing` phải nằm trong `_ARCHITECTURE_KEYS` (predict phải dùng đúng preprocessing của checkpoint).

---

## 3. Training recipe theo backbone

Recipe tự chọn theo backbone. Bạn tự thiết kế schema, nhưng phải đạt 3 yêu cầu:

- (a) Config B0–B3/A1 resolve ra **giá trị y hệt** hiện tại.
- (b) `r1_skelex_class_decoder.yaml` khác `r1_densenet_class_decoder.yaml` **chỉ ở `model.backbone`**. Recipe, preprocessing và batch size tự suy ra từ backbone.
- (c) `config_resolved.yaml` ghi rõ recipe đã dùng.

### 3.1. `densenet` (giữ nguyên)

- Adam, lr 1e-4, wd 1e-4, StepLR(25, 0.1) gọi mỗi epoch, 100 epoch, batch 32, AMP.
- Không clip gradient, không gradient accumulation.
- Với các config cũ, **optimizer phải giữ đúng một param group như hiện tại**.
- Chỉ khi model có tham số β (R1) thì tách thêm một group `weight_decay = 0` riêng cho β.

### 3.2. `skelex` (tái tạo đúng notebook S0)

| Thành phần | Giá trị |
|---|---|
| Optimizer | `AdamW(betas=(0.9, 0.999))`, base lr 5e-5, weight decay 0.05 |
| LLRD 0.75 | `layer_id`: embeddings → 0; `encoder.layer.k` → k+1; layernorm cuối của ViT → L+1; **mọi tham số không thuộc ViT** (token_proj, query_proj, attention, decoder, các classifier, β) → L+2. `lr = base_lr · 0.75^(L+2 − layer_id)`, tức module mới dùng đúng base lr. |
| No weight decay | tham số có `ndim <= 1` (**≤ 1** để bắt cả β là scalar 0 chiều; notebook S0 dùng `== 1`), tên kết thúc `.bias`, tên chứa `layernorm` |
| Schedule | `LambdaLR`, cập nhật **theo optimizer step**. Warmup tuyến tính `(step+1)/warmup_steps` trong 5 epoch, sau đó cosine về 0. `steps_per_epoch = ceil(len(train_loader) / accum)`. |
| Epoch | 50, không early stopping |
| Batch | physical 16, `grad_accum = 2` (loss chia cho accum), eval batch 16 |
| Bước optimizer | khi `batch_idx % accum == 0` hoặc ở batch cuối: `unscale_` → `clip_grad_norm_(1.0)` → `scaler.step` → `update` → `zero_grad` → `scheduler.step()` |
| Loss / AMP / checkpoint | weighted CE từ `class_weights.csv`; AMP; chọn checkpoint theo val Macro-F1 strictly |

- `last.pt` lưu đủ scheduler state theo step để resume.
- Thêm các key recipe mới vào `_RESUME_PROTOCOL_KEYS`.
- History ghi `lr` của group có layer_id cao nhất.

---

## 4. S0 trong package

Tạo `configs/s0_skelex_image.yaml`: SKELEX image-only, recipe `skelex`, head linear `Linear(H, 8)` trên CLS, không knowledge. Đây là bản package của notebook S0. Sau khi push, tôi sẽ chạy trên Colab để kiểm tra có tái tạo được khoảng 69.9% không.

---

## 5. Class-Conditioned Proposition Decoder (R1)

Đầu vào là `H = prop_features [B, M, d]` từ cross-attention hiện có, không đổi (M = 65, d = 256).

```text
g_c   = Normalize(mean_{j∈K_c} Normalize(t_j))   # [8, 768], buffer cố định, thứ tự CLASS_NAMES
ĝ_c   = W_g g_c                                    # W_g: Linear(768, d), dùng chung cho 8 class
r_ic  = Attn(query=ĝ_c, key=H_i, value=H_i, mask: chỉ j ∈ K_c)   # [B, 8, d]
l_kb  = scorer(r_ic)                               # [B, 8]
l_img = Linear(D_glob, 8)(global_feature)          # [B, 8]
logits = l_img + β · l_kb
```

- **Prototype:** tách hàm tạo prototype đang dùng cho B3 thành helper dùng chung. Buffer của decoder đặt tên `class_prototypes`, **không đổi tên buffer `prototypes` của B3**.
- **Attention:** `nn.MultiheadAttention(d, num_heads=cfg.knowledge.decoder.num_heads, dropout=cfg.knowledge.attention_dropout, batch_first=True)`, mặc định 1 head. Mask bool `[8, M]` với `True` tại `j ∉ K_c`, đăng ký là buffer. Trả về `class_attn_weights [B, 8, M]`.
- **Scorer:** `knowledge.decoder.scorer ∈ {shared, per_class}`, mặc định `shared`.
  - `shared`: `Linear(d, 1)` áp cho từng r_ic.
  - `per_class`: weight `[8, d]` + bias `[8]`, tức `s_ic = w_c·r_ic + b_c`.
- **β:** `nn.Parameter(torch.tensor(cfg.knowledge.decoder.beta_init))`, mặc định `beta_init = 0.1` (không dùng 0, vì β = 0 thì nhánh knowledge không nhận gradient). Train được, đặt tên `knowledge_logit_scale`. **Không áp weight decay** ở cả hai recipe.
- **Bỏ global aggregation:** khi dùng decoder này, không tạo `knowledge_aggregation` và không có `z_kb`.
- **Output dict:** `logits`, `logits_img`, `logits_kb`, `knowledge_logit_scale`, `prop_features`, `cross_attn_weights`, `class_attn_weights`.
- **History:** ghi thêm giá trị β mỗi epoch. Predict extras trả thêm `class_attn_weights`, `logits_img`, `logits_kb`.

### Config

```yaml
model:
  knowledge_decoder: global | class_conditioned   # mặc định global = B2/B3/A1 hiện tại
  residual_knowledge_logits: false | true
knowledge:
  decoder: {num_heads: 1, scorer: shared, beta_init: 0.1}
```

Validation:

- `class_conditioned` ⇔ `residual_knowledge_logits: true`.
- `class_conditioned` + `use_contrastive` → raise (R2 tính sau).
- `class_conditioned` + `use_metadata` → raise.
- `class_conditioned` yêu cầu `use_knowledge: true`.

---

## 6. Configs mới

```text
configs/
├── s0_skelex_image.yaml              backbone skelex, image-only
├── r1_densenet_class_decoder.yaml    base b2_knowledge + knowledge_decoder: class_conditioned + residual true
└── r1_skelex_class_decoder.yaml      base r1_densenet_class_decoder + model.backbone: skelex (chỉ đổi dòng này)
```

---

## 7. Checkpoint, predict, resume

- Thêm vào `_ARCHITECTURE_KEYS`: backbone, cấu hình SKELEX, preprocessing, decoder.
- Predict của checkpoint SKELEX phải dựng lại encoder **mà không cần tải pretrained weights** (weights đã nằm trong `state_dict`): dùng `from_config` theo config đã lưu trong checkpoint. Lưu `ViTConfig` (dạng dict) vào checkpoint.
- Resume chạy được cho cả hai recipe.

---

## 8. Tests (CPU, không internet, dùng tiny ViT)

1. **Không phá code cũ:** sinh manifest `state_dict` keys + shapes cho 5 config cũ từ commit `80db443` (`pretrained=false`, KB/cache giả), lưu vào `tests/fixtures/`. Code mới phải khớp tuyệt đối. Optimizer của config cũ vẫn là một param group với cùng lr/wd. Mọi test hiện có vẫn pass.
2. **EncoderOutput:**
   - DenseNet: `[B, 49, 1664]`, `[B, 1664]`, `(7, 7)`.
   - Tiny ViT: `[B, 196, H]`, `[B, H]`, `(14, 14)`. `global_feature == last_hidden_state[:, 0]`, tokens không chứa CLS.
3. **Preprocessing SKELEX:** ảnh hằng số → toàn 0 trước normalize; ảnh có giá trị trong [10, 20] → min 0, max 255. Kết quả khớp với bản chép lại hàm `load_skelex_rgb` của notebook.
4. **LLRD:**
   - lr từng group = `5e-5 · 0.75^(L+2 − id)`.
   - Module không thuộc ViT và β ở base lr.
   - β, bias, layernorm có weight decay 0.
   - Mọi tham số trainable nằm trong đúng một group.
5. **Scheduler SKELEX:** kiểm giá trị lr ở step 0, cuối warmup, step cuối. Với accum 2 và số batch lẻ, số optimizer step mỗi epoch = `ceil(n / 2)`.
6. **Mask decoder:**
   - Đổi giá trị `h_ij` với `j ∉ K_c` thì `r_ic` không đổi.
   - `class_attn_weights` bằng 0 ngoài K_c và tổng bằng 1 trên K_c.
   - Shape đúng cho cả hai loại scorer.
7. **Residual:**
   - `logits == logits_img + β · logits_kb`.
   - Sau một backward, `W_g`, decoder attention, scorer, `query_proj`, `token_proj` và β đều có grad khác 0.
   - Buffer không đổi.
8. **Config:**
   - Resolved config của `r1_skelex` và `r1_densenet` chỉ khác ở `model.backbone` và các giá trị suy ra từ backbone (recipe, preprocessing, batch).
   - Các rule validation ở §5 raise đúng.
9. **Smoke end-to-end:** 1 epoch, dữ liệu giả, tiny ViT cho `s0_skelex_image`, `r1_densenet_class_decoder`, `r1_skelex_class_decoder`. Có resume và predict. Predict checkpoint SKELEX không cần internet.

---

## 9. README

Thêm:

- Bảng run: B0, B2, B3, A1, S0, R1-D, R1-S.
- Cách cài `.[text]` cho SKELEX.
- Ghi chú: phép so là R1-D vs B0 (và vs B2), R1-S vs S0.
