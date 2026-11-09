# CODING GUIDE — BTXRD 8-class, proposition-guided knowledge model

> **Đối tượng đọc:** coding agent trong VS Code.
> **Mục tiêu:** xây một Python package config-driven (`btxrd`) để train B0, B1, B2, B3, A1 bằng cùng một codebase.
> **Nguồn chuẩn:** Folder thesis có file `SOURCE_OF_TRUTH_BTXRD_8CLASS_VI.md` (SOT). Nếu guide này mâu thuẫn với SOT thì SOT thắng. Hỏi lại thay vì tự suy diễn.
> **Phạm vi:** chỉ code model, data, loss, training, prediction và test. Không làm XAI, không làm cross-validation, không viết báo cáo/biểu đồ. Notebook sẽ gọi package và tự lo evaluation, lưu kết quả.

---

## 0. Quy tắc bắt buộc

1. **Không hard-code đường dẫn.** Mọi path (ảnh, CSV, KB, cache, output) đến từ YAML hoặc override lúc gọi. Không đưa bất kỳ đường dẫn Google Drive nào vào repo. Trong ví dụ chỉ dùng placeholder dạng `<DATA_ROOT>/...`.
2. **Config-driven.** B0/B1/B2/B3/A1 chỉ khác nhau bằng YAML. Không copy model code cho từng experiment.
3. **Không đổi protocol huấn luyện** ở mục 7 (augmentation, loss, optimizer, schedule, seed). Đây là protocol đã dùng cho B0/B1.
4. **Không thêm thành phần ngoài SOT:** không proposition-level supervision, không weak label, không concept bottleneck, không fine-tune text encoder, không weighted sampler/oversampling, không thêm ablation/control, không hyperparameter search.
5. **Đặt tên trung lập.** Không đặt tên biến/log kiểu `proposition_prob`, `presence`, `evidence_score`, `verification`. Dùng `prop_features` (H), `cross_attn_weights`, `aggregation_weights` (α).
6. **Test set không được đụng tới trong `train()`.** Train chỉ dùng train + val. Prediction trên test là hàm riêng, do notebook gọi.
7. Python ≥ 3.10, PyTorch + torchvision (dùng bản Colab đang có, không pin cứng version torch).

---

## 1. Dữ liệu đầu vào (đã chuẩn bị sẵn, package chỉ đọc)

| Config key | File | Cột / định dạng | Dùng khi |
|---|---|---|---|
| `paths.image_dir` | thư mục ảnh cache | PNG grayscale 512×512 (đã giữ tỉ lệ + pad đen). Tên file = `Path(image_id).stem + ".png"` | luôn |
| `paths.splits_csv` | `splits.csv` | `image_id`, `class8`, `split` ∈ {train, val, test} | luôn |
| `paths.class_weights_csv` | `class_weights.csv` | `class8`, `weight_sqrt_inv`; thứ tự dòng = `CLASS_NAMES` | luôn |
| `paths.metadata_csv` | `metadata_safe.csv` | `image_id`, `age`, `gender`, `region`, `view`, `class8`, `split` | `use_metadata: true` |
| `paths.kb_json` | `btxrd_kb_8class.json` | xem mục 1.2 | `use_knowledge: true` |
| `paths.text_embedding_cache` | file `.pt` do script tạo | xem mục 3.2 | `use_knowledge: true` |
| `paths.output_dir` | thư mục gốc lưu run | — | luôn |

Tên cột (`image_id`, `class8`, `split`, …) phải cấu hình được qua `data.columns.*` với mặc định như trên.

### 1.1. Lớp (thứ tự cố định, dùng làm index nhãn)

```python
CLASS_NAMES = [
    "normal",
    "osteochondroma",
    "multiple_osteochondromas",
    "simple_bone_cyst",
    "giant_cell_tumor",
    "osteofibroma",
    "synovial_osteochondroma",
    "osteosarcoma",
]
```

Kiểm tra khi load (raise lỗi rõ ràng nếu sai):

- `image_id` là unique.
- Tập `class8` ⊆ `CLASS_NAMES`.
- Tập `split` ⊆ {train, val, test}.
- `class_weights.csv` có đúng thứ tự `CLASS_NAMES`.
- Mọi file ảnh của split đang dùng đều tồn tại. Báo số file thiếu và 5 ví dụ.
- Nếu `data.expected_split_counts` khác `null` thì so khớp. Mặc định `{train: 2510, val: 538, test: 538}`. Để `null` được, sau này notebook có thể truyền split khác.
- Khi `use_metadata: true`: merge one-to-one với metadata. `class8` và `split` trong metadata phải khớp `splits.csv`.

### 1.2. Cấu trúc KB JSON (v1.0, đã khóa — không sửa file)

```json
{
  "classes": ["normal", "..."],
  "meta": {"version": "8class-1.0", "...": "..."},
  "propositions": [
    {"id": "P001", "text": "...", "category": "cortex", "role": "characteristic",
     "classes": ["normal"], "source_ids": ["S10"]}
  ],
  "query_ids": ["P001", "...", "P068"],
  "prototypes": {"normal": ["P001", "..."], "...": []}
}
```

- 68 proposition. `query_ids` là thứ tự chuẩn.
- `prototypes[c]` = các ID có `c` trong trường `classes`. Hai trường này nhất quán. Shared proposition có thể nằm trong nhiều prototype.
- P066–P068 là metadata proposition (`category == "age_sex"`). Mặc định bị loại bằng `knowledge.exclude_ids`.

Số proposition mỗi class sau khi loại P066–P068 (dùng trong unit test):

```python
EXPECTED_PROTO_SIZES_EXCL = {
    "normal": 12, "osteochondroma": 8, "multiple_osteochondromas": 10,
    "simple_bone_cyst": 13, "giant_cell_tumor": 12, "osteofibroma": 7,
    "synovial_osteochondroma": 9, "osteosarcoma": 16,
}
```

Không có class nào được phép rỗng sau khi loại. Raise lỗi nếu xảy ra.

---

## 2. Cấu trúc repo

Theo SOT §4.2, cộng thêm một số file hỗ trợ (đánh dấu `+`):

```text
pyproject.toml                    + package "btxrd", src layout, dependencies (mục 2.1)
requirements.txt                  + core dependencies
requirements-dev.txt              + core + text + dev
README.md                         + cài đặt, cách gọi từ notebook
src/btxrd/
├── __init__.py
├── config.py                     + load YAML, merge base, override, validate
├── constants.py                  + CLASS_NAMES, ImageNet mean/std
├── data/
│   ├── dataset.py                BTXRDDataset, build_dataloaders
│   ├── transforms.py             train/eval transforms
│   └── metadata.py               + MetadataEncoder (fit trên train)
├── knowledge/
│   ├── kb.py                     load_kb, exclude, prototype IDs
│   └── text_encoder.py           MedCPT encode + load cache
├── models/
│   ├── image_encoder.py          DenseNet169 → F, z_img
│   ├── cross_attention.py        proposition-guided cross-attention
│   ├── aggregation.py            LearnedAttentionPooling, MeanPooling
│   ├── fusion.py                 MetadataEncoder MLP, ClassifierHead
│   └── model.py                  BTXRDModel (lắp ghép theo config)
├── losses/
│   ├── classification.py         weighted CE
│   └── contrastive.py            PrototypeContrastiveLoss
├── training/
│   ├── trainer.py                train(cfg) -> result
│   ├── metrics.py                classification metrics
│   ├── checkpoint.py             save/load best.pt, last.pt
│   └── predict.py                + predict(cfg, checkpoint, split)
└── utils/
    ├── seed.py
    └── io.py                     + sha256, run_info, git commit
configs/
├── base.yaml                     + mọi default
├── b0_image.yaml
├── b1_metadata.yaml
├── b2_knowledge.yaml
├── b3_contrastive.yaml
└── ablations/
    └── a1_mean_aggregation.yaml
scripts/
├── cache_text_embeddings.py
├── train.py
└── evaluate.py                   chỉ chạy predict + lưu CSV/JSON tối thiểu
tests/
├── conftest.py                   synthetic dataset, fake KB, fake cache
└── test_*.py
```

### 2.1. Dependencies

Khai báo trong `pyproject.toml`, đồng thời tạo `requirements.txt` (core) và `requirements-dev.txt` (core + text + dev) cho ai quen cài bằng `pip install -r`.

| Nhóm | Thư viện | Ghi chú |
|---|---|---|
| core | `torch>=2.3` | cần `torch.amp.GradScaler("cuda")` |
| core | `torchvision>=0.18` | `densenet169`, `DenseNet169_Weights` |
| core | `numpy`, `pandas>=1.5`, `pillow>=9` | |
| core | `scikit-learn>=1.2` | metrics |
| core | `pyyaml>=6.0` | config |
| core | `tqdm` | progress bar |
| extra `[text]` | `transformers>=4.30` | chỉ cho `cache_text_embeddings.py` |
| extra `[dev]` | `pytest>=7` | unit test |

- **Chỉ đặt cận dưới, không pin cứng** (`==`) torch/torchvision. Colab đã có sẵn torch + CUDA; pin cứng sẽ làm pip cài lại và có thể lệch CUDA.
- Không thêm thư viện nào khác (không Hydra, không Lightning, không timm, không wandb) trừ khi hỏi trước.

---

## 3. Knowledge pipeline

### 3.1. `knowledge/kb.py`

```python
@dataclass
class KnowledgeBase:
    ids: list[str]                    # query IDs sau khi exclude, giữ thứ tự query_ids
    texts: list[str]
    class_to_ids: dict[str, list[str]]  # prototype IDs sau khi exclude
    all_ids: list[str]                # 68 ID gốc
    version: str
    sha256: str                       # sha256 của file JSON

def load_kb(path, exclude_ids: list[str], class_names: list[str]) -> KnowledgeBase
```

- Recompute `class_to_ids` từ trường `classes` của từng proposition, rồi assert khớp `prototypes` trong JSON (trước khi exclude).
- Exclude áp dụng cho **cả** query lẫn prototype.
- `exclude_ids` chứa ID không tồn tại → raise.

### 3.2. `knowledge/text_encoder.py` và `scripts/cache_text_embeddings.py`

- Encoder: **MedCPT query encoder**, HF id cấu hình được qua `knowledge.text_encoder_name`, mặc định `ncbi/MedCPT-Query-Encoder`.
- Cách lấy embedding: tokenizer + `AutoModel`, `eval()`, `torch.no_grad()`, fp32, lấy CLS token `last_hidden_state[:, 0, :]` → 768 chiều. `max_length` cấu hình được (mặc định 64). Agent đối chiếu model card của MedCPT để xác nhận cách pooling và max_length.
- Script cache **toàn bộ 68** proposition (không exclude), để thay đổi `exclude_ids` không cần cache lại.

File cache `.pt`:

```python
{
  "ids": [...68...],
  "texts": [...],
  "embeddings": FloatTensor[68, 768],   # raw, chưa normalize
  "model_name": "ncbi/MedCPT-Query-Encoder",
  "max_length": 64,
  "pooling": "cls",
  "kb_sha256": "...",
  "kb_version": "8class-1.0",
}
```

- `load_text_embeddings(cache_path, kb) -> Tensor[M, 768]`: chọn hàng theo `kb.ids`, **giữ đúng thứ tự**. Nếu `kb_sha256` trong cache ≠ `kb.sha256` → raise (bắt cache lại).
- `transformers` là optional dependency (`pip install -e .[text]`), chỉ cần cho script cache. Training không import `transformers`.
- CLI: `python scripts/cache_text_embeddings.py --kb <KB_JSON> --out <CACHE_PT> [--model ...] [--max-length 64] [--device cuda]`.

---

## 4. Kiến trúc model

Ký hiệu: B = batch, M = số proposition sau exclude (= 65), d = `knowledge.dim` (= 256).

### 4.1. `models/image_encoder.py`

```python
class DenseNet169Encoder(nn.Module):
    # backbone = torchvision.models.densenet169(weights=DEFAULT nếu pretrained else None)
    # chỉ giữ backbone.features
    def forward(self, x):            # x: [B, 3, 224, 224]
        f = self.features(x)         # [B, 1664, 7, 7]  (kết thúc ở norm5)
        f = F.relu(f)                # activation cuối, giống torchvision forward
        z_img = f.mean(dim=(2, 3))   # [B, 1664]  == GAP, tương thích B0
        return f, z_img
```

**Bắt buộc:** `z_img` phải bằng output pre-classifier của `torchvision.models.densenet169` với cùng weights (`classifier = Identity`). Có unit test cho điều này.

### 4.2. Visual tokens (trong `cross_attention.py` hoặc `model.py`)

```python
tokens = f.flatten(2).transpose(1, 2)     # [B, 49, 1664]
V = self.token_proj(tokens)               # Linear(1664, d) → [B, 49, d]
```

- `model.use_positional_embedding: false` là mặc định.
- Nếu `true`: chỉ hỗ trợ fixed 2D sine/cosine encoding (không learnable), cộng vào V. Không phải core ablation, chỉ cần có switch.

### 4.3. Query và cross-attention (`models/cross_attention.py`)

```python
# buffer, không train, không vào optimizer:
self.register_buffer("text_emb", text_emb)       # [M, 768], raw MedCPT
self.query_proj = nn.Linear(768, d)               # W_q, trainable
self.attn = nn.MultiheadAttention(d, num_heads, dropout=attention_dropout, batch_first=True)

def forward(self, V):                             # V: [B, 49, d]
    Q = self.query_proj(self.text_emb)            # [M, d]
    Q = Q.unsqueeze(0).expand(V.size(0), -1, -1)  # [B, M, d]
    H, attn_w = self.attn(Q, V, V, need_weights=True, average_attn_weights=True)
    return H, attn_w                              # H: [B, M, d], attn_w: [B, M, 49]
```

- Đúng một projection 768→d (W_q). Không thêm Linear Q/K/V riêng ngoài `MultiheadAttention`.
- Không residual, không FFN, không LayerNorm thêm. SOT định nghĩa H = CrossAttn(Q, V, V).
- Mặc định: `d = 256`, `num_heads = 4`, `attention_dropout = 0.1`.
- `attn_w` được trả ra trong output dict để notebook có thể visualize sau. Không xử lý gì thêm.

### 4.4. Aggregation (`models/aggregation.py`)

```python
class LearnedAttentionPooling(nn.Module):        # model.aggregation: learned
    # W_h: Linear(d, hidden), w: Linear(hidden, 1, bias=False); hidden = knowledge.pool_hidden (mặc định 128)
    def forward(self, H):                         # [B, M, d]
        e = self.w(torch.tanh(self.W_h(H))).squeeze(-1)   # [B, M]
        alpha = torch.softmax(e, dim=1)                    # [B, M]
        z_kb = (alpha.unsqueeze(-1) * H).sum(dim=1)        # [B, d]
        return z_kb, alpha

class MeanPooling(nn.Module):                     # model.aggregation: mean  (A1)
    def forward(self, H):
        return H.mean(dim=1), None
```

### 4.5. Metadata encoder (`models/fusion.py` + `data/metadata.py`) — cho B1

Encoding 9 chiều, giống notebook B1:

| Feature | Cách mã hóa |
|---|---|
| `age_z` | (age − mean_train) / std_train, std với `ddof=0`, chỉ tính trên split train |
| `gender_M`, `gender_F` | one-hot, thứ tự `["M", "F"]` |
| `region_*` | one-hot, thứ tự `["upper limb", "lower limb", "pelvis"]` |
| `view_*` | one-hot, thứ tự `["frontal", "lateral", "oblique"]` |

- Danh sách category và thứ tự nằm trong config (`metadata.categories`). Giá trị ngoài danh sách → raise.
- `MetadataEncoder.fit(train_df)` → lưu `age_mean`, `age_std`; được ghi vào checkpoint và dùng lại khi predict.
- MLP: `Linear(9, 32) → ReLU → Dropout(0.2) → Linear(32, 32) → ReLU`. `metadata.hidden = 32`, `metadata.dropout = 0.2`.

### 4.6. Fusion và classifier head

```python
z = concat([z_img] + ([z_meta] if use_metadata) + ([z_kb] if use_knowledge), dim=1)
logits = head(z)                                  # [B, 8]
```

`model.head.type`:

- `linear`: `Linear(in_dim, 8)`. Dùng cho B0 và B1, giống notebook.
- `mlp`: `Linear(in_dim, 512) → ReLU → Dropout(0.3) → Linear(512, 8)`. Dùng cho B2, B3, A1, **cùng một cấu hình** cho cả ba (`model.head.hidden = 512`, `model.head.dropout = 0.3`).

`in_dim` tính tự động: 1664 (+32 nếu metadata) (+d nếu knowledge).

### 4.7. Contrastive branch (B3)

Chỉ tạo khi `model.use_contrastive: true`:

```python
# prototype: buffer cố định, không gradient
t_bar = F.normalize(text_emb, dim=-1)                  # [M, 768], cùng ma trận đã chọn theo kb.ids
G[c] = F.normalize(t_bar[idx(K_c)].mean(0), dim=-1)     # [8, 768], thứ tự CLASS_NAMES
self.register_buffer("prototypes", G)

self.phi = nn.Linear(d, 768)                           # φ, trainable
u = F.normalize(self.phi(z_kb), dim=-1)                # [B, 768]
```

- `K_c` lấy từ `kb.class_to_ids` (đã exclude). Index vào ma trận embedding đã chọn theo `kb.ids`.
- Tạo `phi` **sau cùng** trong `__init__`, để thứ tự khởi tạo các module còn lại giống B2.
- B2 (`use_contrastive: false`) không có `phi`, không có buffer prototype.

### 4.8. `BTXRDModel.forward` trả về dict

```python
{
  "logits": [B, 8],
  "z_img": [B, 1664],
  # nếu use_metadata:
  "z_meta": [B, 32],
  # nếu use_knowledge:
  "prop_features": [B, M, d],          # H
  "cross_attn_weights": [B, M, 49],
  "z_kb": [B, d],
  "aggregation_weights": [B, M] | None,   # α (None khi mean)
  # nếu use_contrastive:
  "u": [B, 768],
}
```

Signature: `forward(image, meta=None)`. Nếu `use_metadata` bật mà `meta is None` → raise.

---

## 5. Loss (`losses/`)

```python
ce = nn.CrossEntropyLoss(weight=class_weights)     # sqrt-inverse weights từ CSV

class PrototypeContrastiveLoss(nn.Module):
    def forward(self, u, prototypes, y, tau):
        logits_con = (u.float() @ prototypes.float().T) / tau   # cosine vì cả hai đã normalize
        return F.cross_entropy(logits_con, y)                    # không weight

L = L_CE + λ · L_con
```

- `loss.contrastive_weight` = λ, `loss.contrastive_temperature` = τ.
- Mặc định B3: λ = 0.1, τ = 0.07. Không search.
- Tính contrastive ở fp32 (cast trong autocast).
- Validation config:
  - `use_contrastive: true` mà λ ≤ 0 → raise.
  - `use_contrastive: false` mà λ ≠ 0 → raise.
  - `use_contrastive: true` mà `use_knowledge: false` → raise.
- Không có proposition-level loss.
- Log riêng `loss_ce`, `loss_con`, `loss_total` cho train và val.

---

## 6. Data pipeline (`data/`)

### 6.1. Transforms (giữ nguyên B0/B1)

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train = Compose([
    Resize((224, 224)),
    RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval = Compose([
    Resize((224, 224)),
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

Ảnh mở bằng `PIL.Image.open(path).convert("L")`. Tham số augmentation lấy từ config, default như trên.

### 6.2. Dataset

`__getitem__` trả dict:

```python
{"image": Tensor[3,224,224], "label": int, "image_id": str}
# + "meta": Tensor[9] khi use_metadata
```

### 6.3. DataLoader

- Train: `shuffle=True`, `generator=torch.Generator().manual_seed(seed)`.
- Val/test: `shuffle=False`.
- `batch_size = 32`, `num_workers = 2`, `pin_memory = True`, `persistent_workers = num_workers > 0`. Tất cả cấu hình được.
- `debug.limit_per_split: null | int`: cắt mỗi split còn N mẫu đầu (stratify không cần), để smoke test nhanh.

---

## 7. Training protocol (`training/trainer.py`)

| Thành phần | Giá trị |
|---|---|
| Optimizer | `Adam(params có requires_grad, lr=1e-4, weight_decay=1e-4)`, một param group |
| Scheduler | `StepLR(step_size=25, gamma=0.1)`, gọi `scheduler.step()` một lần mỗi epoch, sau validation |
| Epochs | `train.epochs = 100` |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler`, chỉ bật khi CUDA có sẵn và `train.amp: true` |
| Seed | `seed = 2026`; `seed_everything` seed random/numpy/torch/cuda; `cudnn.benchmark = True` (giống B0, không đòi bit-exact) |
| Chọn checkpoint | val **Macro-F1** cải thiện **strictly** (`>`) → lưu `best.pt` |
| Early stopping | `train.early_stopping.enabled`, `patience = 15`, đếm theo val Macro-F1. B0/B2/B3/A1 tắt; B1 bật (giống notebook) |
| Val loss | weighted CE (và `loss_con` nếu B3) |

Vòng lặp mỗi epoch:

1. Train: forward → loss → `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`. Ghi train loss và train Macro-F1 (theo prediction trong lúc train).
2. Val: `model.eval()`, `no_grad`, tính metric trên toàn bộ val.
3. Ghi một dòng vào `history.csv`: `epoch`, `lr`, `train_*`, `val_*`.
4. Lưu `best.pt` nếu cải thiện; luôn lưu `last.pt`.
5. `scheduler.step()`.
6. Kiểm tra early stopping nếu bật.

**Resume** (Colab hay bị ngắt): `train.resume: true` → load `last.pt` (model, optimizer, scheduler, scaler, epoch, best metric, counter early stopping, history, RNG state) rồi chạy tiếp.

`train(cfg) -> dict`:

```python
{"run_dir": str, "best_epoch": int, "best_val_macro_f1": float,
 "epochs_ran": int, "stopped_early": bool}
```

### 7.1. Metrics (`training/metrics.py`)

`compute_metrics(y_true, y_pred, y_prob=None, class_names=CLASS_NAMES) -> dict`:

- `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`.
- Theo từng class: `<cls>_precision`, `<cls>_recall`, `<cls>_f1`, `<cls>_support`.
- Nếu có `y_prob`: `macro_auroc_ovr` và `<cls>_auroc`.
- Dùng `labels=list(range(8))`, `zero_division=0`, khớp notebook.

Trainer chỉ cần các metric không phụ thuộc prob. Hàm này để notebook dùng lại.

### 7.2. Output của một run

Thư mục: `paths.output_dir / run.name`. Nếu đã tồn tại và không `resume` và không `run.overwrite` → raise.

```text
<run_dir>/
├── config_resolved.yaml     config cuối cùng sau merge + override
├── run_info.json            xem dưới
├── history.csv
├── best.pt
└── last.pt
```

`best.pt` / `last.pt`: `epoch`, `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, `val_metrics`, `best_val_macro_f1`, `config` (dict), `metadata_stats` (nếu có), `kb_ids` (nếu có).

`run_info.json`:

- sha256 của `splits_csv`, `class_weights_csv`, `kb_json`, `text_embedding_cache`, `metadata_csv` (file nào dùng thì ghi).
- Git commit (nếu có), phiên bản torch/torchvision/python, tên GPU.
- Số tham số total/trainable.
- `best_epoch`, `best_val_macro_f1`, `epochs_ran`, thời gian chạy.

**Không** tạo test predictions, confusion matrix, biểu đồ hay summary test trong `train()`.

---

## 8. Prediction (`training/predict.py`, `scripts/evaluate.py`)

```python
predict(cfg, checkpoint_path, split="test") -> pandas.DataFrame
```

Cột của DataFrame: `image_id`, `y_true`, `y_pred`, `true_class`, `pred_class`, `p_<class>` × 8.

- Dùng eval transform, `metadata_stats` lấy từ checkpoint.
- `return_extras=True` (tùy chọn): trả thêm dict numpy `aggregation_weights` [N, M] và `cross_attn_weights` [N, M, 49] (nếu có), kèm `kb_ids`. Chỉ trả ra, không vẽ, không phân tích.

`scripts/evaluate.py --config ... --checkpoint ... --split test --out <DIR>`: gọi `predict`, lưu `predictions_<split>.csv` và `metrics_<split>.json` (từ `compute_metrics`). Không làm gì thêm.

---

## 9. Config

### 9.1. Cơ chế

- `load_config(path, overrides: dict | list[str] | None) -> Config`.
- File experiment có `base: base.yaml` (đường dẫn tương đối với file đó). Deep-merge lên base.
- Override dạng dotted key: dict `{"paths.image_dir": "...", "train.epochs": 2}` hoặc CLI `--set paths.image_dir=... train.epochs=2`. Parse giá trị bằng `yaml.safe_load`.
- Validate sau merge: key lạ → raise (tránh lỗi gõ sai); thiếu path cần thiết → raise; các rule ở mục 5.
- Không dùng Hydra. Dataclass hoặc dict + validate đều được.

### 9.2. `configs/base.yaml`

```yaml
run:
  name: null              # mặc định = tên file config
  overwrite: false

seed: 2026

paths:
  image_dir: null
  splits_csv: null
  class_weights_csv: null
  metadata_csv: null
  kb_json: null
  text_embedding_cache: null
  output_dir: null

data:
  columns: {image_id: image_id, label: class8, split: split}
  image_ext: .png
  input_size: 224
  expected_split_counts: {train: 2510, val: 538, test: 538}
  augmentation:
    degrees: 7
    translate: [0.03, 0.03]
    scale: [0.95, 1.05]
  batch_size: 32
  num_workers: 2
  pin_memory: true

metadata:
  features: [age, gender, region, view]
  categories:
    gender: [M, F]
    region: [upper limb, lower limb, pelvis]
    view: [frontal, lateral, oblique]
  hidden: 32
  dropout: 0.2

model:
  backbone: densenet169
  pretrained: true
  use_knowledge: false
  use_metadata: false
  use_contrastive: false
  aggregation: learned          # learned | mean
  use_positional_embedding: false
  head:
    type: linear                # linear | mlp
    hidden: 512
    dropout: 0.3

knowledge:
  exclude_ids: [P066, P067, P068]
  text_encoder_name: ncbi/MedCPT-Query-Encoder
  text_encoder_frozen: true     # phải luôn true; false → raise
  text_dim: 768
  dim: 256
  num_heads: 4
  attention_dropout: 0.1
  pool_hidden: 128

loss:
  contrastive_weight: 0.0
  contrastive_temperature: 0.07

train:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 1.0e-4
  step_size: 25
  gamma: 0.1
  amp: true
  early_stopping: {enabled: false, patience: 15}
  resume: false

debug:
  limit_per_split: null
```

### 9.3. Các experiment

```yaml
# configs/b0_image.yaml
base: base.yaml
run: {name: b0_image}
```

```yaml
# configs/b1_metadata.yaml
base: base.yaml
run: {name: b1_metadata}
model: {use_metadata: true}
train:
  early_stopping: {enabled: true, patience: 15}
```

```yaml
# configs/b2_knowledge.yaml
base: base.yaml
run: {name: b2_knowledge}
model:
  use_knowledge: true
  aggregation: learned
  head: {type: mlp}
```

```yaml
# configs/b3_contrastive.yaml
base: b2_knowledge.yaml
run: {name: b3_contrastive}
model: {use_contrastive: true}
loss: {contrastive_weight: 0.1, contrastive_temperature: 0.07}
```

```yaml
# configs/ablations/a1_mean_aggregation.yaml
base: ../b2_knowledge.yaml
run: {name: a1_mean_aggregation}
model: {aggregation: mean}
```

Bảng đối chiếu:

| Run | knowledge | metadata | aggregation | contrastive | head | early stop |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| B0 | – | – | – | – | linear | off |
| B1 | – | ✓ | – | – | linear | on (15) |
| B2 | ✓ | – | learned | – | mlp | off |
| B3 | ✓ | – | learned | ✓ (λ=0.1, τ=0.07) | mlp | off |
| A1 | ✓ | – | mean | – | mlp | off |

---

## 10. Scripts và API cho notebook

```bash
pip install -e .            # training
pip install -e .[text]      # thêm transformers cho script cache
pip install -e .[dev]       # pytest
```

```bash
python scripts/cache_text_embeddings.py --kb <KB_JSON> --out <CACHE_PT>
python scripts/train.py --config configs/b2_knowledge.yaml --set paths.image_dir=<...> paths.output_dir=<...>
python scripts/evaluate.py --config configs/b2_knowledge.yaml --checkpoint <RUN_DIR>/best.pt --split test --out <DIR>
```

Ví dụ notebook Colab (README chỉ dùng placeholder):

```python
# !git clone <REPO_URL> && cd <REPO> && pip install -e .[text]
from btxrd.config import load_config
from btxrd.training.trainer import train
from btxrd.training.predict import predict

PATHS = {
    "paths.image_dir":            "<DATA_ROOT>/images_512",
    "paths.splits_csv":           "<DATA_ROOT>/splits.csv",
    "paths.class_weights_csv":    "<DATA_ROOT>/class_weights.csv",
    "paths.metadata_csv":         "<DATA_ROOT>/metadata_safe.csv",
    "paths.kb_json":              "<DATA_ROOT>/btxrd_kb_8class.json",
    "paths.text_embedding_cache": "<DATA_ROOT>/medcpt_query_kb_v1.pt",
    "paths.output_dir":           "<DATA_ROOT>/runs",
}

cfg = load_config("configs/b3_contrastive.yaml", overrides=PATHS)
result = train(cfg)
df_test = predict(cfg, f"{result['run_dir']}/best.pt", split="test")
```

---

## 11. Unit tests (`pytest`, chạy được trên CPU, không cần internet)

`conftest.py` tạo:

- Dataset giả: khoảng 48 PNG grayscale nhỏ, CSV splits/weights/metadata đúng schema, đủ 8 class ở mỗi split.
- KB giả có cấu trúc giống thật (có ID bị exclude).
- Cache embedding random [M, 768] với `kb_sha256` khớp.
- Model tests dùng `model.pretrained: false`.

Test bắt buộc:

1. **KB:**
   - KB thật: 68 ID; sau exclude còn 65, đúng thứ tự P001–P065; kích thước prototype = `EXPECTED_PROTO_SIZES_EXCL`.
   - Recompute từ `classes` khớp `prototypes`.
   - Test này skip nếu không có file KB thật, cấu hình qua biến môi trường `BTXRD_KB_JSON`.
2. **Cache:** chọn đúng thứ tự theo `kb.ids`; sha mismatch → raise.
3. **Image encoder:** `z_img` allclose với `densenet169` của torchvision (classifier = Identity) khi load cùng `state_dict`.
4. **Shapes cho cả 5 config:** logits [B, 8]; B2: H [B, 65, 256], attn [B, 65, 49], z_kb [B, 256], α [B, 65] với tổng α = 1; A1: z_kb == H.mean(1), α là None; B3: u [B, 768], ‖u‖ = 1.
5. **Frozen text:** `text_emb` và `prototypes` là buffer, `requires_grad = False`, không nằm trong optimizer; ‖g_c‖ = 1.
6. **B2 vs B3:** B2 không có `phi`; với λ = 0, loss tổng == CE đúng tuyệt đối.
7. **Gradient:** sau một bước train, `query_proj`, `attn`, pooling, head (và `phi` ở B3) có grad khác 0; buffer không đổi giá trị.
8. **Metadata:** vector 9 chiều đúng thứ tự; age z-score dùng thống kê train; category lạ → raise.
9. **Config:** base merge, override dotted key, key lạ → raise, các rule λ/contrastive → raise, `text_encoder_frozen: false` → raise.
10. **Smoke end-to-end:** `train()` 2 epoch trên dataset giả cho B0, B1, B2, B3, A1 → tạo đủ file output; `resume` chạy tiếp được; `predict()` trả đủ cột.

---

## 12. Definition of done

- [ ] `pip install -e .` chạy được và không cài lại torch trên Colab; `pytest` pass trên CPU.
- [ ] `scripts/train.py` chạy được cả 5 config với `--set debug.limit_per_split=16 train.epochs=1`.
- [ ] Không có đường dẫn tuyệt đối, đường dẫn Drive hay thông tin cá nhân nào trong repo.
- [ ] README có: cài đặt, cache text embedding, train, predict, bảng config.
- [ ] Không có code XAI, cross-validation, vẽ biểu đồ hay đánh giá test bên trong training.

---

## 13. Các quyết định implementation bổ sung đã khóa

Các quyết định dưới đây bổ sung và, nếu có khác biệt, thay thế chi tiết tương ứng ở các mục trên; các phần khác của guide giữ nguyên.

1. **Thứ tự cuối epoch:** train → validation → ghi history → cập nhật best và counter early stopping, lưu `best.pt` nếu val Macro-F1 tăng strictly → `scheduler.step()` → lưu `last.pt` với scheduler state và counter mới → dừng nếu đủ patience. `last.pt` luôn sẵn sàng cho epoch kế tiếp.
2. **Contrastive λ = 0:** validation giữ nghiêm, B3 bắt buộc λ > 0. Test λ = 0 chỉ ở mức hàm loss. B2 không có `phi` và loss B2 == CE; B3 có `phi` và loss B3 == CE + λ·L_con.
3. **Metadata naming:** phần chuẩn hóa/one-hot dữ liệu là `MetadataPreprocessor` trong `data/metadata.py`; neural MLP là `MetadataMLP` trong `models/fusion.py`.
4. **Tên cột:** `data.columns` gồm `image_id`, `label`, `split`, `class_weight`, `age`, `gender`, `region`, `view`. Mặc định lần lượt là `image_id`, `class8`, `split`, `weight_sqrt_inv`, `age`, `gender`, `region`, `view`. File class weights dùng chung key `label` cho cột tên lớp.
5. **Predict config:** kiến trúc lấy từ config checkpoint; config truyền vào chỉ cung cấp runtime settings (path dữ liệu, batch size, workers, pin memory, debug). Mọi key kiến trúc lệch phải raise và liệt kê các key bị lệch. `metadata_stats` luôn lấy từ checkpoint.
6. **Predict không phụ thuộc KB/cache:** `text_emb` và `prototypes` là buffer trong `state_dict`; predict không cần `kb_json` hay text embedding cache để dựng model. Nếu config có cung cấp hai file này thì chỉ dùng để đối chiếu `kb_ids`/sha256; lệch → raise.
7. **Resume:** chỉ cam kết functional resume, không cam kết bit-exact. Vẫn lưu global RNG state và state của train generator.
8. **Dữ liệu metadata:** missing/non-numeric/non-finite age, `age_std == 0`, duplicate ID, merge không one-to-one hoặc category ngoài config đều raise; không impute.
9. **Class weights:** phải đúng 8 dòng theo `CLASS_NAMES`, không trùng, hữu hạn và lớn hơn 0; không normalize lại.
10. **AUROC thiếu lớp:** AUROC lớp không xác định nhận `NaN`; `macro_auroc_ovr` chỉ tính trên lớp hợp lệ và trả thêm `n_auroc_classes`. Khi ghi JSON, giá trị không hữu hạn chuyển thành `null`.
11. **Validation giá trị khóa:** `backbone == densenet169`, `input_size == 224`, `text_dim == 768`, số lớp bằng 8, metadata dimension suy ra từ categories phải bằng 9 và `knowledge.dim % knowledge.num_heads == 0`. `dim`, `num_heads`, `attention_dropout`, `pool_hidden` vẫn cấu hình được; 5 config chính giữ default.
12. **Predict extras:** `return_extras=True` trả `(DataFrame, extras)`. A1 có `aggregation_weights=None`. B0/B1 có mọi attention extra và `kb_ids` bằng `None`.
13. **Run directory an toàn trên Drive:** không xóa thư mục run cũ. Nếu `resume=false, overwrite=true`, đổi tên thư mục cũ thành `<run_name>__old_<YYYYmmdd-HHMMSS>` rồi tạo thư mục mới. `resume=true, overwrite=true` → raise. `resume=true` nhưng thiếu `last.pt` → raise. `resume=false, overwrite=false` và thư mục đã tồn tại → raise.
14. **Dependencies:** `.` dùng cho train/predict; `.[text]` cho cache; `.[dev]` cho pytest; `.[text,dev]` cho môi trường đầy đủ. `requirements-dev.txt` gồm core + transformers + pytest.
