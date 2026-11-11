# Prompt cho coding agent: thực nghiệm chính thức (CV 5 fold)

> Repo `trisle0805/K34-Thesis-BTXRD`, làm trên branch **`r1-dual-branch-aux`** (commit nền `6288da9`).
> Căn cứ: `LATEST/01_SOURCE_OF_TRUTH.md` (mục 3, 5.7, 5.9, 6). Không thay đổi gì ngoài phạm vi dưới đây.

## Nguyên tắc bắt buộc

1. **Mặc định giữ nguyên hành vi cũ.** Với mọi config hiện có (`b0_image.yaml`, `r1_densenet_class_decoder_aux*.yaml`, `s0_skelex_image.yaml`…), loss, dữ liệu và kết quả phải giống từng bit như trước. Mọi tính năng mới tắt theo mặc định.
2. Không hardcode đường dẫn Drive. Đường dẫn và tên run do notebook truyền qua `--set`.
3. Không đổi KB, cache MedCPT, kiến trúc decoder hay công thức `logits = logits_img + β·logits_kb`.
4. Mọi tính năng mới đều có test. `pytest -q` phải pass toàn bộ.

---

## 1. Backbone baseline

Thêm `model.backbone ∈ {densenet169, resnet50, efficientnet_b3, vit_b16, biomedclip_vitb16}` (giữ `skelex` để tương thích ngược). Mọi encoder trả về `EncoderOutput(visual_tokens, global_feature, spatial_shape)` và có thuộc tính `token_dim`, `global_dim`.

| backbone | Trọng số | global_feature | visual_tokens |
|---|---|---|---|
| `resnet50` | torchvision `ResNet50_Weights.IMAGENET1K_V2` | GAP của `layer4` (2048) | 7×7 = 49 token × 2048 |
| `efficientnet_b3` | torchvision `EfficientNet_B3_Weights.IMAGENET1K_V1` | GAP của `features` (1536) | 7×7 = 49 token × 1536 |
| `vit_b16` | HF `google/vit-base-patch16-224` (`ViTModel`, bỏ pooler) | CLS sau layernorm cuối (768) | 196 patch token × 768 |
| `biomedclip_vitb16` | open_clip `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, chỉ dùng `visual.trunk` (timm ViT-B/16) | CLS sau norm cuối (768), không dùng projection head của CLIP | 196 patch token × 768 |

- Head: `Linear(global_dim, 8)`, giống B0.
- **Recipe:** thêm mapping trong `configs/base.yaml → recipes.backbones`:
  - `resnet50`, `efficientnet_b3` → `densenet` (Adam/StepLR, 100 epoch, batch 32);
  - `vit_b16`, `biomedclip_vitb16` → recipe mới `vit`: giống `skelex` (AdamW 5e-5, wd 0,05, LLRD 0,75, warmup 5 + cosine, 50 epoch, batch 16 × accum 2, clip 1,0) nhưng preprocessing khác:
    - `vit_b16` dùng `imagenet_gray3`;
    - `biomedclip_vitb16` dùng preprocessing mới `clip_gray3` (gray → 3 kênh, mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`).
- **LLRD:** tổng quát hóa `training/optim.py`. Mỗi encoder ViT cung cấp `num_layers` và hàm `layer_id(param_name)`:
  - 0 = embeddings / patch_embed / cls / pos;
  - 1…L = block i;
  - L+1 = norm cuối;
  - L+2 = head và mọi module ngoài encoder.

  Giữ kiểm tra "đủ layer id", và giữ nguyên kết quả cho SKELEX.
- Thêm `open_clip_torch` vào extras mới `[biomedclip]` trong `pyproject.toml`. Import lazy, báo lỗi rõ nếu thiếu.

## 2. Augmentation và cân bằng lớp

Thêm config, mặc định giữ hành vi cũ:

```yaml
data:
  augmentation:
    degrees: 7
    translate: [0.03, 0.03]
    scale: [0.95, 1.05]
    hflip_p: 0.0            # mới; protocol chính thức 0.5
    brightness: 0.0         # mới; protocol chính thức 0.1 (ColorJitter)
    contrast: 0.0           # mới; protocol chính thức 0.1
  sampler: none             # mới: none | sqrt_inverse
loss:
  class_weighting: csv      # mới: csv (cũ) | train_split | none
```

- **Thứ tự transform (train):** Resize → RandomHorizontalFlip(hflip_p) → RandomAffine → ColorJitter(brightness, contrast) → Grayscale(3) → ToTensor → Normalize.
  - Khi `hflip_p == 0` và jitter bằng 0 thì **không thêm** các transform đó, để giữ đúng thứ tự gọi RNG như cũ.
  - Val/test không đổi.
- **`sampler: sqrt_inverse`:** `WeightedRandomSampler(weights = n_c^{-1/2} theo nhãn từng ảnh train, num_samples = len(train), replacement = True, generator = generator train hiện có)`. `n_c` đếm trên **chính split train đang dùng** (theo từng fold). Resume phải tái lập được.
- **`class_weighting`:**
  - `csv`: như cũ, đọc `class_weights.csv`;
  - `train_split`: tính 1/√n_c trên split train hiện tại, chuẩn hóa giống cách cột `weight_sqrt_inv` đang dùng (agent kiểm tra file và ghi lại công thức);
  - `none`: CE không trọng số.

  Áp dụng cho mọi CE: final, img, kb.
- Validation config: `sampler: sqrt_inverse` đi kèm `class_weighting: csv` hoặc `train_split` thì báo `ValueError`, để tránh bù hai lần.
- `run_info.json`: ghi số mẫu mỗi lớp của train, trọng số sampler hoặc trọng số CE đã dùng.

## 3. Fold CV gom theo ca

Script `scripts/make_cv_folds.py`:

```text
python scripts/make_cv_folds.py \
  --splits-csv <01_prep/splits.csv> \
  --groups-csv <file nhóm ca> --group-col <tên cột nhóm> \
  --out <01_prep/cv_folds_8class.csv> \
  --n-folds 5 --inner-val-splits 10 --seed 2026 [--allow-missing-groups]
```

- Tập ảnh = toàn bộ `image_id` trong `--splits-csv` (3.586 ảnh). Nhãn lấy từ `data.columns.label` (`class8`).
- Ảnh không có trong file nhóm:
  - mặc định **lỗi**, kèm danh sách;
  - với `--allow-missing-groups`, mỗi ảnh thành một nhóm riêng và được liệt kê trong báo cáo.
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` → cột `fold` (0–4).
- Với mỗi fold k:
  - test = fold k;
  - phần còn lại chia `StratifiedGroupKFold(n_splits=inner_val_splits, shuffle=True, random_state=seed + k)`, lấy split đầu tiên làm val (khoảng 10%), còn lại là train;
  - ghi vào cột `split_fold{k}` ∈ {train, val, test}.
- **Output CSV:** `image_id, class8, group_id, fold, split_fold0 … split_fold4`.
- **Kèm `<out>_report.json`:**
  - số ảnh theo lớp × split cho từng fold;
  - kiểm tra không nhóm nào nằm ở hai split trong cùng một fold;
  - mỗi ảnh là test đúng 1 lần;
  - mọi lớp có mặt ở train, val và test của mọi fold (nếu không thì cảnh báo rõ);
  - sha256 của input và output; seed.
- **Chạy một fold:** dùng loader hiện có, không cần code mới:

  ```text
  --set paths.splits_csv=<cv_folds_8class.csv> data.columns.split=split_fold{k} data.expected_split_counts=null
  ```

  Kiểm tra loader chấp nhận `expected_split_counts: null`.

## 4. PECL

```yaml
model:
  use_pecl: false
loss:
  pecl_weight: 0.0          # A1/A3/A4: 0.1
  pecl_temperature: 0.07
```

- `u = normalize(φ(z_img))`, `φ = Linear(global_dim, text_dim = 768)`.
- `L_PECL = CE( (u · Gᵀ) / τ , y )`, với `G ∈ ℝ^{8×768}` là `build_class_prototypes(...)` (đã có): buffer cố định, không gradient. Dùng CE **không trọng số**.
- Tổng loss: `L = L_cls (+ aux) + pecl_weight · L_PECL`. Không đổi công thức dự đoán.
- **Hoạt động độc lập với decoder:** `use_pecl: true` + `use_knowledge: false` (dòng A1) vẫn nạp KB và cache để dựng prototype, nhưng không tạo cross-attention hay decoder.
- **Hoán vị:** `knowledge.embedding_permutation_seed` phải áp dụng **trước** khi dựng prototype PECL (A4 xáo trộn cả query, prototype decoder và prototype PECL). Viết test kiểm tra điều này.
- Không cho bật `use_pecl` cùng `use_contrastive` (loss cũ trên `z_kb`): báo `ValueError`.
- Log: `train_loss_pecl`, `val_loss_pecl` trong `history.csv`.

## 5. Báo cáo CV

Script `scripts/cv_report.py`:

```text
python scripts/cv_report.py \
  --cv-dir <04_runs/cv> --folds-csv <cv_folds_8class.csv> \
  --configs b0_densenet169 b_resnet50 b_efficientnet_b3 b_vit_b16 b_biomedclip_vitb16 \
            a1_densenet169_pecl a2_densenet169_kdecoder a3_densenet169_kdecoder_pecl a4_densenet169_kdecoder_pecl_shuffled \
  --out <04_runs/cv/_report> --n-boot 2000 --seed 2026
```

- **Layout đầu vào:** `<cv-dir>/<config>/<config>_fold{k}/evaluation/predictions_test.csv`, k = 0–4 (`predictions_test.csv` do `evaluate.py --split test` sinh ra, với `data.columns.split=split_fold{k}`).
- Kiểm tra: mỗi config đủ 5 fold; ghép lại phủ đúng 3.586 ảnh, mỗi ảnh 1 lần; test của fold k đúng là các ảnh có `fold == k`.
- **Output:**
  - `oof_<config>.csv`: dự đoán ngoài fold, gồm xác suất 8 lớp, `fold`, `group_id`;
  - `metrics_pooled.csv`: mỗi config 1 dòng, gồm Acc, BA, Macro-F1, macro AUROC (OvR), kèm 95% CI bootstrap theo `group_id`;
  - `metrics_per_fold.csv` và `macro_f1_mean_sd.csv`;
  - `per_class_pooled.csv`: precision/recall/F1/support từng lớp;
  - `confusion_<config>.csv`;
  - `paired_diff.csv`: chênh Macro-F1 (A − B), 95% CI bootstrap ghép cặp theo nhóm (cùng chỉ số mẫu lại cho A và B), p một phía = tỷ lệ bootstrap ≤ 0, số fold A > B. Các cặp:
    - Q1: a2 − b0;
    - Q2: a1 − b0, a3 − a2;
    - Q3: a3 − b0, a3 − a1;
    - Q4: a3 − a4;
    - Q5: a3 − mỗi baseline;
  - `REPORT.md`: bảng baseline, bảng ablation (dạng Table 5 của RAD: ✓/×), bảng Q1–Q5 kèm kết luận theo quy tắc "CI không chứa 0".
- Config chưa đủ fold thì bỏ qua và ghi cảnh báo, không lỗi.

## 6. Config

**Protocol chung** `configs/cv/_protocol.yaml`:

```yaml
data:
  expected_split_counts: null
  augmentation: {hflip_p: 0.5, brightness: 0.1, contrast: 0.1}
  sampler: sqrt_inverse
loss:
  class_weighting: none
```

**9 config CV**, mỗi config `base: _protocol.yaml` và chỉ khai báo phần khác:

| File | Nội dung |
|---|---|
| `configs/cv/b0_densenet169.yaml` | DenseNet-169, chỉ ảnh |
| `configs/cv/b_resnet50.yaml` | `model.backbone: resnet50` |
| `configs/cv/b_efficientnet_b3.yaml` | `efficientnet_b3` |
| `configs/cv/b_vit_b16.yaml` | `vit_b16` |
| `configs/cv/b_biomedclip_vitb16.yaml` | `biomedclip_vitb16` |
| `configs/cv/a1_densenet169_pecl.yaml` | `use_pecl: true`, `pecl_weight: 0.1` |
| `configs/cv/a2_densenet169_kdecoder.yaml` | decoder `class_conditioned`, residual, `aux_image_weight: 1.0`, `aux_knowledge_weight: 1.0` |
| `configs/cv/a3_densenet169_kdecoder_pecl.yaml` | A2 + `use_pecl: true`, `pecl_weight: 0.1` |
| `configs/cv/a4_densenet169_kdecoder_pecl_shuffled.yaml` | A3 + `knowledge.embedding_permutation_seed: 0` |

**Chạy thử:** `configs/pilot/b0_densenet169_newaug.yaml` = `b0_image.yaml` + augmentation/sampler/class_weighting như `_protocol.yaml`, **giữ** split cố định (`expected_split_counts` 2510/538/538).

## 7. Test bắt buộc

1. Config cũ (`b0_image`, `r1_densenet_class_decoder_aux`, `…_shuffled`): loss của batch đầu và dự đoán từ checkpoint cũ **trùng tuyệt đối** với trước khi sửa.
2. Transform: `hflip_p=0`, jitter 0 cho ra đúng pipeline cũ; bật lên thì có đủ transform.
3. Sampler: trọng số đúng 1/√n_c theo split train; tái lập được với cùng seed và khi resume.
4. `class_weighting=train_split`: khớp với công thức của `class_weights.csv` khi áp lên split cố định.
5. `make_cv_folds`: trên fixture có nhóm:
   - không rò nhóm giữa train/val/test trong mọi fold;
   - mỗi ảnh test đúng 1 lần;
   - ảnh thiếu nhóm thì báo lỗi, trừ khi có `--allow-missing-groups`;
   - output tái lập được.
6. PECL: loss khớp tính tay; A1 không tạo decoder; hoán vị làm đổi prototype PECL đúng theo mapping.
7. Backbone mới: forward với ảnh 224 cho shape đúng (dùng mock hoặc trọng số ngẫu nhiên, không tải mạng trong test); LLRD đủ layer id cho `vit_b16` và `biomedclip_vitb16`.
8. `cv_report`: trên dự đoán giả lập 5 fold, chỉ số gộp khớp tính tay; bootstrap tái lập; thiếu fold thì có cảnh báo.
9. Smoke train 1 epoch (dữ liệu tổng hợp) cho cả 9 config CV và config chạy thử.

## 8. Bàn giao

- Commit lên `r1-dual-branch-aux` kèm một đoạn mô tả ngắn: file mới, config mới, kết quả `pytest -q`.
- Cập nhật README: cách tạo fold, cách chạy 1 fold, cách chạy report.
