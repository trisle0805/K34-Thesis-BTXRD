# CODING GUIDE V2 — Hierarchical Knowledge-guided Decoding (F1, H1, H2, H2-N)

> **Repo:** `trisle0805/K34-Thesis-BTXRD`, nhánh `main`, từ commit `32731c2`.
> **Spec chuẩn:** `thesis/SOURCE_OF_TRUTH_BTXRD_8CLASS_VI.md` **bản v2 (đã khóa 11/11/2026)**. Tôi sẽ thay file này vào repo trước khi bạn bắt đầu. Nếu guide này mâu thuẫn với SOT v2 thì SOT thắng; chỗ nào không rõ thì hỏi, không tự suy diễn.
> **Mục tiêu:** thêm một "query decoder" kiểu RAD. Query là tri thức của từng nút trên cây phân loại; mỗi query ra logit của chính nút đó. Dự đoán bằng softmax phân tầng, có thêm loss PECL. Bốn run F1, H1, H2, H2-N chỉ khác nhau ở config.

---

## 0. Cách làm việc

1. Đọc SOT v2 (§4–§8, §11, §13) và code hiện tại: `models/model.py`, `models/image_encoder.py`, `models/class_decoder.py`, `knowledge/kb.py`, `knowledge/text_encoder.py`, `config.py`, `training/trainer.py`, `training/predict.py`, `training/metrics.py`, `scripts/cache_text_embeddings.py`.
2. **Gửi implementation plan ngắn** (tối đa khoảng 1 trang): file nào thêm, file nào sửa, schema config mới. Chờ tôi xác nhận rồi mới code.
3. Code → chạy `pytest -q` trên clone sạch → commit + push.
4. Báo lại: commit hash, danh sách file đã đổi, kết quả pytest, và các chỗ bạn phải tự quyết.

### Không được làm

- Không phá B0/B1/B2/B3/A1/S0/R1. Config cũ phải ra đúng `state_dict` keys/shapes và optimizer như hiện tại. Checkpoint cũ vẫn predict được.
- Không sửa KB, không đổi MedCPT, không fine-tune text encoder.
- Không thêm nhãn yếu, không có loss hay điểm số ở mức từng proposition.
- Không thêm metadata, multi-scale, lesion supervision, positional embedding.
- Không thêm thư viện mới. Không hard-code đường dẫn hay số chiều.
- Đặt tên trung lập: không dùng `presence`, `evidence_score`, `concept_prob`.

---

## 1. Cây phân loại (taxonomy)

### 1.1. File `configs/taxonomy/btxrd_8class.yaml` (thêm vào repo)

```yaml
version: btxrd-8class-taxonomy-1.0
source_note: "Grouping per S10 (WHO manual of diagnostic imaging, ch. 9.6) + BTXRD benign/malignant labels"
nodes:
  - {id: root,                     parent: null,           name: "bone radiograph"}
  - {id: normal,                   parent: root,           name: "normal bone",                          leaf_class: normal}
  - {id: tumour,                   parent: root,           name: "bone tumour",                          source: "S10 9.6"}
  - {id: benign,                   parent: tumour,         name: "benign bone tumour",                   source: "S10 9.6.1-9.6.5"}
  - {id: osteosarcoma,             parent: tumour,         name: "osteosarcoma",                         leaf_class: osteosarcoma, source: "S10 9.6.6"}
  - {id: cartilage,                parent: benign,         name: "benign cartilage-forming bone tumour",  source: "S10 9.6.2; 9.5"}
  - {id: osteofibroma,             parent: benign,         name: "osteofibrous dysplasia",               leaf_class: osteofibroma, source: "S10 9.6.3"}
  - {id: unknown_origin,           parent: benign,         name: "benign bone tumour of unknown origin", source: "S10 9.6.5"}
  - {id: osteochondroma,           parent: cartilage,      name: "osteochondroma",                       leaf_class: osteochondroma}
  - {id: multiple_osteochondromas, parent: cartilage,      name: "multiple osteochondromas",             leaf_class: multiple_osteochondromas}
  - {id: synovial_osteochondroma,  parent: cartilage,      name: "synovial chondromatosis",              leaf_class: synovial_osteochondroma}
  - {id: simple_bone_cyst,         parent: unknown_origin, name: "simple bone cyst",                     leaf_class: simple_bone_cyst}
  - {id: giant_cell_tumor,         parent: unknown_origin, name: "giant cell tumour of bone",            leaf_class: giant_cell_tumor}
```

### 1.2. Module `knowledge/taxonomy.py`

```python
@dataclass(frozen=True)
class Taxonomy:
    node_ids: list[str]              # 12 nút con (không gồm root), thứ tự cố định theo file
    parent: dict[str, str]
    children: dict[str, list[str]]   # chỉ các nút quyết định
    decision_nodes: list[str]        # theo file hiện tại: root, tumour, benign, cartilage, unknown_origin
    leaf_of_class: dict[str, str]    # CLASS_NAMES -> node id lá
    paths: dict[str, list[str]]      # lá -> danh sách nút từ con của root xuống lá
    names: dict[str, str]
    sha256: str

def load_taxonomy(path, class_names) -> Taxonomy
def flat_taxonomy(class_names) -> Taxonomy   # root -> 8 lá, dùng cho F1
```

Validation (raise nếu sai):
- Đúng một root.
- Không có chu trình.
- Mỗi lá có `leaf_class` duy nhất, và 8 lá phủ đúng `CLASS_NAMES`.
- **Mọi nút quyết định có ≥ 2 con** (nút một con phải gộp, như SOT §4.2).

### 1.3. Gán proposition vào nút (lowest common ancestor)

`assign_propositions(kb, taxonomy) -> dict[node_id, list[prop_id]]`

- Với mỗi proposition (sau exclude P066–P068): lấy tập lá ứng với các lớp trong trường `classes`, rồi gán proposition vào **nút sâu nhất là tổ tiên chung của mọi lá đó**.
- `own(n)` là các proposition được gán cho chính n.
- `sub(n)` là `own(n)` cộng `own` của mọi nút con cháu của n.
- Nút không có `own` thì dùng `sub(n)`. Với cây hiện tại, trường hợp này không xảy ra.
- Proposition có LCA là root (nếu có) không thuộc nút nào. Trong KB hiện tại không có trường hợp này. Nếu gặp thì log cảnh báo và báo lại.

---

## 2. Cache tên nút (cho H2-N)

Mở rộng `scripts/cache_text_embeddings.py`: thêm chế độ `--taxonomy <yaml> --out <pt>` để encode **tên của 12 nút** bằng cùng MedCPT, cùng pooling và cùng `max_length`.

Nội dung cache:

```python
{"node_ids": [...12...], "texts": [...], "embeddings": FloatTensor[12, 768],
 "model_name": ..., "max_length": ..., "pooling": "cls", "taxonomy_sha256": ...}
```

Hàm load phải kiểm tra:
- `model_name` khớp `knowledge.text_encoder_name`;
- `taxonomy_sha256` khớp file taxonomy;
- `node_ids` khớp.

Sai bất kỳ điểm nào → raise.

Thêm config `paths.node_name_cache`, chỉ bắt buộc khi query hoặc prototype dùng tên nút.

---

## 3. Model

### 3.1. Chọn nhánh model

Thêm giá trị mới `model.knowledge_decoder: query`, bên cạnh `global` (v1) và `class_conditioned` (R1).

Khi dùng `query`:
- **Không** tạo `PropositionCrossAttention`, aggregation, `class_decoder`, `classifier` hay β.
- Chỉ tạo image encoder, `QueryDecoder`, và φ nếu bật PECL.

### 3.2. Vector tri thức của nút (buffer cố định, chiều 768)

| `knowledge.query_source` | q̄_n |
|---|---|
| `class_set` (F1, cây phẳng) | Normalize(mean_{j∈K_c} Normalize(t_j)). Đây đúng là `build_class_prototypes` hiện có. |
| `own` (H1, H2) | Normalize(mean_{j∈own(n)} Normalize(t_j)) |
| `name` (H2-N) | Normalize(embedding tên nút từ cache §2) |

Lưu thành `register_buffer("node_text", [N, 768])`, trong đó N = 8 (phẳng) hoặc 12 (phân tầng). Buffer này không có gradient và không vào optimizer.

### 3.3. `models/query_decoder.py`

```python
class QueryDecoder(nn.Module):
    # token_proj: Linear(encoder.token_dim, d)
    # query_proj: Linear(768, d)                   # W_q
    # memory_norm, query_norm: LayerNorm(d)        # như RAD chuẩn hóa memory và query trước decoder
    # layers: num_layers × DecoderLayer (pre-norm):
    #     x = x + Drop(SelfAttn(LN1(x)))           # self-attention giữa các query
    #     x = x + Drop(CrossAttn(LN2(x), V, V))    # cross-attention vào visual token, need_weights=True
    #     x = x + Drop(FFN(LN3(x)))                # Linear(d, ffn) - ReLU - Dropout - Linear(ffn, d)
    # final_norm: LayerNorm(d)
    # out_dropout: Dropout(0.1)
    # head: Linear(d, 1)                           # MỘT head dùng chung cho mọi query

    def forward(self, visual_tokens):              # [B, L, token_dim]
        V = self.memory_norm(self.token_proj(visual_tokens))                          # [B, L, d]
        Q = self.query_norm(self.query_proj(self.node_text)).expand(B, -1, -1)        # [B, N, d]
        for layer in self.layers:
            Q, cross_w = layer(Q, V)                                                  # cross_w: [B, N, L]
        node_logits = self.head(self.out_dropout(self.final_norm(Q))).squeeze(-1)     # [B, N]
        return node_logits, cross_w                                                  # cross_w của layer cuối
```

Config mặc định:

```yaml
knowledge:
  query_decoder:
    num_layers: 4
    num_heads: 4
    ffn_dim: 1024
    dropout: 0.1
    init_std: 0.02
```

d = `knowledge.dim` = 256.

- **Khởi tạo:** mọi `nn.Linear` và `nn.MultiheadAttention` (`in_proj_weight`, `out_proj.weight`) trong QueryDecoder lấy normal(0, `init_std`), bias = 0. Khởi tạo này giống code RAD. Không áp lên image encoder.
- Không dùng positional embedding.
- `visual_tokens` lấy từ `EncoderOutput` hiện có, nên DenseNet (49 token) và SKELEX (196 token) dùng chung code.

### 3.4. Softmax phân tầng (`models/hierarchy.py`)

```python
def hierarchical_log_probs(node_logits, taxonomy) -> tuple[Tensor, Tensor]:
    # Tính ở float32, kể cả khi đang chạy AMP.
    # 1) Mỗi nút quyết định p: log P(c|p) = log_softmax(node_logits[:, children(p)])
    # 2) log P(nút) = log P(cha) + log P(nút|cha), với log P(root) = 0
    # Trả về:
    #   leaf_log_probs  [B, 8]   theo thứ tự CLASS_NAMES
    #   node_log_marg   [B, N]   log xác suất biên của từng nút
```

Index được tính trước một lần, lưu thành buffer, không lặp Python theo từng mẫu.

**Output của model:**

```python
{
  "logits": leaf_log_probs,           # [B, 8] — là LOG-XÁC SUẤT, xem §4.1
  "node_logits": node_logits,         # [B, N]
  "node_log_marg": node_log_marg,     # [B, N]
  "cross_attn_weights": cross_w,      # [B, N, L]
  "z_img": z_img,
  "u": ...                            # chỉ có khi bật PECL
}
```

Cây phẳng được xử lý bằng cùng hàm này: chỉ có root với 8 con, nên `leaf_log_probs = log_softmax(node_logits)`.

---

## 4. Loss

### 4.1. Loss phân loại: giữ nguyên weighted CE hiện có

Vì `logits` là log-xác suất đã chuẩn hóa (tổng exp bằng 1), ta có `log_softmax(logits) == logits`. Do đó `nn.CrossEntropyLoss(weight=class_weights)` hiện có cho đúng:

L_H = −Σ_i w_{y_i} · log P(lá đúng) / Σ_i w_{y_i}

Đây chính là CE có giám sát ở mọi tầng trên đường đi (SOT §6.1).

- **Không viết loss phân loại mới.**
- Viết test chứng minh tương đương với cách tính tay `−Σ_p log P(con đúng | p)`.
- Metric, argmax và `p_<class>` trong predict tiếp tục chạy đúng, vì softmax(logits) = P(lá).

### 4.2. PECL (`losses/pecl.py`)

- φ = Linear(`encoder.global_dim`, 768), khởi tạo normal(0, 0,02). **Không hard-code 1664.**
- u = Normalize(φ(z_img)).
- **Prototype nút** (buffer cố định `node_prototypes` [N, 768]):
  - `knowledge.pecl_prototype: sub` (H2): Normalize(mean_{j∈sub(n)} Normalize(t_j)).
  - `name` (H2-N): Normalize(embedding tên nút).
  - Cây phẳng với `sub`: dùng K_c, tức `build_class_prototypes`.
- sims = (u @ node_prototypesᵀ) / τ, tính ở float32, ra [B, N].
- **L_PECL = trung bình theo batch của −log P_sims(lá đúng)**, với P_sims tính bằng **chính `hierarchical_log_probs(sims, taxonomy)`**. Không có trọng số lớp, giống B3.
  - Với cây phân tầng: tổng softmax CE giữa các nút anh em dọc đường đi.
  - Với cây phẳng: softmax CE trên 8 prototype.
- Tổng loss: L = CE + `loss.pecl_weight` · L_PECL. Mặc định λ = 0,1, τ = 0,07.
- Log riêng `loss_ce`, `loss_pecl` và `loss_total` cho cả train lẫn val.

```yaml
loss:
  pecl_weight: 0.0          # >0 thì bật PECL, tạo φ và buffer prototype
  pecl_temperature: 0.07
knowledge:
  pecl_prototype: sub       # sub | name
```

Giữ nguyên `contrastive_*` của B3. Validation: `pecl_weight > 0` cùng lúc với `use_contrastive: true` → raise.

---

## 5. Config

### 5.1. Key mới trong `base.yaml`

Mặc định phải giữ cho config cũ không đổi hành vi.

```yaml
paths:
  node_name_cache: null
knowledge:
  hierarchy: flat            # flat | tree
  taxonomy_file: taxonomy/btxrd_8class.yaml    # tương đối so với thư mục configs/; có thể override bằng đường dẫn tuyệt đối
  query_source: class_set    # class_set | own | name
  pecl_prototype: sub        # sub | name
  query_decoder: {num_layers: 4, num_heads: 4, ffn_dim: 1024, dropout: 0.1, init_std: 0.02}
loss:
  pecl_weight: 0.0
  pecl_temperature: 0.07
```

### 5.2. Bốn config mới

Tên run mặc định là tên file. Notebook override được.

```yaml
# configs/f1_flat_decoder.yaml
base: base.yaml
model: {use_knowledge: true, knowledge_decoder: query}
knowledge: {hierarchy: flat, query_source: class_set}
loss: {pecl_weight: 0.0}
```

```yaml
# configs/h1_hier_decoder.yaml
base: f1_flat_decoder.yaml
knowledge: {hierarchy: tree, query_source: own}
```

```yaml
# configs/h2_hier_pecl.yaml
base: h1_hier_decoder.yaml
knowledge: {pecl_prototype: sub}
loss: {pecl_weight: 0.1, pecl_temperature: 0.07}
```

```yaml
# configs/h2n_hier_name.yaml
base: h2_hier_pecl.yaml
knowledge: {query_source: name, pecl_prototype: name}
```

### 5.3. Validation

Các trường hợp phải raise:
- `query_source: own` hoặc `hierarchy: tree` mà không đọc được file taxonomy.
- `query_source: name` hoặc `pecl_prototype: name` mà không có `paths.node_name_cache`.
- `query_source: own` đi với `hierarchy: flat`. Ở cây phẳng, own(lá) chỉ còn proposition đặc trưng, sai với định nghĩa F1.
- `knowledge_decoder: query` đi với `use_metadata`, `use_contrastive` hoặc `residual_knowledge_logits`.
- `knowledge.query_decoder.*` và `pecl_*` phải nằm trong `_ARCHITECTURE_KEYS` / `_RESUME_PROTOCOL_KEYS` khi cần.

---

## 6. Checkpoint, predict, resume

- Mọi tensor tri thức (`node_text`, `node_prototypes`) là buffer nằm trong `state_dict`.
- Checkpoint lưu thêm cấu trúc taxonomy (dạng dict), `node_ids`, `taxonomy_sha256`, và proposition-to-node assignment (để tra cứu).
- **Predict không cần file KB, cache hay taxonomy.**
  - Model dựng lại hoàn toàn từ checkpoint.
  - Nếu cfg có truyền các file này thì chỉ dùng để đối chiếu sha256; lệch → raise.
  - Lưu ý: `checkpoint_buffers()` / `_build_model_inputs` hiện giả định có `cross_attention.text_emb`. Phải xử lý nhánh `query` để không đòi buffer này.
- `predict(..., return_extras=True)` trả thêm `node_ids`, `node_logits`, `node_marginals` (= exp(`node_log_marg`)) và `cross_attn_weights`.
- Resume chạy được cho cả 4 config mới.

---

## 7. Metric phân tầng (`training/metrics.py`)

Thêm `hierarchical_metrics(y_true, y_pred, taxonomy)`. Hàm chỉ tính từ **nhãn 8 lớp dự đoán** cộng với taxonomy, nên áp được cho **mọi model, kể cả B0**:

- `normal_vs_tumour_f1`: macro-F1 hai lớp, normal vs mọi lá thuộc nhánh `tumour`.
- `benign_vs_malignant_f1`: macro-F1 hai lớp, tính trên các ca **thật sự là u**. Dự đoán normal cho một ca u tính là sai ở tầng này.
- `severe_error_rate`: tỉ lệ ca mà dự đoán khác nhãn thật ở tầng normal/u, **hoặc** ở tầng lành/ác.
- `node_accuracy_<p>`: với mỗi nút quyết định p, xét các ca có đường đi thật qua p; đếm tỉ lệ ca mà con của p trên đường đi dự đoán trùng con thật.

Trainer gọi thêm hàm này trên val mỗi epoch và ghi vào `history.csv`. Taxonomy dùng file mặc định nếu config không có. **Chọn checkpoint vẫn theo val Macro-F1 8 lớp.** `evaluate.py` cũng ghi các metric này vào `metrics_<split>.json`.

---

## 8. Tests (CPU, không internet)

1. **Taxonomy:**
   - File thật có 12 nút con, 5 nút quyết định, 8 lá khớp `CLASS_NAMES`.
   - Nút một con, chu trình, hoặc lá trùng → raise.
   - `flat_taxonomy` có 1 nút quyết định với 8 con.
2. **Gán LCA** trên KB thật (skip nếu không có biến môi trường `BTXRD_KB_JSON`). Số `own` phải đúng bảng SOT §4.3:
   - tumour 5, benign 3, cartilage 5, unknown_origin 3;
   - normal 12, osteosarcoma 11, synovial_osteochondroma 8, simple_bone_cyst 6, multiple_osteochondromas 4, osteofibroma 4, giant_cell_tumor 3, osteochondroma 1;
   - tổng 65, không chứa P066–P068.
3. **Softmax phân tầng:**
   - Σ exp(leaf_log_probs) = 1.
   - Cây phẳng cho kết quả bằng `log_softmax`.
   - Một ví dụ tính tay khớp.
   - P(tumour) bằng tổng P của 7 lá u.
4. **Tương đương loss:** weighted `CrossEntropyLoss` trên `logits` bằng công thức tay `−Σ w_y Σ_p log P(con đúng|p) / Σ w_y`.
5. **QueryDecoder:**
   - Shape: N = 8 hoặc 12; `cross_attn_weights` [B, N, 49] với DenseNet và [B, N, 196] với tiny ViT.
   - Chỉ có **một** `Linear(d, 1)` làm head.
   - Sau một backward, `query_proj`, `token_proj`, các layer decoder và head đều có grad khác 0.
   - `node_text` và `node_prototypes` không nằm trong `model.parameters()`, giá trị không đổi sau một step.
6. **PECL:**
   - Cây phẳng: kết quả bằng CE trên cos/τ.
   - Cây phân tầng: bằng tổng CE giữa các anh em, tính tay.
   - Chiều vào của φ = `encoder.global_dim`: 1664 với DenseNet, hidden size với tiny ViT.
7. **H2-N:**
   - `node_text` và `node_prototypes` bằng embedding tên nút đã normalize.
   - Đổi embedding proposition trong cache thì output của H2-N không đổi.
8. **Config:**
   - Resolved config của F1/H1/H2/H2-N chỉ khác nhau ở các key nêu trong §5.2.
   - Mọi rule ở §5.3 raise đúng.
9. **Không phá code cũ:** manifest `state_dict` và optimizer của các config cũ không đổi; toàn bộ test cũ vẫn pass.
10. **Metric phân tầng:** ví dụ tính tay cho `severe_error_rate`, `benign_vs_malignant_f1` và `node_accuracy`.
11. **Smoke end-to-end:** train 1 epoch trên dữ liệu giả cho `f1_flat_decoder`, `h1_hier_decoder`, `h2_hier_pecl`, `h2n_hier_name`, kèm cache tên nút giả. Resume được, predict có extras, predict không cần KB/cache/taxonomy.

---

## 9. README

Thêm:
- Bảng run B0 / F1 / H1 / H2 / H2-N và bốn phép so ở SOT §8.2.
- Cách tạo cache tên nút: `python scripts/cache_text_embeddings.py --taxonomy configs/taxonomy/btxrd_8class.yaml --out <NODE_CACHE_PT>`.
- Ghi chú rằng `logits` của nhánh query là log-xác suất của lá.
