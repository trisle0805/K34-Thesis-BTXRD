# V2 hierarchical query experiments

| Run | Decision rule | Query | PECL prototype |
|---|---|---|---|
| B0 | Image classifier | — | — |
| F1 | Flat shared-head decoder | Class proposition set | Off |
| H1 | Hierarchical decoder | Own-node propositions (subtree fallback) | Off |
| H2 | Hierarchical decoder | Own-node propositions | Subtree propositions |
| H2-N | Hierarchical decoder | Node names | Node names |

Configs: `f1_flat_decoder.yaml`, `h1_hier_decoder.yaml`, `h2_hier_pecl.yaml`,
and `h2n_hier_name.yaml`. The comparisons are B0→F1 (knowledge decoder),
F1→H1 (hierarchical decisions), H1→H2 (PECL), and H2-N→H2
(proposition content versus names).

Create the node-name cache with `python scripts/cache_text_embeddings.py --taxonomy configs/taxonomy/btxrd_8class.yaml --out <NODE_CACHE_PT>`
and set `paths.node_name_cache`. H2-N needs neither a KB nor a proposition cache.
The text encoder remains frozen; all node vectors are fixed checkpoint buffers.
For the query branch, `logits` are normalized **leaf log-probabilities**;
weighted CE, softmax probabilities, and argmax remain compatible.
Prediction reconstructs query models from checkpoint taxonomy and vectors;
external assets can be omitted. Explicit supplied assets must match the checkpoint.
Clinical hierarchy metrics do not change checkpoint selection (eight-class validation Macro-F1).
Benign/malignant F1 uses tumour-only ground truth and labels benign/malignant;
normal predictions count as false negatives. Unsupported node metrics are NaN
(JSON `null`). Legacy and F1 metrics use the repository clinical tree.
# R1 backbone and class-conditioned decoder

Install the optional `text` extra (`pip install -e .[text]`) for SKELEX.
Transformers is imported lazily; DenseNet experiments do not require it.

| Run | Config | Encoder | Knowledge |
| --- | --- | --- | --- |
| B0 | `configs/b0_image.yaml` | DenseNet-169 | None |
| B2 | `configs/b2_knowledge.yaml` | DenseNet-169 | Global learned aggregation |
| B3 | `configs/b3_contrastive.yaml` | DenseNet-169 | Global + contrastive |
| A1 | `configs/ablations/a1_mean_aggregation.yaml` | DenseNet-169 | Mean aggregation |
| S0 | `configs/s0_skelex_image.yaml` | SKELEX CLS | None |
| R1-D | `configs/r1_densenet_class_decoder.yaml` | DenseNet-169 | Masked class decoder |
| R1-S | `configs/r1_skelex_class_decoder.yaml` | SKELEX patches | Masked class decoder |

Compare R1-D against B0 and B2; compare R1-S against S0. These are
different backbone baselines, not a single controlled backbone comparison.

Automatic settings resolve after YAML inheritance and command-line overrides.
DenseNet retains Adam/StepLR, 100 epochs, batch 32. SKELEX uses AdamW,
layer-wise LR decay 0.75, base LR 5e-5, weight decay 0.05, physical batch
16, accumulation 2, gradient clipping 1.0, five warmup epochs and cosine
decay over 50 epochs without early stopping. Explicit YAML values override
automatic defaults. SKELEX preprocessing applies grayscale per-image min-max
before RGB conversion, resizing, train-only affine, and ImageNet normalization.

R1 adds masked class queries and residual knowledge logits with trainable
`knowledge_logit_scale` (initially 0.1, no weight decay). Evaluation attention
is normalized within each class's proposition set. Prediction extras include
`class_attn_weights`, `logits_img`, and `logits_kb`.

Checkpoints embed actual ViT configuration and weights, so prediction and
resume never download pretrained SKELEX weights. SKELEX resume locks the
original epoch horizon and training-loader length. Tiny offline encoder tests
use `model.pretrained=false`; this is not a replacement for the pretrained
Colab reproduction experiment. Historical compatibility fixtures record keys,
shapes, and parameter order generated from commit
`80db4436262106050fda45d13d4b27b1dc6d9003`.

# BTXRD 8-Class Classification

Reproducible PyTorch package for the five locked BTXRD experiments described in `thesis/SOURCE_OF_TRUTH_BTXRD_8CLASS_VI.md` and `coding_guide/CODING_GUIDE_BTXRD.md`.

## Experiments

| Config | Inputs / objective |
|---|---|
| `configs/b0_image.yaml` | DenseNet-169 image baseline |
| `configs/b1_metadata.yaml` | Image + train-fitted metadata |
| `configs/b2_knowledge.yaml` | Image + proposition cross-attention |
| `configs/b3_contrastive.yaml` | B2 + class-prototype contrastive loss |
| `configs/ablations/a1_mean_aggregation.yaml` | B2 with mean proposition aggregation |

All paths are supplied through YAML or dotted command-line overrides. No local or Google Drive path is embedded in the package.

## Installation

Core training and prediction:

```bash
pip install -e .
```

Text-cache generation and tests:

```bash
pip install -e ".[text,dev]"
pytest -q
```

In Colab, clone the GitHub repository on each runtime, install it in editable mode, and point `paths.output_dir` to persistent storage. Existing run folders are never deleted: with `run.overwrite=true`, the previous folder is renamed with an `__old_<timestamp>` suffix.

## Required data

The runtime paths are:

- `paths.image_dir`: grayscale PNG directory.
- `paths.splits_csv`: `image_id`, `class8`, and `split` columns.
- `paths.class_weights_csv`: exactly eight rows in the locked class order, with `class8` and `weight_sqrt_inv` columns.
- `paths.metadata_csv`: required by B1; includes labels/splits plus `age`, `gender`, `region`, and `view`.
- `paths.kb_json`: required to train B2/B3/A1 and to generate the text cache.
- `paths.text_embedding_cache`: required to train B2/B3/A1.
- `paths.output_dir`: durable experiment-output root.

Override column names through `data.columns` when necessary. Split counts are validated against `data.expected_split_counts`; set that key to `null` only for synthetic/debug data.

## Generate the MedCPT cache

The cache script encodes all 68 propositions in canonical `query_ids` order with `ncbi/MedCPT-Query-Encoder`, CLS pooling, and raw 768-dimensional embeddings:

```bash
python scripts/cache_text_embeddings.py \
  --kb <KB_JSON> \
  --out <CACHE_PT> \
  --model ncbi/MedCPT-Query-Encoder \
  --max-length 64 \
  --batch-size 32 \
  --device cuda
```

`--model`, `--max-length`, `--batch-size`, and `--device` are optional and default to the values shown above.

## Train

Example B2 run:

```bash
python scripts/train.py --config configs/b2_knowledge.yaml --set \
  paths.image_dir=/path/images \
  paths.splits_csv=/path/splits.csv \
  paths.class_weights_csv=/path/class_weights.csv \
  paths.kb_json=/path/kb.json \
  paths.text_embedding_cache=/path/text_embeddings.pt \
  paths.output_dir=<OUTPUT_ROOT>
```

For B1, also set `paths.metadata_csv`. Training reads only the train and validation splits and writes:

- `config_resolved.yaml`
- `history.csv`
- `run_info.json`
- `best.pt`
- `last.pt`

To continue an interrupted run, use the same run name and set `train.resume=true`. `last.pt` contains optimizer, scheduler, scaler, early-stopping, history, RNG, and train-loader generator state for functional continuation.

A small pipeline check can be requested with:

```bash
python scripts/train.py --config configs/b0_image.yaml --set \
  paths.image_dir=/path/images \
  paths.splits_csv=/path/splits.csv \
  paths.class_weights_csv=/path/class_weights.csv \
  paths.output_dir=/path/runs \
  debug.limit_per_split=16 train.epochs=1 data.num_workers=0
```

## Predict and evaluate

```bash
python scripts/evaluate.py \
  --config configs/b2_knowledge.yaml \
  --checkpoint /path/run/best.pt \
  --split test \
  --out /path/evaluation \
  --set paths.image_dir=/path/images \
        paths.splits_csv=/path/splits.csv \
        paths.class_weights_csv=/path/class_weights.csv \
        paths.output_dir=/path/runs
```

Knowledge prediction reconstructs the architecture and proposition buffers from the checkpoint, so `paths.kb_json` and `paths.text_embedding_cache` may remain unset. If provided, they are strictly verified against the checkpoint. Architectural settings in the runtime config must match the checkpoint; runtime settings such as data paths, batch size, workers, pin memory, and debug limit may differ.

The evaluation script writes `predictions_<split>.csv` and `metrics_<split>.json`. Undefined per-class AUROC values are represented as JSON `null`.

## Python API

```python
from btxrd.config import load_config
from btxrd.training.predict import predict
from btxrd.training.trainer import train

cfg = load_config("configs/b0_image.yaml", {
    "paths.image_dir": "/path/images",
    "paths.splits_csv": "/path/splits.csv",
    "paths.class_weights_csv": "/path/class_weights.csv",
    "paths.output_dir": "/path/runs",
})
result = train(cfg)
frame = predict(cfg, f"{result['run_dir']}/best.pt", split="test")
```

For knowledge-only prediction without KB/cache files, load with `require_training_assets=False`:

```python
cfg = load_config(
    "configs/b2_knowledge.yaml",
    runtime_overrides,
    require_training_assets=False,
)
frame, extras = predict(cfg, checkpoint_path, return_extras=True)
```

`extras` contains spatial cross-attention, learned aggregation weights when applicable, and checkpoint KB IDs. A1 returns `aggregation_weights=None`; B0/B1 return all attention extras and `kb_ids` as `None`.
