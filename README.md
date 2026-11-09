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

In Colab, clone the GitHub repository on each runtime, install it in editable mode, and point `paths.output_dir` to a persistent Google Drive directory. Existing run folders are never deleted: with `run.overwrite=true`, the previous folder is renamed with an `__old_<timestamp>` suffix.

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
  --config configs/b2_knowledge.yaml \
  --set paths.kb_json=/path/kb.json \
        paths.text_embedding_cache=/path/text_embeddings.pt \
        paths.image_dir=/unused \
        paths.splits_csv=/unused \
        paths.class_weights_csv=/unused \
        paths.output_dir=/unused
```

The additional paths are configuration-schema requirements; the cache script does not read image or split data.

## Train

Example B2 run:

```bash
python scripts/train.py --config configs/b2_knowledge.yaml --set \
  paths.image_dir=/path/images \
  paths.splits_csv=/path/splits.csv \
  paths.class_weights_csv=/path/class_weights.csv \
  paths.kb_json=/path/kb.json \
  paths.text_embedding_cache=/path/text_embeddings.pt \
  paths.output_dir=/content/drive/MyDrive/BTXRD/runs
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
