# KB multitask

M1–M4 inherit K0 and retain its diagnosis path and initialization. The independent
80-query branch supervises KB65 literature and 15 anatomical locations. Auxiliary
gradients update the shared backbone; proposition scores never enter diagnosis.
Normal has one-hot targets; other classes distribute epsilon over off-diagonal
Jaccard similarities, falling back to one-hot when none overlap.

Inputs: existing KB JSON/CLS cache and patient text/token cache, plus the tracked
location definition, its CLS cache, and location annotations. Annotation CSV columns
are the configured image ID column and L01–L15 (binary); every split image must be
covered. Normal rows must be all zero; tumor rows need at least one positive.

Example commands (replace data paths with local assets):

```powershell
python scripts/cache_location_embeddings.py --csv kb/location_propositions_v1.csv --out location_embeddings.pt
python scripts/train.py --config configs/cv/m1_densenet169_kb_multitask.yaml --set paths.image_dir=images paths.splits_csv=splits.csv paths.output_dir=runs
python scripts/evaluate.py --config configs/cv/m1_densenet169_kb_multitask.yaml --checkpoint runs/m1_densenet169_kb_multitask/best.pt --out evaluation --set paths.image_dir=images paths.splits_csv=splits.csv paths.output_dir=runs
```

Set other input paths through `--set` when they differ from config defaults.
Resume uses the same command with `train.resume=true train.epochs=<new horizon>`.
Multitask resume/evaluation validates source hashes and embedded fixed buffers;
the original assets must remain accessible. Evaluation writes proposition scores
and columnwise AUROC, with valid-column and sample counts (location: tumors only).
No online text encoder is loaded during training or prediction.