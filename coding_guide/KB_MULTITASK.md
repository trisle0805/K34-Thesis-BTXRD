# KB multitask

M1–M4 inherit K0 and retain its diagnosis path and initialization. The independent
80-query branch supervises KB65 literature and 15 anatomical locations. Auxiliary
gradients update the shared backbone; proposition scores never enter diagnosis.
Diagnosis uses exactly K0's unweighted `cross_entropy(logits, labels)`.
The fixed boolean 8x65 `membership` buffer defines each class's literature set.
Raw float32 `kb_class_scores` are logsumexp over member logits minus log set
cardinality, including shared propositions in every member class. Literature
supervision is class CE on these scores, never literature BCE. Every class must
have nonempty membership. History uses `loss_ce`, `loss_kb`, `loss_bce_loc`.

Location BCE excludes normal rows. A boolean `location_mask` selects only columns
with positive tumor annotations in the training fold, fitted once at build time.
Validation reuses it; evaluation restores it from the checkpoint without loading
training annotations. Resume recomputes and verifies it. Empty tumor batches or
empty masks yield zero. Zero weights skip the corresponding loss terms.
Run information lists `excluded_location_columns`.

Inputs: existing KB JSON/CLS cache and patient text/token cache, plus the tracked
location definition, its CLS cache, and location annotations. Annotation CSV columns
are the configured image ID column and L01–L15 (binary); every requested split
image must be covered. Unrequested missing/invalid annotation rows are ignored.
Normal rows must be all zero; tumor rows need at least one positive.

Example commands (replace data paths with local assets):

```powershell
python scripts/cache_proposition_queries.py --csv kb/location_propositions_v1.csv --out location_queries.pt --device cuda
python scripts/train.py --config configs/cv/m1_densenet169_kbmt.yaml --set paths.image_dir=images paths.splits_csv=fold0/splits.csv paths.output_dir=runs run.name=m1_fold0
python scripts/evaluate.py --config configs/cv/m1_densenet169_kbmt.yaml --checkpoint runs/m1_fold0/best.pt --out evaluation --set paths.image_dir=images paths.splits_csv=fold0/splits.csv paths.output_dir=runs
```

Set other input paths through `--set` when they differ from config defaults.
Resume uses the same command with `train.resume=true train.epochs=<new horizon>`.
Multitask resume/evaluation validates source hashes and embedded fixed buffers;
the original assets must remain accessible. Evaluation writes proposition scores
and raw Bx8 `kb_class_scores`, retaining Bx80 `scores`, `ids`, and `image_ids`.
Auxiliary `kb_set` reports argmax accuracy and macro-F1; location AUROC retains
valid-column and sample counts and uses tumors only. Membership and training
mask are verified against checkpoint buffers. Old multitask checkpoints using
the former loss definition are intentionally incompatible.
No online text encoder is loaded during training or prediction.

Public controls are `loss.lit_weight`, `loss.loc_weight`,
and `knowledge.location_query_cache`. Location annotations default to
`location_labels_8class.csv`; the definition defaults to the repository CSV.
M2 is `m2_densenet169_kbmt_shuffled.yaml`, M3 is
`m3_densenet169_kbmt_noloc.yaml`, M4 is `m4_densenet169_kbmt_nolit.yaml`.
Each directly inherits K0. Query caches record `encoder_name`, `max_length`,
`ids`, `texts`, and `csv_sha256`. Multitask score artifacts use `ids`.
M1 uses literature/location weights 1/1; M2 uses 1/1 with seed-0 permutation of
literature65 only; M3 uses 1/0; M4 uses 0/1. Location15 and membership never
shuffle. `run_info.knowledge_embedding_source_ids` records literature sources
only. Training protocol and pre-multitask architectures are unchanged.