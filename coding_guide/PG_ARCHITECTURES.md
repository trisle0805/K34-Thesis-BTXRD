# Proposition-guided architectures

New experiments inherit `configs/cv/_protocol.yaml`; existing configs remain unchanged.

- `k0_image_text_concat.yaml`: DenseNet GAP + masked patient token mean, projected to 256 with LayerNorm, then a linear eight-class classifier.
- `k1_pg_image.yaml`: 65 fixed MedCPT CLS queries attending to image tokens only.
- `k2_pg_image_text.yaml`: independent image and patient attention, concatenated proposition features.

PG uses eight 96-dimensional attention heads without query projections, shared proposition scoring, class-local learned attention (128 hidden units) or mean pooling, and a class bias. No auxiliary or contrastive losses are allowed.

`scripts/cache_patient_tokens.py` accepts `--text-csv`, `--output`, `--encoder`, `--max-length`, `--batch-size`, and `--device`. Defaults are MedCPT Query Encoder, length 64, batch 64, CPU. Only CSV columns `image_id` and `text` are read. Encoding is frozen, eval, and no-grad. Configure `paths.patient_text_csv`, `paths.patient_token_cache`, and `patient_text.max_length` consistently.

Cache loading checks dtype, dimensions, finite values, unique IDs, nonempty boolean masks, encoder, length, source CSV SHA256, and full requested split coverage before creating loaders. Cached token tensors are stored in float16, converted to float32 at the attention boundary.

Evaluation writes `propositions_<split>.npz` with `s`, `alpha`, `image_ids`, and `prop_ids`. Checkpoints retain Q and class membership A and reject inconsistent buffers on load. Embedding permutation continues to use the existing deterministic permutation mechanism. Run info includes per-module parameter totals/trainable counts and patient CSV/cache hashes.

Synthetic CPU tests do not download models or use real patient data. Existing checkpoint equality tests remain part of the regression suite.