# Proposition-guided architectures

New experiments inherit `configs/cv/_protocol.yaml`; existing configs remain unchanged.

- `k0_densenet169_imgtxt.yaml`: DenseNet GAP + masked text token mean, projected to 256 with LayerNorm, then a linear eight-class classifier.
- `k1_densenet169_pg.yaml`: 65 fixed MedCPT CLS queries with independent image and text attention, concatenated proposition features, correct KB.
- `k2_densenet169_pg_shuffled.yaml`: inherits k1 unchanged except `knowledge.embedding_permutation_seed: 0`. Both PG parameter counts are identical.

PG uses eight 96-dimensional attention heads without query projections, shared proposition scoring, class-local learned attention (128 hidden units) or mean pooling, and a class bias. No auxiliary or contrastive losses are allowed.

`scripts/cache_text_tokens.py` accepts `--text-csv`, `--out`, `--encoder`, `--max-length`, `--batch-size`, and `--device`. Defaults are MedCPT Query Encoder, length 64, batch 64, CPU. Only CSV columns `image_id` and `text` are read. Encoding is frozen, eval, and no-grad. Configure `paths.text_csv`, `paths.text_token_cache`, `text.encoder_name`, and `text.max_length` consistently.

Cache loading checks dtype, dimensions, finite values, unique IDs, nonempty boolean masks, encoder, length, source CSV SHA256, and full requested split coverage before creating loaders. Cached token tensors are stored in float16, converted to float32 at the attention boundary.

Evaluation writes `evaluation/proposition_scores_<split>.npz` with `s`, `alpha`, `image_ids`, and `prop_ids` when the evaluation output directory is `evaluation`. Checkpoints retain Q and class membership A and reject inconsistent buffers on load. Embedding permutation continues to use the existing deterministic permutation mechanism. Run info includes architecture, use_text_branch, pool, backbone/image attention/text attention/scorer/pooling/bias totals and trainable counts, and SHA256 hashes of all used assets.

Synthetic CPU tests do not download models or use real patient data. Existing checkpoint equality tests remain part of the regression suite.