"""Build a reusable full-KB MedCPT embedding cache."""

from __future__ import annotations

import argparse

from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.kb import load_kb
from btxrd.knowledge.text_encoder import encode_texts, save_text_embedding_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--kb')
    inputs.add_argument('--taxonomy')
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.taxonomy:
        from pathlib import Path
        import torch
        from btxrd.knowledge.taxonomy import load_taxonomy
        taxonomy = load_taxonomy(args.taxonomy, CLASS_NAMES)
        texts = [taxonomy.names[n] for n in taxonomy.node_ids]
        embeddings = encode_texts(texts, args.model, args.max_length, args.device, args.batch_size)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(node_ids=taxonomy.node_ids, texts=texts, embeddings=embeddings.float(),
                        model_name=args.model, max_length=args.max_length, pooling='cls', taxonomy_sha256=taxonomy.sha256), args.out)
        return
    kb = load_kb(args.kb, [], CLASS_NAMES)
    embeddings = encode_texts(kb.all_texts, args.model, args.max_length, args.device, args.batch_size)
    save_text_embedding_cache(args.out, kb, embeddings, args.model, args.max_length)


if __name__ == "__main__":
    main()
