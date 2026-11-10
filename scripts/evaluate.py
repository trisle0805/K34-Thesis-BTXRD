"""Generate predictions and classification metrics for one split."""

from __future__ import annotations

import argparse
from pathlib import Path

from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.training.metrics import compute_metrics
from btxrd.training.predict import predict
from btxrd.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = predict(
        load_config(args.config, args.set, require_training_assets=False),
        args.checkpoint,
        args.split,
    )
    frame.to_csv(output_dir / f"predictions_{args.split}.csv", index=False)
    probabilities = frame[[f"p_{name}" for name in CLASS_NAMES]].to_numpy()
    metrics = compute_metrics(frame["y_true"], frame["y_pred"], probabilities)
    from btxrd.training.metrics import hierarchical_metrics, clinical_taxonomy
    from btxrd.training.checkpoint import load_checkpoint
    from btxrd.knowledge.taxonomy import Taxonomy
    checkpoint = load_checkpoint(args.checkpoint, 'cpu')
    saved = checkpoint.get('taxonomy')
    taxonomy = Taxonomy(**saved) if saved and 'tumour' in saved['node_ids'] else clinical_taxonomy()
    metrics.update(hierarchical_metrics(frame['y_true'], frame['y_pred'], taxonomy))
    metrics['clinical_taxonomy_sha256'] = taxonomy.sha256
    write_json(output_dir / f"metrics_{args.split}.json", metrics)


if __name__ == "__main__":
    main()
