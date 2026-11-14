"""Generate predictions and classification metrics for one split."""

from __future__ import annotations

import argparse
import numpy as np
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
    frame, extras = predict(
        load_config(args.config, args.set, require_training_assets=False),
        args.checkpoint,
        args.split,
        return_extras=True,
    )
    branch_outputs = {key: extras[key] for key in ('logits_img', 'logits_kb', 'beta') if key in extras}
    if branch_outputs:
        np.savez(output_dir / f'branch_logits_{args.split}.npz', **branch_outputs)
    frame.to_csv(output_dir / f"predictions_{args.split}.csv", index=False)
    if 'proposition_scores' in extras:
        np.savez(output_dir / f'propositions_{args.split}.npz', s=extras['proposition_scores'],
                 alpha=extras['class_pool_weights'], image_ids=frame.image_id.to_numpy(dtype=str),
                 prop_ids=np.asarray(extras['kb_ids'], dtype=str))
    probabilities = frame[[f"p_{name}" for name in CLASS_NAMES]].to_numpy()
    metrics = compute_metrics(frame["y_true"], frame["y_pred"], probabilities)
    write_json(output_dir / f"metrics_{args.split}.json", metrics)


if __name__ == "__main__":
    main()
