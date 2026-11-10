"""Run against the actual baseline checkout to generate compatibility fixtures."""
import json
from pathlib import Path
import sys
import torch
from btxrd.config import load_config
from btxrd.models.model import BTXRDModel
from btxrd.constants import CLASS_NAMES


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for name, path in [
        ('b0', 'configs/b0_image.yaml'), ('b1', 'configs/b1_metadata.yaml'),
        ('b2', 'configs/b2_knowledge.yaml'), ('b3', 'configs/b3_contrastive.yaml'),
        ('a1', 'configs/ablations/a1_mean_aggregation.yaml'),
    ]:
        cfg = load_config(path, {f'paths.{key}': 'fixture' for key in (
            'image_dir', 'splits_csv', 'class_weights_csv', 'metadata_csv',
            'kb_json', 'text_embedding_cache', 'output_dir')})
        ids = [f'p{i}' for i in range(65)]
        classes = {cls: ids[i::8] for i, cls in enumerate(CLASS_NAMES)}
        model = BTXRDModel(cfg, torch.zeros(65, 768), classes, ids, pretrained=False)
        manifest = {'commit': '80db4436262106050fda45d13d4b27b1dc6d9003',
                    'state_dict': {key: list(value.shape) for key, value in model.state_dict().items()},
                    'parameters': [name for name, _ in model.named_parameters()]}
        (output / f'{name}_80db443.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()