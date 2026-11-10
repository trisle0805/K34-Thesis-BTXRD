"""Image transforms matching the locked B0/B1 protocol."""

from __future__ import annotations

from torchvision import transforms
import numpy as np
from PIL import Image

from btxrd.constants import IMAGENET_MEAN, IMAGENET_STD

def load_skelex_rgb(image):
    array = np.asarray(image.convert('L'), dtype=np.float32)
    lo, hi = array.min(), array.max()
    array = ((array - lo) / (hi - lo) * 255 if np.isfinite(lo) and np.isfinite(hi) and hi > lo
             else np.zeros_like(array))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert('RGB')

def _skelex_transform(data_cfg, training):
    items = [transforms.Lambda(load_skelex_rgb), transforms.Resize((224, 224))]
    if training:
        items.append(transforms.RandomAffine(7, (0.03, 0.03), (0.95, 1.05)))
    return transforms.Compose(items + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_train_transform(data_cfg):
    if data_cfg.get('preprocessing') == 'skelex_minmax_rgb':
        return _skelex_transform(data_cfg, True)
    aug = data_cfg["augmentation"] if isinstance(data_cfg, dict) else data_cfg.augmentation
    size = data_cfg["input_size"] if isinstance(data_cfg, dict) else data_cfg.input_size
    get = lambda obj, key: obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomAffine(
                degrees=get(aug, "degrees"),
                translate=tuple(get(aug, "translate")),
                scale=tuple(get(aug, "scale")),
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(data_cfg):
    if data_cfg.get('preprocessing') == 'skelex_minmax_rgb':
        return _skelex_transform(data_cfg, False)
    size = data_cfg["input_size"] if isinstance(data_cfg, dict) else data_cfg.input_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
