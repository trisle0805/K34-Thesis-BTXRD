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
    size = data_cfg['input_size']
    aug = data_cfg['augmentation']
    items = [transforms.Lambda(load_skelex_rgb), transforms.Resize((size, size))]
    if training:
        items.append(transforms.RandomAffine(aug['degrees'], tuple(aug['translate']), tuple(aug['scale'])))
    return transforms.Compose(items + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _normalization(data_cfg):
    if data_cfg.get('preprocessing') == 'vit_gray3':
        return transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if data_cfg.get('preprocessing') == 'clip_gray3':
        return transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def build_train_transform(data_cfg):
    if data_cfg.get('preprocessing') == 'skelex_minmax_rgb':
        return _skelex_transform(data_cfg, True)
    aug = data_cfg["augmentation"] if isinstance(data_cfg, dict) else data_cfg.augmentation
    size = data_cfg["input_size"] if isinstance(data_cfg, dict) else data_cfg.input_size
    get = lambda obj, key: obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            *([transforms.RandomHorizontalFlip(aug.get("hflip_p", 0))] if aug.get("hflip_p", 0) else []),
            transforms.RandomAffine(
                degrees=get(aug, "degrees"),
                translate=tuple(get(aug, "translate")),
                scale=tuple(get(aug, "scale")),
            ),
            *([transforms.ColorJitter(brightness=aug.get("brightness", 0), contrast=aug.get("contrast", 0))] if aug.get("brightness", 0) or aug.get("contrast", 0) else []),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            _normalization(data_cfg),
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
            _normalization(data_cfg),
        ]
    )
