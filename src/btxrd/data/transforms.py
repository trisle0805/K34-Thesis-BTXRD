"""Image transforms matching the locked B0/B1 protocol."""

from __future__ import annotations

from torchvision import transforms

from btxrd.constants import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(data_cfg):
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
    size = data_cfg["input_size"] if isinstance(data_cfg, dict) else data_cfg.input_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
