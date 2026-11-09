from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from btxrd.config import load_config
from btxrd.data.dataset import build_dataloaders
from btxrd.models.model import BTXRDModel
from btxrd.training.checkpoint import save_checkpoint
from btxrd.training.predict import predict
from btxrd.training.trainer import _prepare_run_dir, train


class TinyImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.channel_scale = nn.Parameter(torch.linspace(0.5, 1.5, 1664))

    def forward(self, image: torch.Tensor):
        spatial = F.adaptive_avg_pool2d(image.mean(dim=1, keepdim=True), (7, 7))
        feature_map = spatial * self.channel_scale.view(1, -1, 1, 1)
        return feature_map, feature_map.mean(dim=(2, 3))


def test_dataloaders_fit_metadata_only_on_train(config_overrides):
    cfg = load_config("configs/b1_metadata.yaml", config_overrides())
    bundle = build_dataloaders(cfg, ("train", "val"))
    batch = next(iter(bundle.loaders["train"]))
    assert batch["image"].shape == (8, 3, 224, 224)
    assert batch["meta"].shape == (8, 9)
    stats = bundle.metadata_preprocessor.state_dict()
    assert stats["age_mean"] == pytest.approx(13.5)


def test_safe_run_directory_contract(config_overrides, tmp_path):
    cfg = load_config("configs/b0_image.yaml", config_overrides(output_dir=tmp_path, run_name="safe"))
    run_dir = _prepare_run_dir(cfg)
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_run_dir(cfg)

    overwrite_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(output_dir=tmp_path, run_name="safe", **{"run.overwrite": True}),
    )
    new_dir = _prepare_run_dir(overwrite_cfg)
    backups = list(tmp_path.glob("safe__old_*"))
    assert new_dir.is_dir() and len(backups) == 1
    assert (backups[0] / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_checkpoint_driven_prediction_without_kb_cache(config_overrides, tmp_path):
    training_cfg = load_config("configs/b2_knowledge.yaml", config_overrides())
    embeddings = torch.randn(68, 768)
    model = BTXRDModel(training_cfg, embeddings, pretrained=False)
    checkpoint_path = tmp_path / "b2.pt"
    save_checkpoint(checkpoint_path, {
        "epoch": 0,
        "model_state": model.state_dict(),
        "config": training_cfg.to_dict(),
        "best_val_macro_f1": 0.0,
        "metadata_stats": None,
        "kb_ids": [f"P{index:03d}" for index in range(1, 69)],
        "class_to_ids": None,
        "kb_sha256": "checkpoint-only",
        "model_buffers": model.checkpoint_buffers(),
    })
    runtime_overrides = config_overrides(
        **{"paths.kb_json": None, "paths.text_embedding_cache": None},
    )
    runtime_cfg = load_config(
        "configs/b2_knowledge.yaml",
        runtime_overrides,
        require_training_assets=False,
    )
    frame, extras = predict(runtime_cfg, checkpoint_path, split="test", return_extras=True)
    assert len(frame) == 8
    assert len([column for column in frame if column.startswith("p_")]) == 8
    assert extras["cross_attn_weights"].shape == (8, 68, 49)
    assert extras["aggregation_weights"].shape == (8, 68)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/b0_image.yaml",
        "configs/b1_metadata.yaml",
        "configs/b2_knowledge.yaml",
        "configs/b3_contrastive.yaml",
        "configs/ablations/a1_mean_aggregation.yaml",
    ],
)
def test_train_predict_smoke_all_experiments(monkeypatch, config_overrides, tmp_path, config_path):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", TinyImageEncoder)
    run_name = Path(config_path).stem
    overrides = config_overrides(
        output_dir=tmp_path,
        run_name=run_name,
        **{"train.epochs": 1, "train.amp": False},
    )
    cfg = load_config(config_path, overrides)
    result = train(cfg)
    run_dir = Path(result["run_dir"])
    assert {"config_resolved.yaml", "history.csv", "run_info.json", "best.pt", "last.pt"} <= {
        path.name for path in run_dir.iterdir()
    }
    prediction, extras = predict(cfg, run_dir / "best.pt", split="test", return_extras=True)
    assert prediction.shape == (8, 13)
    if cfg.model.use_knowledge:
        assert extras["cross_attn_weights"].shape == (8, 65, 49)
        if cfg.model.aggregation == "mean":
            assert extras["aggregation_weights"] is None
        else:
            assert extras["aggregation_weights"].shape == (8, 65)
    else:
        assert extras == {"aggregation_weights": None, "cross_attn_weights": None, "kb_ids": None}


def test_functional_resume_continues_epoch(monkeypatch, config_overrides, tmp_path):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", TinyImageEncoder)
    first_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(output_dir=tmp_path, run_name="resume", **{"train.epochs": 1, "train.amp": False}),
    )
    first = train(first_cfg)
    resumed_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(
            output_dir=tmp_path,
            run_name="resume",
            **{"train.epochs": 2, "train.resume": True, "train.amp": False},
        ),
    )
    resumed = train(resumed_cfg)
    checkpoint = torch.load(Path(resumed["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epochs_ran"] == 2
    assert checkpoint["epoch"] == 1
    assert checkpoint["scheduler_state"]["last_epoch"] == 2
    assert "rng_state" in checkpoint and "train_generator_state" in checkpoint

def test_resume_rejects_config_mismatch(monkeypatch, config_overrides, tmp_path):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", TinyImageEncoder)
    first_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(output_dir=tmp_path, run_name="mismatch", **{"train.epochs": 1, "train.amp": False}),
    )
    train(first_cfg)
    resumed_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(
            output_dir=tmp_path,
            run_name="mismatch",
            **{"train.epochs": 2, "train.resume": True, "train.amp": False, "train.lr": 2e-4},
        ),
    )
    with pytest.raises(ValueError, match=r"Resume config differs.*train\.lr"):
        train(resumed_cfg)

def test_resume_completed_run_returns_without_training(monkeypatch, config_overrides, tmp_path):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", TinyImageEncoder)
    first_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(output_dir=tmp_path, run_name="complete", **{"train.epochs": 1, "train.amp": False}),
    )
    first = train(first_cfg)
    resumed_cfg = load_config(
        "configs/b0_image.yaml",
        config_overrides(
            output_dir=tmp_path,
            run_name="complete",
            **{"train.epochs": 1, "train.resume": True, "train.amp": False},
        ),
    )
    monkeypatch.setattr(
        "btxrd.training.trainer.build_dataloaders",
        lambda *args, **kwargs: pytest.fail("completed resume must not build dataloaders"),
    )
    resumed = train(resumed_cfg)
    assert resumed == first

def test_resume_early_stopped_run_returns_without_training(monkeypatch, config_overrides, tmp_path):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", TinyImageEncoder)
    first_cfg = load_config(
        "configs/b1_metadata.yaml",
        config_overrides(output_dir=tmp_path, run_name="early-stopped", **{"train.epochs": 1, "train.amp": False}),
    )
    first = train(first_cfg)
    checkpoint_path = Path(first["run_dir"]) / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["early_stopping_counter"] = first_cfg.train.early_stopping.patience
    save_checkpoint(checkpoint_path, checkpoint)
    resumed_cfg = load_config(
        "configs/b1_metadata.yaml",
        config_overrides(
            output_dir=tmp_path,
            run_name="early-stopped",
            **{"train.epochs": 2, "train.resume": True, "train.amp": False},
        ),
    )
    monkeypatch.setattr(
        "btxrd.training.trainer.build_dataloaders",
        lambda *args, **kwargs: pytest.fail("early-stopped resume must not build dataloaders"),
    )
    resumed = train(resumed_cfg)
    assert resumed["epochs_ran"] == 1
    assert resumed["stopped_early"] is True
