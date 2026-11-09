"""Training entry point implementing the locked BTXRD protocol."""

from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from btxrd.data.dataset import build_dataloaders
from btxrd.knowledge.kb import load_kb
from btxrd.knowledge.text_encoder import load_text_embeddings
from btxrd.losses.classification import build_classification_loss
from btxrd.losses.contrastive import PrototypeContrastiveLoss
from btxrd.models.model import BTXRDModel
from btxrd.training.checkpoint import load_checkpoint, save_checkpoint
from btxrd.training.metrics import compute_metrics
from btxrd.utils.io import environment_info, git_commit, sha256_file, write_json
from btxrd.utils.seed import capture_rng_state, restore_rng_state, seed_everything
from btxrd.constants import CLASS_NAMES


def _prepare_run_dir(cfg) -> Path:
    run_dir = Path(cfg.paths.output_dir) / cfg.run.name
    if cfg.train.resume:
        if cfg.run.overwrite:
            raise ValueError("train.resume and run.overwrite cannot both be true")
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {run_dir / 'last.pt'}")
        return run_dir
    if run_dir.exists():
        if not cfg.run.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = run_dir.with_name(f"{run_dir.name}__old_{stamp}")
        suffix = 1
        while backup.exists():
            backup = run_dir.with_name(f"{run_dir.name}__old_{stamp}_{suffix}")
            suffix += 1
        run_dir.rename(backup)
    run_dir.mkdir(parents=True)
    return run_dir


def _build_model_inputs(cfg, checkpoint: dict[str, Any] | None = None):
    if not cfg.model.use_knowledge:
        return None, None, None, None
    if checkpoint is not None and "text_embeddings" in checkpoint.get("model_buffers", {}):
        embeddings = checkpoint["model_buffers"]["text_embeddings"]
        kb_ids = checkpoint.get("kb_ids")
        class_to_ids = checkpoint.get("class_to_ids")
        if kb_ids is None or (cfg.model.use_contrastive and class_to_ids is None):
            raise ValueError("Knowledge checkpoint is missing kb_ids/class_to_ids")
        return embeddings, kb_ids, class_to_ids, checkpoint.get("kb_sha256")
    kb = load_kb(cfg.paths.kb_json, cfg.knowledge.exclude_ids, CLASS_NAMES)
    embeddings = load_text_embeddings(cfg.paths.text_embedding_cache, kb)
    return embeddings, kb.ids, kb.class_to_ids, kb.sha256


def _run_epoch(model, loader, device, ce_loss, con_loss, cfg, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss_ce": 0.0, "loss_con": 0.0, "loss_total": 0.0}
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, leave=False, disable=len(loader) < 2):
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        metadata = batch.get("meta")
        if metadata is not None:
            metadata = metadata.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda" and cfg.train.amp
        context = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, context:
            output = model(image, metadata)
            loss_ce = ce_loss(output["logits"], labels)
            if cfg.model.use_contrastive:
                loss_con = con_loss(output["u"], getattr(model, "prototypes"), labels, cfg.loss.contrastive_temperature)
            else:
                loss_con = loss_ce.new_zeros(())
            loss_total = loss_ce + cfg.loss.contrastive_weight * loss_con
        if training:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        count = labels.shape[0]
        sample_count += count
        totals["loss_ce"] += float(loss_ce.detach()) * count
        totals["loss_con"] += float(loss_con.detach()) * count
        totals["loss_total"] += float(loss_total.detach()) * count
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(output["logits"].detach().argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise ValueError("Cannot run an epoch with an empty dataloader")
    metrics = compute_metrics(y_true, y_pred)
    return {**{key: value / sample_count for key, value in totals.items()}, **metrics}


def train(cfg) -> dict[str, Any]:
    started = time.time()
    seed_everything(cfg.seed)
    run_dir = _prepare_run_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RNG and DataLoader generator states must remain CPU ByteTensors. Model and
    # optimizer loading will move parameter-associated state to the target device.
    resume_checkpoint = load_checkpoint(run_dir / "last.pt", "cpu") if cfg.train.resume else None
    metadata_stats = resume_checkpoint.get("metadata_stats") if resume_checkpoint else None
    generator_state = resume_checkpoint.get("train_generator_state") if resume_checkpoint else None
    data = build_dataloaders(cfg, ("train", "val"), metadata_stats, generator_state)
    embeddings, kb_ids, class_to_ids, kb_sha256 = _build_model_inputs(cfg, resume_checkpoint)
    model = BTXRDModel(cfg, embeddings, class_to_ids, kb_ids).to(device)
    optimizer = Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.train.amp)
    ce_loss = build_classification_loss(data.class_weights, device)
    con_loss = PrototypeContrastiveLoss()

    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = -1
    early_counter = 0
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint["scaler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_val_macro_f1"])
        best_epoch = int(resume_checkpoint.get("best_epoch", -1))
        early_counter = int(resume_checkpoint.get("early_stopping_counter", 0))
        history = list(resume_checkpoint.get("history", []))
        restore_rng_state(resume_checkpoint["rng_state"])

    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    stopped_early = False
    for epoch in range(start_epoch, cfg.train.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = _run_epoch(model, data.loaders["train"], device, ce_loss, con_loss, cfg, optimizer, scaler)
        val_metrics = _run_epoch(model, data.loaders["val"], device, ce_loss, con_loss, cfg)
        row = {"epoch": epoch, "lr": lr}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = val_metrics["macro_f1"] > best_metric
        if improved:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch
            early_counter = 0
        else:
            early_counter += 1
        metadata_state = data.metadata_preprocessor.state_dict() if data.metadata_preprocessor else None
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_metrics": val_metrics,
            "best_val_macro_f1": best_metric,
            "best_epoch": best_epoch,
            "early_stopping_counter": early_counter,
            "history": history,
            "config": cfg.to_dict(),
            "metadata_stats": metadata_state,
            "kb_ids": kb_ids,
            "class_to_ids": class_to_ids,
            "kb_sha256": kb_sha256,
            "model_buffers": model.checkpoint_buffers(),
        }
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        scheduler.step()
        state["scheduler_state"] = scheduler.state_dict()
        state["rng_state"] = capture_rng_state()
        state["train_generator_state"] = data.train_generator.get_state() if data.train_generator else None
        save_checkpoint(run_dir / "last.pt", state)
        if cfg.train.early_stopping.enabled and early_counter >= cfg.train.early_stopping.patience:
            stopped_early = True
            break

    result = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
    }
    used_files = ["splits_csv", "class_weights_csv"]
    if cfg.model.use_metadata:
        used_files.append("metadata_csv")
    if cfg.model.use_knowledge:
        used_files.extend(["kb_json", "text_embedding_cache"])
    info = {
        **result,
        "duration_seconds": time.time() - started,
        "git_commit": git_commit(Path.cwd()),
        "environment": environment_info(),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "sha256": {name: sha256_file(getattr(cfg.paths, name)) for name in used_files},
    }
    write_json(run_dir / "run_info.json", info)
    return result
