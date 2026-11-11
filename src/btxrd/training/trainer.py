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
from btxrd.losses.classification import branch_classification_loss
from btxrd.knowledge.permutation import permute_embeddings
from btxrd.losses.contrastive import PrototypeContrastiveLoss
from btxrd.models.model import BTXRDModel
from btxrd.training.checkpoint import load_checkpoint, save_checkpoint
from btxrd.training.metrics import compute_metrics
from btxrd.training.predict import _ARCHITECTURE_KEYS
from btxrd.utils.io import environment_info, git_commit, sha256_file, write_json
from btxrd.utils.seed import capture_rng_state, restore_rng_state, seed_everything
from btxrd.constants import CLASS_NAMES
from btxrd.training.optim import build_optimizer, build_scheduler
from btxrd.config import migrate_config

_RESUME_PROTOCOL_KEYS = (
    'train.recipe', 'train.grad_accum', 'train.clip_grad_norm',
    'train.warmup_epochs', 'train.layer_decay',
    "loss",
    "seed",
    "train.lr",
    "train.weight_decay",
    "train.step_size",
    "train.gamma",
    "train.amp",
    "train.early_stopping",
    "data.augmentation",
    "data.batch_size",
)
_RESUME_ALLOWED_DIFFERENCES = (
    "paths",
    "data.num_workers",
    "data.pin_memory",
    "train.epochs",
    "train.resume",
)

def _get_dotted(data: dict[str, Any], key: str) -> Any:
    value: Any = data
    for part in key.split("."):
        value = value[part]
    return value

def _flatten_config(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, dotted))
        else:
            flattened[dotted] = value
    return flattened

def _validate_resume_config(cfg, checkpoint_config: dict[str, Any]) -> None:
    checkpoint_config = migrate_config(checkpoint_config)
    runtime = cfg.to_dict()
    if cfg.model.backbone == 'skelex':
        runtime['model']['skelex']['vit_config'] = checkpoint_config['model']['skelex']['vit_config']
    if cfg.train.recipe in {'skelex', 'vit'} and runtime['train']['epochs'] != checkpoint_config['train']['epochs']:
        raise ValueError('Resume config differs from checkpoint at: train.epochs (SKELEX horizon is locked)')
    required_keys = (*_ARCHITECTURE_KEYS, *_RESUME_PROTOCOL_KEYS)
    required_mismatches = {
        key for key in required_keys
        if _get_dotted(runtime, key) != _get_dotted(checkpoint_config, key)
    }
    runtime_flat = _flatten_config(runtime)
    checkpoint_flat = _flatten_config(checkpoint_config)
    all_keys = runtime_flat.keys() | checkpoint_flat.keys()
    disallowed_mismatches = {
        key for key in all_keys
        if runtime_flat.get(key) != checkpoint_flat.get(key)
        and not any(key == allowed or key.startswith(f"{allowed}.") for allowed in _RESUME_ALLOWED_DIFFERENCES)
    }
    mismatches = sorted(required_mismatches | disallowed_mismatches)
    if mismatches:
        raise ValueError(f"Resume config differs from checkpoint at: {', '.join(mismatches)}")

def _completed_resume_result(cfg, checkpoint: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    history = list(checkpoint.get("history", []))
    next_epoch = int(checkpoint["epoch"]) + 1
    early_counter = int(checkpoint.get("early_stopping_counter", 0))
    stopped_early = (
        cfg.train.early_stopping.enabled
        and early_counter >= cfg.train.early_stopping.patience
    )
    if not stopped_early and next_epoch < cfg.train.epochs:
        return None
    return {
        "run_dir": str(run_dir),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
    }


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
    if not (cfg.model.use_knowledge or cfg.get_path('model.use_pecl', False)):
        return None, None, None, None
    if checkpoint is not None and "text_embeddings" in checkpoint.get("model_buffers", {}):
        embeddings = checkpoint["model_buffers"]["text_embeddings"]
        kb_ids = checkpoint.get("kb_ids")
        class_to_ids = checkpoint.get("class_to_ids")
        needs_mapping = (cfg.model.use_contrastive or cfg.get_path('model.use_pecl', False)
                         or cfg.get_path('model.knowledge_decoder', 'global') == 'class_conditioned')
        if kb_ids is None or (needs_mapping and class_to_ids is None):
            raise ValueError("Knowledge checkpoint is missing kb_ids/class_to_ids")
        return embeddings, kb_ids, class_to_ids, checkpoint.get("kb_sha256")
    kb = load_kb(cfg.paths.kb_json, cfg.knowledge.exclude_ids, CLASS_NAMES)
    embeddings = load_text_embeddings(
        cfg.paths.text_embedding_cache,
        kb,
        cfg.knowledge.text_encoder_name,
    )
    embeddings, _ = permute_embeddings(embeddings, kb.ids, cfg.knowledge.embedding_permutation_seed)
    return embeddings, kb.ids, kb.class_to_ids, kb.sha256


def _run_epoch(model, loader, device, ce_loss, con_loss, cfg, optimizer=None, scaler=None, scheduler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss_ce": 0.0, "loss_con": 0.0, "loss_total": 0.0}
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    branch_predictions = {key: [] for key in ('img', 'kb', 'avg')}
    accum = cfg.train.grad_accum
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(loader, leave=False, disable=len(loader) < 2), 1):
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        metadata = batch.get("meta")
        if metadata is not None:
            metadata = metadata.to(device, non_blocking=True)
        amp_enabled = device.type == "cuda" and cfg.train.amp
        context = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, context:
            output = model(image, metadata)
            supervised, loss_ce, branch_ce = branch_classification_loss(
                output, labels, ce_loss, cfg.loss.aux_image_weight, cfg.loss.aux_knowledge_weight)
            if cfg.model.use_contrastive:
                loss_con = con_loss(output["u"], getattr(model, "prototypes"), labels, cfg.loss.contrastive_temperature)
            else:
                loss_con = loss_ce.new_zeros(())
            loss_total = supervised + cfg.loss.contrastive_weight * loss_con
            if cfg.get_path('model.use_pecl', False):
                loss_pecl = torch.nn.functional.cross_entropy(
                    output['pecl_u'] @ model.pecl_prototypes.T / cfg.loss.pecl_temperature, labels)
                loss_total = loss_total + cfg.loss.pecl_weight * loss_pecl
            else:
                loss_pecl = None
        if training:
            scaler.scale(loss_total / accum).backward()
            if batch_index % accum == 0 or batch_index == len(loader):
                if cfg.train.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        count = labels.shape[0]
        if loss_pecl is not None:
            totals.setdefault('loss_pecl', 0.0)
            totals['loss_pecl'] += float(loss_pecl.detach()) * count
        sample_count += count
        totals["loss_ce"] += float(loss_ce.detach()) * count
        if branch_ce:
            totals.setdefault('loss_ce_final', 0.0)
            totals['loss_ce_final'] += float(loss_ce.detach()) * count
            for key, value in branch_ce.items():
                totals.setdefault(f'loss_ce_{key}', 0.0)
                totals[f'loss_ce_{key}'] += float(value.detach()) * count
            if not training:
                logits = {key: output[f'logits_{key}'].detach() for key in ('img', 'kb')}
                logits['avg'] = (logits['img'] + logits['kb']) / 2
                for key, value in logits.items():
                    branch_predictions[key].extend(value.argmax(1).cpu().tolist())
        totals["loss_con"] += float(loss_con.detach()) * count
        totals["loss_total"] += float(loss_total.detach()) * count
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(output["logits"].detach().argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise ValueError("Cannot run an epoch with an empty dataloader")
    metrics = compute_metrics(y_true, y_pred)
    for key, predictions in branch_predictions.items():
        if predictions:
            metrics[f'macro_f1_{key}'] = compute_metrics(y_true, predictions)['macro_f1']
    return {**{key: value / sample_count for key, value in totals.items()}, **metrics}


def train(cfg) -> dict[str, Any]:
    started = time.time()
    seed_everything(cfg.seed)
    run_dir = _prepare_run_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RNG and DataLoader generator states must remain CPU ByteTensors. Model and
    # optimizer loading will move parameter-associated state to the target device.
    resume_checkpoint = load_checkpoint(run_dir / "last.pt", "cpu") if cfg.train.resume else None
    if resume_checkpoint:
        _validate_resume_config(cfg, resume_checkpoint["config"])
        completed_result = _completed_resume_result(cfg, resume_checkpoint, run_dir)
        if completed_result is not None:
            return completed_result
    metadata_stats = resume_checkpoint.get("metadata_stats") if resume_checkpoint else None
    generator_state = resume_checkpoint.get("train_generator_state") if resume_checkpoint else None
    data = build_dataloaders(cfg, ("train", "val"), metadata_stats, generator_state)
    if resume_checkpoint and cfg.train.recipe in {'skelex', 'vit'} and resume_checkpoint.get('train_batches') != len(data.loaders['train']):
        raise ValueError('SKELEX resume requires unchanged train loader steps')
    embeddings, kb_ids, class_to_ids, kb_sha256 = _build_model_inputs(cfg, resume_checkpoint)
    if resume_checkpoint and cfg.model.backbone == 'skelex':
        from btxrd.config import Config
        saved = cfg.to_dict()
        saved['model']['skelex']['vit_config'] = resume_checkpoint['vit_config']
        cfg = Config(saved, cfg.source_path)
    model = BTXRDModel(cfg, embeddings, class_to_ids, kb_ids, pretrained=False if resume_checkpoint else None).to(device)
    actual_config = cfg.to_dict()
    if cfg.model.backbone == 'skelex':
        actual_config['model']['skelex']['vit_config'] = model.image_encoder.vit.config.to_dict()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(data.loaders['train']))
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

    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(actual_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    stopped_early = False
    for epoch in range(start_epoch, cfg.train.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = _run_epoch(model, data.loaders["train"], device, ce_loss, con_loss, cfg, optimizer, scaler,
                       scheduler if cfg.train.recipe in {'skelex', 'vit'} else None)
        val_metrics = _run_epoch(model, data.loaders["val"], device, ce_loss, con_loss, cfg)
        row = {"epoch": epoch, "lr": lr}
        if cfg.train.recipe in {'skelex', 'vit'}:
            row['lr'] = max(optimizer.param_groups, key=lambda group: group['layer_id'])['lr']
        if hasattr(model, 'knowledge_logit_scale'):
            row['knowledge_logit_scale'] = float(model.knowledge_logit_scale.detach())
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
            "config": actual_config,
            "metadata_stats": metadata_state,
            "kb_ids": kb_ids,
            "class_to_ids": class_to_ids,
            "kb_sha256": kb_sha256,
            "model_buffers": model.checkpoint_buffers(),
            "vit_config": model.image_encoder.vit.config.to_dict() if cfg.model.backbone == 'skelex' else None,
            'train_batches': len(data.loaders['train']),
        }
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        if cfg.train.recipe == 'densenet':
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
    used_files = ["splits_csv"]
    if cfg.loss.class_weighting == "csv":
        used_files.append("class_weights_csv")
    if cfg.model.use_metadata:
        used_files.append("metadata_csv")
    if cfg.model.use_knowledge or cfg.model.use_pecl:
        used_files.extend(["kb_json", "text_embedding_cache"])
    info = {
        'knowledge_permuted': bool((cfg.model.use_knowledge or cfg.model.use_pecl) and cfg.knowledge.embedding_permutation_seed is not None),
        'knowledge_embedding_source_ids': (permute_embeddings(
            embeddings, kb_ids, cfg.knowledge.embedding_permutation_seed)[1] if kb_ids else {}),
        **data.balance_info,
        **result,
        "duration_seconds": time.time() - started,
        "git_commit": git_commit(Path(__file__).resolve()),
        "environment": environment_info(),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "sha256": {name: sha256_file(getattr(cfg.paths, name)) for name in used_files},
    }
    write_json(run_dir / "run_info.json", info)
    return result
