"""Backbone training recipes and optimizer-step scheduling."""
import math
import re
import torch


def build_optimizer(model, cfg):
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if cfg.train.recipe == 'densenet':
        ordinary = [p for n, p in parameters if n != 'knowledge_logit_scale']
        groups = [{'params': ordinary}]
        beta = [p for n, p in parameters if n == 'knowledge_logit_scale']
        if beta:
            groups.append({'params': beta, 'weight_decay': 0.0})
        return torch.optim.Adam(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                    betas=tuple(cfg.train.betas))
    layers = model.image_encoder.vit.config.num_hidden_layers
    groups = {}
    backbone_layer_ids = set()
    for name, parameter in parameters:
        layer = layers + 2
        if name.startswith('image_encoder.vit.'):
            if '.embeddings.' in name:
                layer = 0
            elif match := re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', name):
                layer = int(match.group(1)) + 1
            else:
                layer = layers + 1
            backbone_layer_ids.add(layer)
        decay = 0.0 if parameter.ndim <= 1 or name.endswith('.bias') or 'layernorm' in name else cfg.train.weight_decay
        group = groups.setdefault((layer, decay), {'params': [], 'layer_id': layer, 'weight_decay': decay,
            'lr': cfg.train.lr * cfg.train.layer_decay ** (layers + 2 - layer)})
        group['params'].append(parameter)
    if backbone_layer_ids != set(range(layers + 2)):
        raise ValueError(f'Invalid ViT LLRD layer ids: {sorted(backbone_layer_ids)}; expected {list(range(layers + 2))}')
    return torch.optim.AdamW(list(groups.values()), betas=tuple(cfg.train.betas))


def build_scheduler(optimizer, cfg, batches):
    if cfg.train.recipe == 'densenet':
        return torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.step_size, cfg.train.gamma)
    steps = math.ceil(batches / cfg.train.grad_accum)
    warmup = cfg.train.warmup_epochs * steps
    total = cfg.train.epochs * steps
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)