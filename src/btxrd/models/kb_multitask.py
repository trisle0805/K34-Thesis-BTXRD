"""Independent proposition supervision alongside the unchanged K0 diagnosis head."""
import torch
from torch import nn
from btxrd.constants import CLASS_NAMES
from btxrd.models.proposition_guided import FixedQueryAttention, validate_text


def soft_targets(class_to_ids, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be in [0, 1]')
    sets = [set(class_to_ids[name]) for name in CLASS_NAMES]
    matrix = torch.eye(len(sets))
    for i, left in enumerate(sets):
        if CLASS_NAMES[i].lower() == 'normal':
            continue
        similarities = torch.tensor([
            len(left & right) / len(left | right) if i != j and left | right else 0.
            for j, right in enumerate(sets)])
        if similarities.sum() > 0:
            matrix[i] = (1 - epsilon) * matrix[i] + epsilon * similarities / similarities.sum()
    return matrix


class PropositionBranch(nn.Module):
    def __init__(self, queries, class_to_ids, ids, epsilon):
        super().__init__()
        if queries is None or queries.shape != (80, 768) or not torch.isfinite(queries).all():
            raise ValueError('kb_multitask requires finite 80x768 queries')
        if len(ids) != 80 or len(set(ids)) != 80:
            raise ValueError('Expected 80 unique proposition IDs')
        self.register_buffer('Q', queries.detach().float().clone())
        self.register_buffer('soft_targets', soft_targets(class_to_ids, epsilon))
        self.register_buffer('literature_targets', torch.tensor([
            [item in class_to_ids[name] for item in ids[:65]] for name in CLASS_NAMES
        ], dtype=torch.float32))
        self.image_attention = FixedQueryAttention(1664)
        self.text_attention = FixedQueryAttention(768)
        self.scorer = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, image, tokens, mask):
        validate_text(tokens, mask)
        visual, _ = self.image_attention(self.Q, image)
        text, _ = self.text_attention(self.Q, tokens.float(), mask)
        return self.scorer(torch.cat((visual, text), -1)).squeeze(-1)


def multitask_loss(output, labels, branch, locations, cfg):
    from torch.nn import functional as F
    logits = output['logits']
    soft = -(branch.soft_targets[labels] * logits.log_softmax(-1)).sum(-1).mean()
    hard = F.cross_entropy(logits, labels)
    zero = logits.new_zeros(())
    lit = F.binary_cross_entropy_with_logits(output['proposition_scores'][:, :65], branch.literature_targets[labels]) if cfg.literature_weight else zero
    loc = F.binary_cross_entropy_with_logits(output['proposition_scores'][:, 65:], locations) if cfg.location_weight else zero
    total = soft + cfg.literature_weight * lit + cfg.location_weight * loc
    return total, dict(loss_soft_ce=soft, loss_hard_ce=hard, loss_literature_bce=lit, loss_location_bce=loc)