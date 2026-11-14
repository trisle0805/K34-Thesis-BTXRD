"""Independent proposition supervision alongside the unchanged K0 diagnosis head."""
import torch
from torch import nn
from btxrd.constants import CLASS_NAMES
from btxrd.models.proposition_guided import FixedQueryAttention, validate_text


NORMAL_INDEX = next(i for i, name in enumerate(CLASS_NAMES) if name.lower() == 'normal')

def kb_class_scores(scores, membership):
    """Float32 log-mean-exp over each class's (possibly shared) propositions."""
    values = scores[:, :65].float().unsqueeze(1)
    return values.masked_fill(~membership.unsqueeze(0), -torch.inf).logsumexp(-1) - membership.sum(-1).float().log()

def train_location_mask(dataset):
    """Fit only on the training dataset, never validation or prediction rows."""
    frame = dataset.frame
    tumor = frame[dataset.columns['label']].ne(CLASS_NAMES[NORMAL_INDEX])
    values = [dataset.locations[item] for item in frame.loc[tumor, dataset.columns['image_id']].astype(str)]
    return torch.stack(values).sum(0).gt(0) if values else torch.zeros(15, dtype=torch.bool)


class PropositionBranch(nn.Module):
    def __init__(self, queries, class_to_ids, ids):
        super().__init__()
        if queries is None or queries.shape != (80, 768) or not torch.isfinite(queries).all():
            raise ValueError('kb_multitask requires finite 80x768 queries')
        if len(ids) != 80 or len(set(ids)) != 80:
            raise ValueError('Expected 80 unique proposition IDs')
        self.register_buffer('Q', queries.detach().float().clone())
        self.register_buffer('membership', torch.tensor([
            [item in class_to_ids[name] for item in ids[:65]] for name in CLASS_NAMES
        ], dtype=torch.bool))
        if not self.membership.any(1).all():
            raise ValueError('Every class requires at least one literature proposition')
        self.register_buffer('location_mask', torch.zeros(15, dtype=torch.bool))
        self.image_attention = FixedQueryAttention(1664)
        self.text_attention = FixedQueryAttention(768)
        self.scorer = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, image, tokens, mask):
        validate_text(tokens, mask)
        visual, attn_img = self.image_attention(self.Q, image)
        text, attn_txt = self.text_attention(self.Q, tokens.float(), mask)
        scores = self.scorer(torch.cat((visual, text), -1)).squeeze(-1)
        return dict(proposition_scores=scores, kb_class_scores=kb_class_scores(scores, self.membership),
                attn_img=attn_img, attn_txt=attn_txt)


def multitask_loss(output, labels, branch, locations, cfg):
    from torch.nn import functional as F
    logits = output['logits']
    ce = F.cross_entropy(logits, labels)
    zero = logits.new_zeros(())
    lit = F.cross_entropy(output['kb_class_scores'], labels) if cfg.lit_weight else zero
    tumor = labels != NORMAL_INDEX
    loc = zero
    if cfg.loc_weight and tumor.any() and branch.location_mask.any():
        loc = F.binary_cross_entropy_with_logits(output['proposition_scores'][tumor, 65:][:, branch.location_mask].float(), locations[tumor][:, branch.location_mask].float())
    total = ce
    if cfg.lit_weight:
        total = total + cfg.lit_weight * lit
    if cfg.loc_weight:
        total = total + cfg.loc_weight * loc
    return total, dict(loss_kb=lit, loss_bce_loc=loc)