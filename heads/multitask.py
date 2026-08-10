import torch
from torch import nn
import torch.nn.functional as F

# Allowed destinations for the native S0-S3 lifecycle.  Recovery/termination
# is represented as a censored/non-transition outcome rather than S3->S0.
TRANSITION_MASK = torch.tensor([
    [0, 1, 0, 0],  # S0 -> S1
    [0, 0, 1, 0],  # S1 -> S2
    [0, 0, 0, 1],  # S2 -> S3
    [0, 0, 0, 0],  # S3 -> explicit recovery/termination (censored here)
], dtype=torch.bool)


class MultiTaskHeads(nn.Module):
    """Shared current-stage, transition and duration heads.

    The transition head is masked using the model's *predicted* current stage at
    inference. During training, callers may provide the ground-truth stage to
    avoid compounding an early classification error into the transition loss.
    """
    def __init__(self, d, horizon=64):
        super().__init__()
        self.stage = nn.Linear(d, 4)
        self.next = nn.Linear(d, 4)
        self.sojourn = nn.Linear(d, 2)
        self.register_buffer("transition_mask", TRANSITION_MASK, persistent=False)

    def _mask_next(self, next_logits, stage_logits, stage_target=None):
        stage = stage_target if stage_target is not None else stage_logits.argmax(-1)
        mask = self.transition_mask.to(next_logits.device)[stage]
        # S3 has no ordinary next-stage destination; leave all logits finite so
        # the caller can represent termination/censoring separately.
        s3 = stage == 3
        mask[s3] = True
        return next_logits.masked_fill(~mask, -1e4)

    def forward(self, z, stage_target=None):
        stage_logits = self.stage(z)
        raw_next = self.next(z)
        next_logits = self._mask_next(raw_next, stage_logits, stage_target)
        return {
            "stage_logits": stage_logits,
            "next_stage_logits": next_logits,
            "sojourn_params": self.sojourn(z),
            "latent_embedding": z,
        }


def ordinal_distance_penalty(logits, target):
    """Penalize errors more strongly as they move farther from the true stage."""
    probs = F.softmax(logits, dim=-1)
    stages = torch.arange(4, device=logits.device, dtype=probs.dtype)
    expected = (probs * stages).sum(-1)
    return torch.abs(expected - target.float()).mean()


def rule_consistency_loss(stage_logits, x, feature_names):
    """Soft rule-grounding regularizer.

    This is deliberately weak: it nudges stage probabilities toward the native
    feature signatures without turning the simulator's feature rules into labels.
    It is only active when the relevant columns exist.
    """
    if not feature_names:
        return stage_logits.new_tensor(0.0)
    idx = {n: i for i, n in enumerate(feature_names)}
    scores = torch.zeros(stage_logits.size(0), 4, device=stage_logits.device)
    def add(stage, names, sign=1.0):
        vals = [x[:, idx[n]].abs() * sign for n in names if n in idx]
        if vals:
            scores[:, stage] += torch.stack(vals, -1).mean(-1)
    add(1, [n for n in ("grad_H", "grad_G", "grad_I", "rate_H", "rate_G", "rate_I") if n in idx])
    add(2, [n for n in ("entropy", "jsd") if n in idx])
    add(3, [n for n in ("qdrm_H", "qdrm_G", "qdrm_I", "corr_H", "corr_G", "corr_I") if n in idx])
    p = F.softmax(stage_logits, -1)
    target = F.softmax(scores, -1)
    return F.kl_div(torch.log(p + 1e-8), target, reduction="batchmean")


def common_loss(out, batch, feature_names=None, lambdas=None):
    lambdas = lambdas or {"stage": 1.0, "next": 1.0, "time": 0.2, "ord": 0.1, "rule": 0.05, "deephit": 0.5}
    stage_loss = F.cross_entropy(out["stage_logits"], batch["stage"])
    mask = batch["next_stage"] >= 0
    if mask.any():
        next_loss = F.cross_entropy(out["next_stage_logits"][mask], batch["next_stage"][mask])
        pred_mean = F.softplus(out["sojourn_params"][:, 0]) + 1e-3
        time_loss = F.smooth_l1_loss(pred_mean[mask], batch["sojourn"][mask])
    else:
        next_loss = stage_loss.new_tensor(0.)
        time_loss = stage_loss.new_tensor(0.)
    ord_loss = ordinal_distance_penalty(out["stage_logits"], batch["stage"])
    rule_loss = rule_consistency_loss(out["stage_logits"], batch["x"][:, -1], feature_names or [])

    hit = stage_loss.new_tensor(0.)
    if "event_pmf" in out and mask.any():
        # Proper event-type/time mass: [batch, event_type, time].
        pmf = torch.softmax(out["event_pmf"].reshape(out["event_pmf"].size(0), -1), -1)
        B, _, H = out["event_pmf"].shape
        t = batch["sojourn"].long().clamp(1, H) - 1
        idx = batch["next_stage"].clamp(0, 3) * H + t
        hit = -torch.log(pmf[torch.arange(B, device=pmf.device)[mask], idx[mask]] + 1e-8).mean()

    total = (lambdas["stage"] * stage_loss + lambdas["next"] * next_loss +
             lambdas["time"] * time_loss + lambdas["ord"] * ord_loss +
             lambdas["rule"] * rule_loss + lambdas["deephit"] * hit)
    return total, {
        "stage": float(stage_loss.detach()), "next": float(next_loss.detach()),
        "time": float(time_loss.detach()), "ordinal": float(ord_loss.detach()),
        "rule": float(rule_loss.detach()), "deephit": float(hit.detach())
    }


def decode_output(out):
    sp = torch.softmax(out["stage_logits"], -1)
    np_ = torch.softmax(out["next_stage_logits"], -1)
    return {
        "stage_logits": out["stage_logits"].detach().cpu(),
        "stage_probabilities": sp.detach().cpu(),
        "predicted_stage": sp.argmax(-1).detach().cpu(),
        "next_stage_logits": out["next_stage_logits"].detach().cpu(),
        "next_stage_probabilities": np_.detach().cpu(),
        "sojourn_distribution": out.get("sojourn_params"),
        "estimated_sojourn_remaining": F.softplus(out["sojourn_params"][:, 0]).detach().cpu(),
        "confidence": sp.max(-1).values.detach().cpu(),
        "latent_embedding": out["latent_embedding"].detach().cpu(),
        "attribution_payload": out.get("attribution_payload", {}),
        **({"event_pmf": out["event_pmf"].detach().cpu()} if "event_pmf" in out else {}),
    }
