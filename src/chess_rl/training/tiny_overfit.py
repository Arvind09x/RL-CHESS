"""T03 fixed-batch fit diagnostics; shared T01 losses remain authoritative."""
from dataclasses import dataclass, asdict
import math

import torch

from chess_rl.tensor_ops import legal_policy_log_probs
from .losses import policy_class_cross_entropy, soft_wdl_cross_entropy


@dataclass(frozen=True)
class FitGate:
    step_cap: int = 1000
    accuracy_min: float = 1.0
    mean_kl_max: float = 0.01
    max_kl_max: float = 0.03
    entropy_identity_atol: float = 2e-6

    def __post_init__(self):
        if type(self.step_cap) is not int or self.step_cap < 1:
            raise ValueError("step_cap must be a positive integer")
        for name in ('accuracy_min', 'mean_kl_max', 'max_kl_max', 'entropy_identity_atol'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("gate thresholds must be finite nonnegative numbers")
        if not 0 < self.accuracy_min <= 1 or self.entropy_identity_atol == 0:
            raise ValueError("invalid accuracy/numerical identity threshold")

    def passes(self, metrics):
        return (metrics['legal_best_move_accuracy'] >= self.accuracy_min
                and metrics['teacher_wdl_kl_mean'] <= self.mean_kl_max
                and metrics['teacher_wdl_kl_max'] <= self.max_kl_max
                and metrics['teacher_wdl_kl_min'] >= -self.entropy_identity_atol
                and metrics['entropy_identity_error'] <= self.entropy_identity_atol)


def fit_metrics(policy_logits, wdl_logits, tensors):
    """KL(teacher || model), in nats, with exact zero teacher terms excluded.

    Float64 diagnostics retain D07's float32 targets without renormalizing them.
    T01 CE is independently compared to entropy + KL; no KL clamping is used.
    Argmax is over legal support, with the smallest vocabulary ID breaking ties.
    """
    policy_ce = policy_class_cross_entropy(policy_logits, tensors['legal_mask'], tensors['policy_classes'])
    wdl_ce = soft_wdl_cross_entropy(wdl_logits, tensors['wdl'])
    legal = legal_policy_log_probs(policy_logits, tensors['legal_mask'])
    predicted = legal.argmax(dim=1)
    q = tensors['wdl'].double()
    log_q = torch.where(q > 0, q, torch.ones_like(q)).log()
    log_p = wdl_logits.double().log_softmax(dim=1)
    entropy = -(q * log_q).sum(dim=1)
    kl = (q * (log_q - log_p)).sum(dim=1)
    result = dict(legal_best_move_accuracy=float((predicted == tensors['policy_classes']).double().mean()),
        correct_moves=int((predicted == tensors['policy_classes']).sum()),
        samples=len(q), policy_ce=float(policy_ce), wdl_cross_entropy=float(wdl_ce),
        teacher_entropy_floor=float(entropy.mean()),
        wdl_ce_minus_entropy=float(wdl_ce) - float(entropy.mean()),
        teacher_wdl_kl_mean=float(kl.mean()), teacher_wdl_kl_max=float(kl.max()),
        teacher_wdl_kl_min=float(kl.min()),
        entropy_identity_error=abs(float(wdl_ce) - float(entropy.mean() + kl.mean())),
        per_sample_kl=kl.tolist(), predicted_legal_move_ids=predicted.tolist())
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError("nonfinite fit metrics")
    return result


def measure(model, batch):
    prior_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            tensors = batch.to_torch(device='cpu')
            policy, wdl = model(tensors['features'])
            return fit_metrics(policy, wdl, tensors)
    finally:
        model.train(prior_mode)


def bounded_fit(model, optimizer, batch, *, weights, gate, record):
    """Log baseline and every completed update; stop at the first passing gate.

    A cap failure is an ordinary retained result. Exceptions propagate, leaving
    already flushed records intact. record receives JSON-compatible dictionaries.
    """
    from .trainer import supervised_step
    metrics = measure(model, batch)
    record(dict(step=0, metrics=metrics, update_losses=None))
    # Require a real update even if a random initialization meets the gate.
    for step in range(1, gate.step_cap + 1):
        update = supervised_step(model, optimizer, batch, weights=weights)
        metrics = measure(model, batch)
        record(dict(step=step, metrics=metrics, update_losses=asdict(update)))
        if gate.passes(metrics):
            return dict(status='passed', step=step, cap_reached=step == gate.step_cap, metrics=metrics)
    return dict(status='failed_step_cap', step=gate.step_cap, cap_reached=True, metrics=metrics)
