"""R03 one mixed update on two audited, frozen training sources."""
from dataclasses import dataclass

import numpy as np
import torch

from chess_rl.contracts import FEATURE_SHAPE, MOVE_COUNT
from chess_rl.model import ChessNetwork
from chess_rl.tensor_ops import validate_soft_wdl
from .losses import (policy_class_cross_entropy, policy_distribution_cross_entropy,
                     soft_wdl_cross_entropy, validate_policy_classes, validate_policy_distribution)
from .mixed_state import MixedState, MixedSettings, require
from .trainer import StepMetrics, finite_tensors


def validate_optimizer_settings(payload, settings):
    """R03 constant standard Adam choices; common persistence owns full schema."""
    require(payload["kind"] == "Adam", "mixed updates require Adam")
    expected = dict(lr=settings.learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False, maximize=False, foreach=False,
        capturable=False, differentiable=False, fused=False, decoupled_weight_decay=False)
    for options in [payload["defaults"], *[g["options"] for g in payload["groups"]]]:
        for key, value in expected.items():
            actual = tuple(options[key]) if key == "betas" else options[key]
            require(type(actual) is type(value) and actual == value,
                    f"mixed optimizer settings mismatch: {key}")


@dataclass(frozen=True)
class MixedMetrics:
    supervised: StepMetrics
    selfplay: StepMetrics
    supervised_contribution: float
    selfplay_contribution: float
    total_loss: float


def mixed_losses(policy, wdl, *, supervised_mask, classes, teacher_wdl,
                 selfplay_mask, visits, final_wdl, settings):
    """Two target contracts, weighted by the frozen exact batch fraction."""
    require(type(settings) is MixedSettings, "MixedSettings required")
    ns, nr = settings.supervised_batch_size, settings.selfplay_batch_size
    require(policy.shape == (ns + nr, MOVE_COUNT) and wdl.shape == (ns + nr, 3),
            "logit shape must match frozen mixed composition")
    require(supervised_mask.shape == (ns, MOVE_COUNT) and selfplay_mask.shape == (nr, MOVE_COUNT)
            and teacher_wdl.shape == (ns, 3) and final_wdl.shape == (nr, 3),
            "target shape must match frozen source counts")
    validate_soft_wdl(final_wdl, wdl[ns:])
    require(bool(((final_wdl == 0) | (final_wdl == 1)).all()),
            "self-play WDL must be an exact final one-hot outcome")
    sp = policy_class_cross_entropy(policy[:ns], supervised_mask, classes)
    sw = soft_wdl_cross_entropy(wdl[:ns], teacher_wdl)
    rp = policy_distribution_cross_entropy(policy[ns:], selfplay_mask, visits)
    rw = soft_wdl_cross_entropy(wdl[ns:], final_wdl)
    supervised = settings.policy_weight * sp + settings.wdl_weight * sw
    selfplay = settings.policy_weight * rp + settings.wdl_weight * rw
    s = settings.supervised_fraction * supervised
    r = (1 - settings.supervised_fraction) * selfplay
    total = s + r
    require(bool(torch.isfinite(total)), "mixed loss must be finite")
    metrics = MixedMetrics(StepMetrics(ns, float(sp.detach()), float(sw.detach()), float(supervised.detach())),
        StepMetrics(nr, float(rp.detach()), float(rw.detach()), float(selfplay.detach())),
        float(s.detach()), float(r.detach()), float(total.detach()))
    return total, metrics


def mixed_step(model, optimizer, sources, state):
    """Return metrics and candidate committed state after one optimizer step.

    Input/loss/gradient failures prevent optimizer execution and do not advance
    sampling. Arbitrary optimizer failure is not transactional: restore the last
    immutable checkpoint after such a failure. Sources never mutate replay.
    """
    from .checkpoints import _optimizer_payload
    require(type(state) is MixedState, "MixedState required; a D07 cursor is insufficient")
    state.validate_sources(sources)
    require(type(model) is ChessNetwork and all(layer.self_attn.relative is not None
            for layer in model.backbone.layers), "mixed training requires combined M06")
    parameters = list(model.parameters())
    device = parameters[0].device
    require(device.type in ("cpu", "cuda") and all(p.device == device and p.dtype == parameters[0].dtype
                for p in parameters) and finite_tensors(parameters), "finite uniform supported parameters required")
    payload = _optimizer_payload(model, optimizer)
    validate_optimizer_settings(payload, state.settings)
    require(len(optimizer.state) == (len(parameters) if state.updates else 0)
            and all(optimizer.state[p]["step"].item() == state.updates for p in optimizer.state),
            "optimizer/mixed update progress mismatch")
    si, ri, next_state = state.next_selection()
    supervised, replay = sources.batches(si, ri)
    ns, nr = len(supervised.references), len(replay.sample_ids)
    for batch, size, target in ((supervised, ns, "policy_classes"), (replay, nr, "policy_distributions")):
        for name, shape, dtype in (("features", (size, *FEATURE_SHAPE), np.float32),
                ("legal_mask", (size, MOVE_COUNT), np.bool_), ("wdl", (size, 3), np.float32),
                (target, (size,) if target == "policy_classes" else (size, MOVE_COUNT),
                 np.int64 if target == "policy_classes" else np.float32)):
            value = getattr(batch, name)
            require(type(value) is np.ndarray and value.shape == shape and value.dtype == dtype,
                    f"invalid mixed {name} shape/dtype")
        require(bool(np.isfinite(batch.features).all()), "mixed features must be finite")
    tensors = supervised.to_torch(device=device)
    r = {name: torch.tensor(getattr(replay, name), device=device) for name in ("legal_mask", "policy_distributions", "wdl")}
    validate_policy_classes(tensors["legal_mask"], tensors["policy_classes"])
    validate_policy_distribution(r["legal_mask"], r["policy_distributions"])
    dtype = parameters[0].dtype
    validate_soft_wdl(tensors["wdl"], torch.zeros((ns, 3), dtype=dtype, device=device))
    validate_soft_wdl(r["wdl"], torch.zeros((nr, 3), dtype=dtype, device=device))
    require(bool(((r["wdl"] == 0) | (r["wdl"] == 1)).all()), "exact final WDL required")
    features = torch.tensor(np.concatenate((supervised.features, replay.features)), dtype=dtype, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        policy, wdl = model(features)
        loss, metrics = mixed_losses(policy, wdl, supervised_mask=tensors["legal_mask"],
            classes=tensors["policy_classes"], teacher_wdl=tensors["wdl"], selfplay_mask=r["legal_mask"],
            visits=r["policy_distributions"], final_wdl=r["wdl"], settings=state.settings)
        loss.backward()
        require(all(p.grad is not None for p in parameters) and finite_tensors(p.grad for p in parameters),
                "every combined-model parameter requires a finite mixed gradient")
        optimizer.step()
        require(finite_tensors(parameters), "nonfinite mixed optimizer result")
        return metrics, next_state
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
