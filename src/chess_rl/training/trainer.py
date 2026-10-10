"""One supervised optimizer step on an accepted D07 training batch.

No epoch loop, checkpoint, shuffling, dataset loading or runtime/RNG mutation.
D07's loader owns replay, provenance and terminal exclusion; use its batches.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from chess_rl.contracts import FEATURE_SHAPE, MOVE_COUNT
from chess_rl.model import ChessNetwork
from chess_rl.supervised import SupervisedBatch
from chess_rl.tensor_ops import validate_soft_wdl
from .losses import policy_class_cross_entropy, soft_wdl_cross_entropy, validate_policy_classes


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    wdl: float = 1.0

    def __post_init__(self):
        for value in (self.policy, self.wdl):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("loss weights must be finite positive numbers")


@dataclass(frozen=True)
class StepMetrics:
    batch_size: int
    policy_loss: float
    wdl_loss: float
    total_loss: float


def finite_tensors(values):
    """One host synchronization for a collection of tensors on one device."""
    values = tuple(values)
    return bool(values) and bool(torch.stack([torch.isfinite(value).all() for value in values]).all())


def supervised_step(model: ChessNetwork, optimizer: torch.optim.Optimizer,
                    batch: SupervisedBatch, *, weights: LossWeights = LossWeights()) -> StepMetrics:
    """Validate the owned batch, train both heads, and update unique parameters.

    Returns pre-update scalar losses. Leaves batch arrays/references untouched.
    On input/loss/gradient failure no optimizer step is attempted. Gradients are
    cleared before backward and on backward failure; no rollback of an arbitrary
    optimizer is promised. Runtime settings and optimizer choices are explicit
    caller responsibilities, recorded by the bounded T01 check.
    """
    if not isinstance(model, ChessNetwork) or any(layer.self_attn.relative is None for layer in model.backbone.layers):
        raise ValueError("supervised training requires the combined GAB + Q/K/V network")
    if not isinstance(weights, LossWeights):
        raise ValueError("weights must be LossWeights")
    parameters = list(model.parameters())
    supplied = [p for group in optimizer.param_groups for p in group['params']]
    if len(supplied) != len(parameters) or {id(p) for p in supplied} != {id(p) for p in parameters}:
        raise ValueError("optimizer must own every unique model parameter exactly once")
    if not isinstance(batch, SupervisedBatch) or batch.split != "train":
        raise ValueError("supervised step requires a D07 training batch")
    size = len(batch.references)
    for name, shape, dtype in (
            ("features", (size, *FEATURE_SHAPE), np.float32),
            ("legal_mask", (size, MOVE_COUNT), np.bool_),
            ("policy_classes", (size,), np.int64), ("wdl", (size, 3), np.float32)):
        value = getattr(batch, name)
        if size == 0 or not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"invalid D07 batch {name} shape/dtype or empty batch")
    if not np.isfinite(batch.features).all():
        raise ValueError("features must be finite")
    if not finite_tensors(parameters):
        raise ValueError("model parameters must be finite")
    device, dtype = parameters[0].device, parameters[0].dtype
    tensors = batch.to_torch(device=device)
    # Validate targets before forward (without introducing a second loss owner).
    validate_policy_classes(tensors['legal_mask'], tensors['policy_classes'])
    validate_soft_wdl(tensors['wdl'], torch.zeros((size, 3), device=device, dtype=dtype))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        policy, wdl = model(tensors['features'].to(dtype=dtype))
        policy_loss = policy_class_cross_entropy(policy, tensors['legal_mask'], tensors['policy_classes'])
        wdl_loss = soft_wdl_cross_entropy(wdl, tensors['wdl'])
        total = weights.policy * policy_loss + weights.wdl * wdl_loss
        if not torch.isfinite(total):
            raise ValueError("combined loss must be finite")
        total.backward()
        if any(p.grad is None for p in parameters) or not finite_tensors(p.grad for p in parameters):
            raise ValueError("every combined-model parameter requires a finite gradient")
        metrics = StepMetrics(size, float(policy_loss.detach()), float(wdl_loss.detach()), float(total.detach()))
        optimizer.step()
        if not finite_tensors(parameters):
            raise ValueError("optimizer produced nonfinite parameters")
        return metrics
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
