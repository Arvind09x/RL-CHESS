"""Shared AdamW construction, gradient clipping and warmup/cosine schedule.

Settings follow the 12 October 2026 audit: AdamW as in Chessformer/Maia-3
(arXiv 2605.19091, App. A.1), global-norm clipping as in Leela-CF (2409.12272)
and linear warmup with cosine decay. Weight decay 0.01 on matrix weights is an
experimental small-data choice, not a cited value. P and every R share this
builder so optimizer differences never enter the P-versus-R comparison.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from chess_rl.model import ChessNetwork

GROUPING_VERSION = "adamw-grouped-v1"
SCHEDULE_VERSION = "warmup-cosine-v1"


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_gradient_norm: float | None = 10.0
    grouping: str = GROUPING_VERSION

    def __post_init__(self):
        for name in ("learning_rate", "weight_decay", "beta1", "beta2", "eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"optimizer {name} must be a finite nonnegative number")
        if self.learning_rate <= 0 or self.eps <= 0 or not (self.beta1 < 1 and self.beta2 < 1):
            raise ValueError("learning rate and eps must be positive; betas below one")
        if self.max_gradient_norm is not None and (type(self.max_gradient_norm) not in (int, float)
                or not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0):
            raise ValueError("max_gradient_norm must be None or finite positive")
        if self.grouping != GROUPING_VERSION:
            raise ValueError("unsupported parameter grouping version")


@dataclass(frozen=True)
class ScheduleSettings:
    """Learning-rate multiplier as a pure function of completed updates."""
    warmup_steps: int = 500
    total_steps: int = 1
    final_fraction: float = 0.1
    version: str = SCHEDULE_VERSION

    def __post_init__(self):
        if type(self.warmup_steps) is not int or type(self.total_steps) is not int or self.warmup_steps < 0 or self.total_steps < 1:
            raise ValueError("warmup_steps must be nonnegative and total_steps positive integers")
        if type(self.final_fraction) not in (int, float) or not (0 <= self.final_fraction <= 1):
            raise ValueError("final_fraction must lie in [0, 1]")
        if self.version != SCHEDULE_VERSION:
            raise ValueError("unsupported schedule version")

    def multiplier(self, completed_steps: int) -> float:
        """Multiplier for the update that follows `completed_steps` updates."""
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer")
        if completed_steps < self.warmup_steps:
            return (completed_steps + 1) / self.warmup_steps
        span = self.total_steps - self.warmup_steps
        if span <= 0:
            return 1.0
        position = min(completed_steps - self.warmup_steps, span) / span
        cosine = 0.5 * (1 + math.cos(math.pi * position))
        return self.final_fraction + (1 - self.final_fraction) * cosine


def parameter_groups(model: ChessNetwork):
    """Deterministic (decay, no-decay) parameter name lists.

    Decay: every two-dimensional weight (linear layers, attention projections,
    the GAB template bank). No decay: biases, LayerNorm affine parameters, the
    absolute square embeddings and the relative q/k/v position tables.
    Shared parameters appear once, under their first registered name.
    """
    if not isinstance(model, ChessNetwork):
        raise ValueError("parameter grouping requires ChessNetwork")
    norm_names = {name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)}
    decay, no_decay, seen = [], [], set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        if parameter.ndim < 2 or owner in norm_names or name.endswith("square_embeddings") \
                or name.endswith((".relative.q", ".relative.k", ".relative.v")):
            no_decay.append(name)
        else:
            decay.append(name)
    return decay, no_decay


def build_optimizer(model: ChessNetwork, settings: OptimizerSettings) -> torch.optim.AdamW:
    """AdamW with two named groups; foreach/fused disabled for the determinism contract."""
    if type(settings) is not OptimizerSettings:
        raise ValueError("build_optimizer requires OptimizerSettings")
    named = dict(model.named_parameters())
    decay, no_decay = parameter_groups(model)
    groups = [dict(params=[named[n] for n in decay], weight_decay=settings.weight_decay),
              dict(params=[named[n] for n in no_decay], weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=settings.learning_rate, betas=(settings.beta1, settings.beta2),
                             eps=settings.eps, weight_decay=settings.weight_decay, foreach=False, fused=False)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float):
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite positive")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def gradient_norm(parameters) -> float:
    """Global L2 norm in float64 over the supplied unique parameters' gradients."""
    gradients = [p.grad for p in parameters]
    if not gradients or any(g is None for g in gradients):
        raise ValueError("gradient norm requires every model gradient")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for g in gradients]))
    if not torch.isfinite(norm):
        raise ValueError("gradient norm must be finite")
    return float(norm)


def clip_gradients(parameters, max_norm: float | None):
    """Scale gradients in place to a global norm of at most max_norm.

    Returns (unclipped_norm, clipped). With max_norm None nothing is modified.
    """
    parameters = list(parameters)
    norm = gradient_norm(parameters)
    if max_norm is None or norm <= max_norm:
        return norm, False
    scale = max_norm / (norm + 1e-6)
    with torch.no_grad():
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return norm, True


def describe(settings: OptimizerSettings, schedule: ScheduleSettings | None = None):
    """JSON-ready provenance record."""
    record = dict(kind="AdamW", **asdict(settings))
    if schedule is not None:
        record["schedule"] = asdict(schedule)
    return record
