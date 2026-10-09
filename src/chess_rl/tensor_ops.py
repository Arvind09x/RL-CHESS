"""Shared Torch normalization for training and future inference consumers.

Callers adjudicate terminal states before supplying logits and legal support.
This module owns numerical semantics; the NumPy chess adapter stays Torch-free.
"""
import torch


def finite_matrix(value: torch.Tensor, name: str, width: int | None = None) -> None:
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0
            or (width is not None and value.shape[1] != width)):
        raise ValueError(f"{name} requires a nonempty floating [B,{width or 'N'}] tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def validate_legal_support(legal_mask: torch.Tensor) -> None:
    if (not isinstance(legal_mask, torch.Tensor) or legal_mask.dtype != torch.bool
            or legal_mask.ndim != 2 or 0 in legal_mask.shape):
        raise ValueError("legal_mask must be a nonempty boolean [B,N] tensor")
    if not legal_mask.any(dim=1).all():
        raise ValueError("policy requires nonempty legal support in every row")


def legal_policy_log_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """One stable normalization over legal entries; illegal log-probs are -inf.

    Reject nonfinite raw logits even at illegal entries. Masking before the
    normalization cuts every illegal logit's loss gradient exactly to zero.
    Do not multiply dense targets by this result: select supported entries.
    """
    finite_matrix(logits, "policy logits")
    validate_legal_support(legal_mask)
    if legal_mask.shape != logits.shape or legal_mask.device != logits.device:
        raise ValueError("legal_mask must be boolean with the logits' shape and device")
    result = torch.log_softmax(logits.masked_fill(~legal_mask, -torch.inf), dim=1)
    if not torch.isfinite(result[legal_mask]).all():
        raise ValueError("legal log probabilities are nonfinite (dtype range exceeded)")
    return result


def legal_policy_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Legal priors with exact zero at illegal entries, using the same owner."""
    return legal_policy_log_probs(logits, legal_mask).exp()


def validate_soft_wdl(targets: torch.Tensor, logits: torch.Tensor) -> None:
    """Validate, never harden or renormalize root-side [win,draw,loss] targets."""
    finite_matrix(targets, "WDL targets", 3)
    if targets.shape != logits.shape or targets.device != logits.device or targets.requires_grad:
        raise ValueError("WDL targets must match logits' shape/device and require no gradient")
    if (targets < 0).any() or (targets > 1).any():
        raise ValueError("WDL targets must be probabilities in [0,1]")
    # D07 float32 teacher counts/1000 may have ordinary rounding error.
    if not torch.allclose(targets.sum(dim=1), torch.ones_like(targets[:, 0]), atol=1e-6, rtol=0):
        raise ValueError("WDL targets must sum to one")
