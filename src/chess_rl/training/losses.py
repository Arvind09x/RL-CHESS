"""T01 losses. Policy teachers are classes; WDL teachers stay soft."""
import torch

from chess_rl.contracts import MOVE_COUNT, WDL_ORDER
from chess_rl.tensor_ops import (finite_matrix, legal_policy_log_probs,
                                validate_legal_support, validate_soft_wdl)


def validate_policy_classes(legal_mask: torch.Tensor, classes: torch.Tensor) -> None:
    """The class-target boundary, also checked by the step before forward."""
    validate_legal_support(legal_mask)
    if legal_mask.shape[1] != MOVE_COUNT:
        raise ValueError("legal_mask must use the fixed move vocabulary")
    if (not isinstance(classes, torch.Tensor) or classes.dtype != torch.int64
            or classes.shape != (legal_mask.shape[0],) or classes.device != legal_mask.device):
        raise ValueError("policy classes must be int64 [B] on the legal_mask's device")
    if ((classes < 0) | (classes >= MOVE_COUNT)).any():
        raise ValueError("policy class is outside the move vocabulary")
    selected = classes[:, None]
    if not legal_mask.gather(1, selected).all():
        raise ValueError("policy teacher class must be legal")


def policy_class_cross_entropy(logits: torch.Tensor, legal_mask: torch.Tensor,
                               classes: torch.Tensor) -> torch.Tensor:
    """Mean negative legal log probability of each integer teacher class."""
    finite_matrix(logits, "policy logits", MOVE_COUNT)
    validate_policy_classes(legal_mask, classes)
    log_probs = legal_policy_log_probs(logits, legal_mask)
    selected = classes[:, None]
    loss = -log_probs.gather(1, selected).mean()
    if not torch.isfinite(loss):
        raise ValueError("policy loss must be finite")
    return loss


def soft_wdl_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean soft-target CE, retaining WDL_ORDER from the input root's player.

    Zero target entries select zero log-probability *before* multiplication,
    so zero mass never multiplies -inf. Unrepresentable positive-mass losses
    fail explicitly rather than being clamped or silently accepted.
    """
    finite_matrix(logits, "WDL logits", len(WDL_ORDER))
    validate_soft_wdl(targets, logits)
    log_probs = torch.log_softmax(logits, dim=1)
    supported = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
    loss = -(targets * supported).sum(dim=1).mean()
    if not torch.isfinite(loss):
        raise ValueError("WDL loss must be finite")
    return loss
