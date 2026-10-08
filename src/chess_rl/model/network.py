"""M06 fixed GAB + Q/K/V relative network with the accepted M03/M04 heads.

Component attribution and pinned source notices remain in backbone.py,
policy.py, wdl.py and relative.py. See docs/m06-relative.md.
"""
import torch
from torch import nn

from chess_rl.contracts import WDL_ORDER
from .backbone import EncoderBackbone
from .policy import PairPolicyHead
from .wdl import WDLHead


class ChessNetwork(nn.Module):
    """Features [B,64,128] to (policy [B,4352], WDL [B,3]) raw logits.

    Policy uses pair4352-v1; WDL rows use shared WDL_ORDER from the input
    position's side-to-move perspective. Both heads consume the same
    encoder-final tokens. No mask, softmax, value conversion or adjudication.
    Defaults retain training dropout .1 and independent head widths 128.
    B must be positive; the accepted backbone does not support empty batches.
    Construct optimizers from parameters() to deduplicate the shared GAB bank;
    move the whole network together to preserve coherent device/dtype settings.
    """

    wdl_order = WDL_ORDER

    def __init__(self, *, d_policy: int = 128, d_value: int = 128,
                 dropout: float = 0.1, relative_enabled: bool = True,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.backbone = EncoderBackbone(dropout=dropout, relative_enabled=relative_enabled, **factory)
        self.policy_head = PairPolicyHead(d_policy=d_policy, **factory)
        self.wdl_head = WDLHead(d_value=d_value, **factory)

    def forward(self, features: torch.Tensor, *, relative_enabled: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits; relative_enabled=False diagnoses M05 common weights.

        Constructor disabling omits tables for historical M05 acceptance. Forward
        disabling retains tables but bypasses them without mutating parameters.
        """
        if features.ndim == 3 and features.shape == (0, 64, 128):
            raise ValueError("ChessNetwork requires a nonempty batch (B >= 1)")
        tokens = self.backbone(features, relative_enabled=relative_enabled)
        return self.policy_head(tokens), self.wdl_head(tokens)
