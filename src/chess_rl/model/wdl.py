"""M04 WDL computation adapted from CSSLab/maia3 models.py:323–326,396–397.

Pin: 1e13597c42d4858b7cfd7cfdae01e297263364b2. Daniel Monroe, George
Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab,
Chessformer, ICLR 2026. AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local changes: standalone head, fixed input width, independent d_value,
contract output row semantics. See docs/m04-wdl.md; no checkpoint equivalence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from chess_rl.contracts import WDL_ORDER


class WDLHead(nn.Module):
    """Encoder-final [B,64,256] tokens to raw [B,3] side-to-move logits.

    Rows use shared WDL_ORDER for newly trained weights and targets. Consumers
    own softmax, value conversion, terminal handling and outcome labels.
    d_value is independent of the policy width. Constructor resets are retained.
    """

    wdl_order = WDL_ORDER

    def __init__(self, *, d_value: int = 128, device="cpu", dtype=torch.float32):
        super().__init__()
        if isinstance(d_value, bool) or not isinstance(d_value, int) or d_value <= 0:
            raise ValueError("d_value must be a positive integer")
        self.d_value = d_value
        factory = {"device": device, "dtype": dtype}
        self.last_ln = nn.LayerNorm(256, eps=1e-5, elementwise_affine=True, bias=True, **factory)
        self.fc_value_hid = nn.Linear(256, d_value, bias=True, **factory)
        self.fc_value = nn.Linear(d_value, len(WDL_ORDER), bias=True, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (64, 256):
            raise ValueError("WDLHead expects [B,64,256] inputs")
        pooled = self.last_ln(x.mean(dim=1))
        return self.fc_value(F.relu(self.fc_value_hid(pooled)))
