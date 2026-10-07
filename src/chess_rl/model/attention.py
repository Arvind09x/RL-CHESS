"""M02 GAB self-attention, adapted from CSSLab/maia3 MHA.

Source: maia3/models.py:47–165 at
1e13597c42d4858b7cfd7cfdae01e297263364b2. Daniel Monroe, George Eilender,
Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, Chessformer,
ICLR 2026. AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local changes: fixed 256/8 dimensions and reuse the isolated M01 CompactGAB.
No static or Q/K/V relative terms are implemented at this milestone.
"""
import torch
from torch import nn

from .gab import CompactGAB, SharedTemplateBank


class GABSelfAttention(nn.Module):
    """Packed Q/K/V attention with GAB added after content-logit scaling."""

    def __init__(self, bank: SharedTemplateBank, dropout: float = 0.1):
        super().__init__()
        factory = {"device": bank.weight.device, "dtype": bank.weight.dtype}
        # Preserve MultiheadAttention's packed projection and constructor reset.
        self.mha = nn.MultiheadAttention(
            256, 8, dropout=dropout, batch_first=True, bias=True, **factory
        )
        self.gab = CompactGAB(bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.gab(x).reshape(x.shape[0] * 8, 64, 64)
        output, _ = self.mha(x, x, x, attn_mask=bias, need_weights=False)
        return output
