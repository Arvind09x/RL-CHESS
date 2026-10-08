"""M06 combined self-attention, adapted from CSSLab/maia3 MHA.

Source: maia3/models.py:47–165 at
1e13597c42d4858b7cfd7cfdae01e297263364b2. Daniel Monroe, George Eilender,
Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, Chessformer,
ICLR 2026. AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local changes: fixed 256/8 dimensions and reuse the isolated M01 CompactGAB.
Relative operation attribution is retained in relative.py.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .gab import CompactGAB, SharedTemplateBank
from .relative import RelativeQKV


class GABSelfAttention(nn.Module):
    """Packed Q/K/V attention with GAB added after content-logit scaling."""

    def __init__(self, bank: SharedTemplateBank, dropout: float = 0.1, *, relative_enabled: bool = False):
        super().__init__()
        factory = {"device": bank.weight.device, "dtype": bank.weight.dtype}
        # Preserve MultiheadAttention's packed projection and constructor reset.
        self.mha = nn.MultiheadAttention(
            256, 8, dropout=dropout, batch_first=True, bias=True, **factory
        )
        self.gab = CompactGAB(bank)
        self.relative = RelativeQKV(**factory) if relative_enabled else None

    def forward(self, x: torch.Tensor, *, relative_enabled: bool = True) -> torch.Tensor:
        bias = self.gab(x)
        if self.relative is None or not relative_enabled:
            output, _ = self.mha(x, x, x, attn_mask=bias.reshape(x.shape[0] * 8, 64, 64),
                                 need_weights=False)
            return output
        # Retain the existing packed tensors and their constructor initialization.
        projected = F.linear(x, self.mha.in_proj_weight, self.mha.in_proj_bias)
        q, k, v = [part.reshape(x.shape[0], 64, 8, 32).transpose(1, 2)
                   for part in projected.chunk(3, dim=-1)]
        logits = (q @ k.transpose(-1, -2) + self.relative.logits(q, k)) / 32 ** 0.5
        probability = F.dropout(torch.softmax(logits + bias, dim=-1),
                                p=self.mha.dropout, training=self.training)
        heads = probability @ v + self.relative.values(probability)
        output = heads.transpose(1, 2).reshape(x.shape[0], 64, 256)
        return F.linear(output, self.mha.out_proj.weight, self.mha.out_proj.bias)
