"""Compact generated attention bias (M01 only).

Adapted from CSSLab/maia3, maia3/models.py, commit
1e13597c42d4858b7cfd7cfdae01e297263364b2: MHA.__init__, MHA._sq_bias and
MAIA3Model's shared-bank initialization. Authors: Daniel Monroe, George
Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab.
AGPL-3.0; retained text: docs/notices/maia3-AGPL-3.0.txt.
Local changes: isolate the compact 8/32/32 generator from attention and require
an explicitly shared bank. See docs/upstream-sources.md for attribution.
"""

import torch
from torch import nn


class SharedTemplateBank(nn.Module):
    """Create once, then pass this same module to every layer's generator.

    Build an optimizer from the enclosing module's parameters() so PyTorch
    deduplicates shared parameters. Do not concatenate each generator's list.
    """

    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4096, 32, device=device, dtype=dtype))
        nn.init.xavier_normal_(self.weight)


class CompactGAB(nn.Module):
    """Generate [B,8,64,64] biases from [B,64,256] square embeddings.

    Each instance owns separate coefficient projections and norms; its bank
    is retained by identity. LayerNorm(256) acts across all heads jointly.
    """

    def __init__(self, bank: SharedTemplateBank):
        super().__init__()
        if not isinstance(bank, SharedTemplateBank):
            raise TypeError("bank must be a SharedTemplateBank instance")
        self.bank = bank
        factory = {"device": bank.weight.device, "dtype": bank.weight.dtype}
        self.sm1 = nn.Linear(256, 8, **factory)
        self.sm2 = nn.Linear(512, 32, **factory)
        self.ln1 = nn.LayerNorm(32, **factory)
        self.sm3 = nn.Linear(32, 256, **factory)
        self.ln2 = nn.LayerNorm(256, **factory)
        self.sm_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (64, 256):
            raise ValueError("CompactGAB expects [B,64,256] inputs")
        batch = x.shape[0]
        y = self.sm1(x).reshape(batch, 512)
        y = self.ln1(self.sm_act(self.sm2(y)))
        y = self.ln2(self.sm_act(self.sm3(y))).reshape(batch, 8, 32)
        bias = torch.einsum("bhi,oi->bho", y, self.bank.weight)
        return bias.reshape(batch, 8, 64, 64)
