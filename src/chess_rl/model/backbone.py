"""M02 encoder backbone adapted from the pinned Maia-3 source.

Source: CSSLab/maia3, maia3/models.py:36–42, 170–283, 295–322, 351–369,
commit 1e13597c42d4858b7cfd7cfdae01e297263364b2. Daniel Monroe, George
Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab,
Chessformer, ICLR 2026. AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local changes: fixed contract dimensions, full 128-channel adapter inputs,
no Elo/time conditioning, explicit M01 bank module. See docs/m02-backbone.md.
"""
import torch
from torch import nn

from .attention import GABSelfAttention
from .gab import SharedTemplateBank


class EncoderBlock(nn.Module):
    """Independent attention/FFN parameters with Post-LayerNorm residuals."""

    def __init__(self, bank: SharedTemplateBank, dropout: float = 0.1, *, relative_enabled: bool = False):
        super().__init__()
        factory = {"device": bank.weight.device, "dtype": bank.weight.dtype}
        self.self_attn = GABSelfAttention(bank, dropout, relative_enabled=relative_enabled)
        self.linear1 = nn.Linear(256, 256, **factory)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 256, **factory)
        self.norm1 = nn.LayerNorm(256, **factory)
        self.norm2 = nn.LayerNorm(256, **factory)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, *, relative_enabled: bool = True) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, relative_enabled=relative_enabled)))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout2(ff))


class EncoderBackbone(nn.Module):
    """Map [B,64,128] to [B,64,256]; M02 GAB diagnostic by default; network enables M06 terms.

    Training dropout defaults to 0.1. Zero is supported for numerical checks.
    Build optimizers from this module's parameters() to deduplicate the bank.
    Move the entire backbone together to preserve shared parameter identity.
    """

    def __init__(self, *, dropout: float = 0.1, relative_enabled: bool = False,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        factory = {"device": device, "dtype": dtype}
        self.input_projection = nn.Linear(128, 256, **factory)
        self.bank = SharedTemplateBank(**factory)
        self.square_embeddings = nn.Parameter(torch.zeros(64, 256, **factory))
        # Construct each block separately; do not clone identical initial weights.
        self.layers = nn.ModuleList([EncoderBlock(self.bank, dropout, relative_enabled=relative_enabled) for _ in range(8)])
        self.norm = nn.LayerNorm(256, **factory)

    def forward(self, x: torch.Tensor, *, relative_enabled: bool = True) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (64, 128):
            raise ValueError("EncoderBackbone expects [B,64,128] inputs")
        x = self.input_projection(x) + self.square_embeddings
        for block in self.layers:
            x = block(x, relative_enabled=relative_enabled)
        return self.norm(x)
