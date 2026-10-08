"""Q/K/V relative operations ported from the pinned Leela training fork.

Source: daniel-monroe/lczero-training, tf/tfprocess.py:83–91,285–341,
1960–2007 at 7b3ee3fcdb49d8abeb034718935b099134c11882.
Copyright (C) 2017–2018 Gian-Carlo Pascutto; Daniel Monroe and Leela contributors.
GPL-3.0-or-later: docs/notices/lczero-tfprocess-header.txt.
Local changes: PyTorch, logical [head, displacement, depth] tables and indexed
selection instead of the dense one-hot factorizer. No position-only cross term.
"""
import torch
from torch import nn


class RelativeQKV(nn.Module):
    """Three independent zero tables; R[h,r,d] = upstream[d*8+h,r]."""

    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.q = nn.Parameter(torch.zeros(8, 225, 32, **factory))
        self.k = nn.Parameter(torch.zeros(8, 225, 32, **factory))
        self.v = nn.Parameter(torch.zeros(8, 225, 32, **factory))
        squares = torch.arange(64, device=device)
        rank, file = squares // 8, squares % 8
        # Query minus key for BOTH content terms, including the key operand.
        index = 15 * (rank[:, None] - rank[None, :] + 7)
        index = index + file[:, None] - file[None, :] + 7
        self.register_buffer("index", index, persistent=False)

    def logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Unscaled q_i·U_delta + k_j·T_delta, [B,8,64,64]."""
        return (torch.einsum("bhid,hijd->bhij", q, self.q[:, self.index])
                + torch.einsum("bhjd,hijd->bhij", k, self.k[:, self.index]))

    def values(self, probability: torch.Tensor) -> torch.Tensor:
        """sum_j alpha[i,j] V_delta, using the same dropped probabilities."""
        return torch.einsum("bhij,hijd->bhid", probability, self.v[:, self.index])
