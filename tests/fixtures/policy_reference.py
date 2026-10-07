"""Frozen M03 Maia-3 policy body, retained verbatim with attribution.

CSSLab/maia3 maia3/models.py:373–394 at
1e13597c42d4858b7cfd7cfdae01e297263364b2; Daniel Monroe, George Eilender,
Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, ICLR 2026.
AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt. Local wrapper only; no upstream
model import. Independent scalar/component references live in policy_checks.
"""
import math
from types import SimpleNamespace

import chess
import torch


def pinned_policy(head, x):
    self = SimpleNamespace(proj_sq_from=head.proj_sq_from,
                           proj_sq_to=head.proj_sq_to,
                           promo_bias_proj=head.promo_bias_proj,
                           cfg=SimpleNamespace(head_hid_dim=head.d_policy))
    if True:
        sq_from = self.proj_sq_from(x[:, :64, :])
        sq_to = self.proj_sq_to(x[:, :64, :])
        scores_base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.cfg.head_hid_dim)
        scores_flat = scores_base.reshape(x.size(0), 64 * 64)  # (B, 4096)

        rank7_indices = [chess.square(file, 6) for file in range(8)]  # squares 48-55
        rank8_indices = [chess.square(file, 7) for file in range(8)]  # squares 56-63

        rank8_features = sq_to[:, rank8_indices, :]  # (B, 8, head_hid_dim)
        promo_biases = self.promo_bias_proj(rank8_features) * math.sqrt(self.cfg.head_hid_dim)  # (B, 8, 4) for q,r,b,n

        promotion_logits = []
        for from_file in range(8):  # source file (a-h)
            from_sq = rank7_indices[from_file]
            for to_file in range(8):  # target file (a-h)
                to_sq = rank8_indices[to_file]
                base_score = scores_base[:, from_sq, to_sq]  # (B,)
                for piece_idx in range(4):  # q=0, r=1, b=2, n=3
                    bias = promo_biases[:, to_file, piece_idx]  # (B,)
                    promotion_logits.append((base_score + bias).unsqueeze(1))
        promotion_logits = torch.cat(promotion_logits, dim=1)  # (B, 256)
        logits_move = torch.cat([scores_flat, promotion_logits], dim=1)  # (B, 4352)
    return logits_move
