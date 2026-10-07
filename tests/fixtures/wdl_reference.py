"""Pinned Maia-3 value-only forward excerpt; no full upstream model import.

CSSLab/maia3 commit 1e13597c42d4858b7cfd7cfdae01e297263364b2,
maia3/models.py:396–397. Daniel Monroe, George Eilender, Philip Chalmers,
Zhenwei Tang, Ashton Anderson / UofT CSSLab, Chessformer, ICLR 2026.
AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local change: wrapper and return, omit ponder/policy; output row interpretation
is our contract, whereas pinned uci.py:196–200 interprets loss/draw/win.
"""
from torch.nn import functional as F


def pinned_wdl(self, x):
        x = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(x)))          # (B, 3)
        return logits_value
