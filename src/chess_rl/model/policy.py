"""M03 pair policy, adapted from the pinned Maia-3 operations.

Source: CSSLab/maia3, maia3/models.py:331–334, 373–394, commit
1e13597c42d4858b7cfd7cfdae01e297263364b2. Daniel Monroe, George Eilender,
Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, Chessformer,
ICLR 2026. AGPL-3.0: docs/notices/maia3-AGPL-3.0.txt.
Local changes: standalone head, fixed 256 input width, validated d_policy,
vectorized promotion assembly. See docs/m03-policy.md.
"""
import math

import torch
from torch import nn


class PairPolicyHead(nn.Module):
    """Contextual [B,64,256] tokens to raw pair4352-v1 logits [B,4352].

    d_policy is the policy-head configuration field, initially 128. All slots
    remain present; legality and move conversion belong to the adapter.
    Linear constructor initialization is retained without a model-wide reset.
    """

    def __init__(self, *, d_policy: int = 128, device="cpu", dtype=torch.float32):
        super().__init__()
        if isinstance(d_policy, bool) or not isinstance(d_policy, int) or d_policy <= 0:
            raise ValueError("d_policy must be a positive integer")
        self.d_policy = d_policy
        factory = {"device": device, "dtype": dtype}
        self.proj_sq_from = nn.Linear(256, d_policy, bias=False, **factory)
        self.proj_sq_to = nn.Linear(256, d_policy, bias=False, **factory)
        self.promo_bias_proj = nn.Linear(d_policy, 4, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (64, 256):
            raise ValueError("PairPolicyHead expects [B,64,256] inputs")
        q = self.proj_sq_from(x)
        k = self.proj_sq_to(x)
        scale = math.sqrt(self.d_policy)
        scores = (q @ k.transpose(1, 2)) / scale
        adjustment = self.promo_bias_proj(k[:, 56:64]) * scale
        # [B,from_file,to_file,piece], with q/r/b/n varying fastest.
        promotions = scores[:, 48:56, 56:64, None] + adjustment[:, None, :, :]
        return torch.cat((scores.reshape(x.shape[0], 4096),
                          promotions.reshape(x.shape[0], 256)), dim=1)
