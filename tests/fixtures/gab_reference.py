"""Verbatim MHA._sq_bias from CSSLab/maia3/maia3/models.py, lines 95–118.
Pin: 1e13597c42d4858b7cfd7cfdae01e297263364b2.
Full-file SHA256: 9172bfd41e770baef734f472c6506e7615aa3f4b3f4227f41e9cf35e6ef0be0c.
Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson
/ UofT CSSLab. AGPL-3.0, docs/notices/maia3-AGPL-3.0.txt.
Local wrapper only: no full Maia import or attention constructor.
"""
import torch


class PinnedOperation:
    def _sq_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 64, d_model) – board‑square tokens only (ignore CLS, elo, …)
        returns (B, H, 64, 64) – square‑pair bias matrix
        """
        squares = x                         # (B, 64, d_model)
        B = squares.size(0)

        if self.sm1 is not None:
            y = self.sm1(squares)                                  # (B, 64, p)
            y = y.reshape(B, -1)                                   # (B, 64*p)
            y = self.sm_act(self.sm2(y))                           # (B, hdim)
        else:
            y = torch.mean(squares, dim=1)                           # (B, d_model)
            y = self.sm_act(self.sm2(y))                           # (B, hdim)


        y = self.ln1(y)
        y = self.sm_act(self.sm3(y))                           # (B, H*gen)
        y = self.ln2(y).view(B, self.num_heads, self.gen_size) # (B, H, gen)

        #  einsum → (B, H, 64*64)
        b = torch.einsum("bhi,oi->bho", y, self.gab_weight)
        return b.view(B, self.num_heads, 64, 64)
