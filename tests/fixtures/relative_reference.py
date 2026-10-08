"""Independent M06 reference derived from pinned factorizer/layout equations.

Leela tf/tfprocess.py:83–91,285–341,1960–2007, commit
7b3ee3fcdb49d8abeb034718935b099134c11882. Copyright (C) 2017–2018
Gian-Carlo Pascutto; Daniel Monroe / Leela contributors. GPL-3.0-or-later;
docs/notices/lczero-tfprocess-header.txt. No production relative/attention calls.
"""
import numpy as np
import torch


def factorizer():
    f = np.zeros((225, 4096))
    for rank_i in range(8):
        for file_i in range(8):
            for rank_j in range(8):
                for file_j in range(8):
                    f[15*(rank_i-rank_j+7)+file_i-file_j+7,
                      64*(8*rank_i+file_i)+8*rank_j+file_j] = 1
    return f


FACTOR = factorizer()


def upstream_layout(table):
    # Explicit depth-before-head rows, never a head-first reshape.
    return torch.stack([table[h, :, d] for d in range(32) for h in range(8)])


def expand(table, dense=False):
    raw = upstream_layout(table)
    if dense:
        expanded = raw @ torch.as_tensor(FACTOR, dtype=raw.dtype, device=raw.device)
    else:
        # Independent one-hot factorizer argmax, never production indices.
        expanded = raw[:, torch.as_tensor(FACTOR.argmax(0), device=raw.device)]
    return expanded.reshape(32, 8, 64, 64)


def contractions(q, k, alpha, tables, dense=False):
    u, t, v = [expand(table, dense) for table in tables]
    q_logits, k_logits, values = [], [], []
    for h in range(8):
        # Per-query/per-key batched matrix products, distinct from production einsum.
        q_logits.append((q[:, h].transpose(0, 1) @ u[:, h].permute(1, 0, 2)).transpose(0, 1))
        k_logits.append((k[:, h].transpose(0, 1) @ t[:, h].permute(2, 0, 1)).permute(1, 2, 0))
        values.append((alpha[:, h].transpose(0, 1) @ v[:, h].permute(1, 2, 0)).transpose(0, 1))
    return torch.stack(q_logits, 1), torch.stack(k_logits, 1), torch.stack(values, 1)


def attention(q, k, v, bias, tables):
    zero = torch.zeros(q.shape[0], 8, 64, 64, dtype=q.dtype)
    qr, kr, _ = contractions(q, k, zero, tables)
    logits = (q @ k.transpose(-1, -2) + qr + kr) / 32**.5 + bias
    exp = torch.exp(logits - logits.max(-1, keepdim=True).values)
    alpha = exp / exp.sum(-1, keepdim=True)
    _, _, vr = contractions(q, k, alpha, tables)
    return alpha @ v + vr
