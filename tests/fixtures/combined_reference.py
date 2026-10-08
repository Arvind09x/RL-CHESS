"""M06 independent backbone reference, extending accepted M02 explicit math.

Affine matmuls, manual population norms, erf GELU and pinned-layout relative
reference; no production forward/attention/relative operations or Torch layers.
"""
import torch
from .relative_reference import attention

def independent_reference(x, p):
    """Explicit head-wise attention, erf GELU and biased-variance norms.

    Uses matmul and manual normalization, not MHA, SDPA, CompactGAB, block
    forward, functional linear/layer_norm or production projection helpers.
    """
    stages = {}

    def affine(value, name):
        return value @ p[name + ".weight"].T + p[name + ".bias"]

    def gelu(value):
        return value * 0.5 * (1 + torch.erf(value / 2 ** 0.5))

    def norm(value, name):
        centered = value - value.mean(-1, keepdim=True)
        variance = centered.square().mean(-1, keepdim=True)
        return centered / torch.sqrt(variance + 1e-5) * p[name + ".weight"] + p[name + ".bias"]

    x = stages["projection"] = affine(x, "input_projection")
    x = stages["position"] = x + p["square_embeddings"]
    for i in range(8):
        prefix = f"layers.{i}."
        gab = prefix + "self_attn.gab."
        squares = affine(x, gab + "sm1")
        board = torch.cat(squares.unbind(dim=1), dim=-1)
        hidden = norm(gelu(affine(board, gab + "sm2")), gab + "ln1")
        coefficients = norm(gelu(affine(hidden, gab + "sm3")), gab + "ln2").split(32, -1)
        mha = prefix + "self_attn.mha."
        # Split the packed tensor in the reference without altering its initialization.
        weights = p[mha + "in_proj_weight"].split(256, 0)
        biases = p[mha + "in_proj_bias"].split(256, 0)
        q, k, v = [x @ w.T + b for w, b in zip(weights, biases)]
        q, k, v = [z.reshape(x.shape[0], 64, 8, 32).transpose(1, 2) for z in (q,k,v)]
        bias = torch.stack([(coefficients[h] @ p["bank.weight"].T).reshape(x.shape[0],64,64)
                            for h in range(8)], dim=1)
        tables = [p[prefix+"self_attn.relative."+n] for n in ("q","k","v")]
        heads = attention(q,k,v,bias,tables)
        attention_out = heads.transpose(1,2).reshape(x.shape[0],64,256)
        attention_result = stages[f"attention.{i}"] = affine(attention_out, mha + "out_proj")
        x = stages[f"norm1.{i}"] = norm(x + attention_result, prefix + "norm1")
        ff = affine(gelu(affine(x, prefix + "linear1")), prefix + "linear2")
        x = stages[f"norm2.{i}"] = norm(x + ff, prefix + "norm2")
    x = stages["final"] = norm(x, "norm")
    return x, stages
