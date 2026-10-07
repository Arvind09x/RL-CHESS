"""Independent M02 CPU acceptance math; never calls production math helpers."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chess_rl.adapter import encode_position, make_state
from chess_rl.model import EncoderBackbone
from .helpers import fingerprint

TOLERANCES = {
    torch.float32: {"atol": 3e-5, "rtol": 5e-4},
    torch.float64: {"atol": 2e-11, "rtol": 2e-10},
}


def asymmetric_input(batch, dtype, width=128):
    index = torch.arange(batch * width * 64, dtype=torch.float64).reshape(batch, width, 64)
    x = torch.sin(index * 0.017) + 0.3 * torch.cos(index * 0.039)
    x += torch.arange(64, dtype=torch.float64)[None, None, :] * 0.009
    return x.to(dtype).transpose(1, 2).detach().requires_grad_()


def objective(output):
    index = torch.arange(output.numel(), dtype=torch.float64).reshape(output.shape)
    probe = (torch.sin(index * 0.031) + 0.4 * torch.cos(index * 0.011)).to(output.dtype)
    return ((output * probe).sum() + 0.03 * output.square().sum()) / output.numel() ** 0.5


def error(actual, expected, dtype):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
    return float((actual - expected).detach().abs().max())


def specimen(dtype):
    torch.manual_seed(8102026)
    model = EncoderBackbone(dropout=0, dtype=dtype)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ("norm" in name or ".ln" in name) and name.endswith("weight"):
                p.uniform_(0.7, 1.3)
            else:
                p.uniform_(-0.04, 0.05)
            assert torch.all(p != 0)
    return model


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
        heads = []
        for h in range(8):
            sl = slice(h * 32, (h + 1) * 32)
            gab_bias = (coefficients[h] @ p["bank.weight"].T).reshape(x.shape[0], 64, 64)
            logits = (q[..., sl] @ k[..., sl].transpose(-1, -2)) / 32 ** 0.5 + gab_bias
            # Explicit stable softmax over destination squares.
            exp = torch.exp(logits - logits.max(-1, keepdim=True).values)
            probability = exp / exp.sum(-1, keepdim=True)
            heads.append(probability @ v[..., sl])
        attention = stages[f"attention.{i}"] = affine(torch.cat(heads, -1), mha + "out_proj")
        x = stages[f"norm1.{i}"] = norm(x + attention, prefix + "norm1")
        ff = affine(gelu(affine(x, prefix + "linear1")), prefix + "linear2")
        x = stages[f"norm2.{i}"] = norm(x + ff, prefix + "norm2")
    x = stages["final"] = norm(x, "norm")
    return x, stages


def numerical_case(batch, dtype):
    model = specimen(dtype)
    x = asymmetric_input(batch, dtype)
    assert not x.is_contiguous()
    original = x.detach().clone()
    ref_x = x.detach().clone().requires_grad_()
    params = dict(model.named_parameters())
    ref_params = {name: p.detach().clone().requires_grad_() for name, p in params.items()}
    expected, ref_stages = independent_reference(ref_x, ref_params)
    stages, hooks = {}, []

    def capture(name):
        return lambda module, args, output: stages.__setitem__(name, output)

    hooks.append(model.input_projection.register_forward_hook(capture("projection")))
    hooks.append(model.layers[0].register_forward_pre_hook(
        lambda module, args: stages.__setitem__("position", args[0])))
    for i, block in enumerate(model.layers):
        for name, module in (("attention", block.self_attn), ("norm1", block.norm1), ("norm2", block.norm2)):
            hooks.append(module.register_forward_hook(capture(f"{name}.{i}")))
    hooks.append(model.norm.register_forward_hook(capture("final")))
    try:
        actual = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert actual.shape == (batch, 64, 256)
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
    stage_errors = {name: error(stages[name], value, dtype) for name, value in ref_stages.items()}
    actual_grads = torch.autograd.grad(objective(actual), (x, *params.values()))
    ref_grads = torch.autograd.grad(objective(expected), (ref_x, *ref_params.values()))
    gradients = {name: error(a, b, dtype) for name, a, b in zip(("input", *params), actual_grads, ref_grads)}
    assert all(torch.count_nonzero(g) for g in actual_grads)
    return {"batch_size": batch, "dtype": str(dtype), "input_contiguous": False,
            "tolerance": TOLERANCES[dtype], "output_max_abs_error": stage_errors["final"],
            "stage_max_abs_errors": stage_errors, "gradient_max_abs_errors": gradients,
            "gradient_max_abs_error": max(gradients.values()), "all_gradients_nonzero": True,
            "unique_parameter_tensors": len(params), "input_unchanged": True}


def sharing_case(dtype):
    torch.manual_seed(8722)
    model = EncoderBackbone(dropout=0, dtype=dtype)
    generators = [block.self_attn.gab for block in model.layers]
    assert all(g.bank is model.bank and g.bank.weight is model.bank.weight for g in generators)
    private = [{id(p) for p in b.parameters() if p is not model.bank.weight} for b in model.layers]
    generator_private = [{id(p) for p in g.parameters() if p is not model.bank.weight} for g in generators]
    assert all(len(ids) == 22 for ids in private)
    assert all(len(ids) == 10 for ids in generator_private)
    assert len(set.union(*private)) == sum(map(len, private))
    assert len({id(g) for g in generators}) == len({id(b) for b in model.layers}) == 8
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    entries = [p for group in optimizer.param_groups for p in group["params"]]
    assert sum(p is model.bank.weight for p in entries) == 1
    assert len(entries) == len({id(p) for p in entries}) == 182
    # Isolate each layer's bank contribution, then backpropagate their sum.
    losses = [objective(block(asymmetric_input(1, dtype, 256))) for block in model.layers]
    individual = [torch.autograd.grad(loss, model.bank.weight, retain_graph=True)[0] for loss in losses]
    assert all(torch.count_nonzero(g) for g in individual)
    sum(losses).backward()
    accumulation_error = error(model.bank.weight.grad, torch.stack(individual).sum(0), dtype)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for block in model.layers for p in block.parameters())
    before = model.bank.weight.detach().clone()
    expected = before - 0.01 * model.bank.weight.grad
    optimizer.step()
    step_error = error(model.bank.weight, expected, dtype)
    return {"dtype": str(dtype), "generator_count": 8, "bank_parameter_count": 1,
            "private_block_tensors": list(map(len, private)), "private_generator_tensors": list(map(len, generator_private)),
            "optimizer_bank_entries": 1, "optimizer_unique_parameter_tensors": len(entries),
            "backbone_parameter_elements": sum(p.numel() for p in entries),
            "individual_bank_gradient_norms": [float(g.norm()) for g in individual],
            "accumulation_max_abs_error": accumulation_error, "optimizer_step_max_abs_error": step_error}


def dropout_case():
    torch.manual_seed(123)
    model = EncoderBackbone()
    x = asymmetric_input(2, torch.float32)
    original = x.detach().clone()
    with torch.no_grad():
        model.train()
        torch.manual_seed(1)
        train1 = model(x)
        torch.manual_seed(1)
        replay = model(x)
        torch.manual_seed(2)
        train2 = model(x)
        assert torch.equal(train1, replay) and not torch.equal(train1, train2)
        model.eval()
        eval1, eval2 = model(x), model(x)
        assert torch.equal(eval1, eval2) and not torch.equal(eval1, train1)
        # Verify that each retained dropout site can independently affect output.
        for block in model.layers:
            block.self_attn.mha.dropout = 0
            for d in (block.dropout, block.dropout1, block.dropout2):
                d.p = 0
        site_differences = {}
        first = model.layers[0]
        sites = {"attention_probability": (first.self_attn.mha, "dropout"),
                 "attention_residual": (first.dropout1, "p"),
                 "ffn_hidden": (first.dropout, "p"), "ffn_residual": (first.dropout2, "p")}
        for name, (site, attr) in sites.items():
            setattr(site, attr, 0.1)
            model.train()
            torch.manual_seed(3)
            a = model(x)
            torch.manual_seed(4)
            b = model(x)
            assert not torch.equal(a, b)
            model.eval()
            assert torch.equal(model(x), eval1)
            site_differences[name] = float((a - b).abs().max())
            setattr(site, attr, 0)
        model.train()
        assert torch.equal(model(x), eval1)
    assert torch.equal(x.detach(), original)
    return {"training_default": 0.1, "seeded_train_replay_exact": True,
            "train_outputs_differ": True, "eval_outputs_repeat_exactly": True,
            "zero_dropout_train_equals_eval": True, "isolated_site_max_differences": site_differences,
            "train_max_difference": float((train1 - train2).abs().max()), "input_unchanged": True}


def adapter_case():
    records = json.loads((Path(__file__).parent / "fixtures/adapter_records.json").read_text())
    states = [make_state(r["start_fen"], r["moves_uci"], history_complete=r["history_complete"]) for r in records]
    before = [fingerprint(s) for s in states]
    board_ids = [id(s.board) for s in states]
    features = np.stack([encode_position(s).features for s in states])
    tensor = torch.from_numpy(features).requires_grad_()
    original = tensor.detach().clone()
    torch.manual_seed(1337)
    model = EncoderBackbone(dropout=0).eval()
    output = model(tensor)
    assert output.shape == (13, 64, 256) and torch.isfinite(output).all()
    objective(output).backward()
    assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(tensor.detach(), original)
    assert [fingerprint(s) for s in states] == before
    assert [id(s.board) for s in states] == board_ids
    np.testing.assert_array_equal(np.stack([encode_position(s).features for s in states]), features)
    return {"fixture_count": 13, "names": [r["name"] for r in records],
            "output_shape": list(output.shape), "finite_outputs_and_gradients": True,
            "states_all_prefixes_provenance_board_identity_and_features_unchanged": True,
            "terminal_fixtures_included_as_tensor_probes_only": True}


def constructor_case():
    torch.manual_seed(17)
    model = EncoderBackbone()
    assert len(model.layers) == 8
    assert model.square_embeddings.shape == (64, 256) and torch.count_nonzero(model.square_embeddings) == 0
    assert model.input_projection.weight.shape == (256, 128)
    for block in model.layers:
        mha = block.self_attn.mha
        assert mha.in_proj_weight.shape == (768, 256)
        assert mha.q_proj_weight is mha.k_proj_weight is mha.v_proj_weight is None
        assert mha.batch_first and mha.num_heads == 8 and mha.head_dim == 32 and mha.dropout == 0.1
        assert mha.bias_k is mha.bias_v is None and not mha.add_zero_attn
        assert torch.count_nonzero(mha.in_proj_bias) == torch.count_nonzero(mha.out_proj.bias) == 0
        assert mha.in_proj_weight.abs().max() <= (6 / (768 + 256)) ** 0.5
        assert block.linear1.weight.shape == block.linear2.weight.shape == (256, 256)
        assert block.activation.approximate == "none"
        assert all(d.p == 0.1 and not d.inplace for d in (block.dropout, block.dropout1, block.dropout2))
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert module.eps == 1e-5 and module.elementwise_affine and module.bias is not None
            assert torch.all(module.weight == 1) and torch.count_nonzero(module.bias) == 0
        if isinstance(module, nn.Linear):
            assert module.bias is not None
            bound = module.in_features ** -0.5
            assert module.weight.abs().max() <= bound and module.bias.abs().max() <= bound
    # Seeded exact constructor parity catches custom resets or split-QKV replacement.
    for i, block in enumerate(model.layers):
        seed = 901 + i
        torch.manual_seed(seed)
        attention = type(block.self_attn)(model.bank)
        torch.manual_seed(seed)
        expected = nn.MultiheadAttention(256, 8, dropout=0.1, bias=True, batch_first=True)
        for name, p in attention.mha.named_parameters():
            torch.testing.assert_close(p, dict(expected.named_parameters())[name], rtol=0, atol=0)
    assert all(not torch.equal(model.layers[0].self_attn.mha.in_proj_weight, block.self_attn.mha.in_proj_weight)
               for block in model.layers[1:])
    count = sum(p.numel() for p in model.parameters())
    assert count == 3_567_168
    assert len([d for d in model.modules() if isinstance(d, nn.Dropout)]) == 24
    return {"backbone_parameter_elements": count, "unique_parameter_tensors": len(list(model.parameters())),
            "packed_mha_constructor_exact": True, "separately_constructed_blocks": True,
            "square_embeddings_initially_zero": True, "layernorm_epsilon": 1e-5,
            "all_retained_biases_enabled": True, "training_dropout": 0.1,
            "dropout_module_count": 24, "attention_probability_dropout_sites": 8}
