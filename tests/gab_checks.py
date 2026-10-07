"""Offline M01 references and acceptance checks; no production math helpers.

Pinned operation attribution is in fixtures/gab_reference.py. The functional
reference below independently specifies projections, GELU, normalization and
per-head bank multiplication rather than calling production forward methods.
"""
from types import SimpleNamespace

import torch
from torch import nn
from .fixtures.gab_reference import PinnedOperation
from chess_rl.model.gab import CompactGAB, SharedTemplateBank

TOLERANCES = {
    torch.float32: {"rtol": 2e-4, "atol": 2e-5},
    torch.float64: {"rtol": 1e-10, "atol": 1e-11},
}


def specimen(dtype):
    torch.manual_seed(7102026)
    model = CompactGAB(SharedTemplateBank(dtype=dtype))
    # Include non-default affine norm weights/biases and nonzero projection biases.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("ln") and name.endswith("weight"):
                parameter.uniform_(0.7, 1.3)
            else:
                parameter.uniform_(-0.08, 0.11)
            assert torch.all(parameter != 0)
    return model


def asymmetric_input(batch, dtype):
    # Generate in double precision then cast, preserving reproducible asymmetry.
    index = torch.arange(batch * 256 * 64, dtype=torch.float64).reshape(batch, 256, 64)
    x = torch.sin(index * 0.017) + 0.3 * torch.cos(index * 0.039)
    x += torch.arange(64, dtype=torch.float64)[None, None, :] * 0.009
    return x.to(dtype).transpose(1, 2).detach().requires_grad_()


def functional_reference(x, p):
    def linear(value, name):
        return value @ p[name + ".weight"].T + p[name + ".bias"]

    def gelu(value):
        return 0.5 * value * (1 + torch.erf(value / (2.0 ** 0.5)))

    def norm(value, name):
        mean = value.mean(-1, keepdim=True)
        variance = ((value - mean) ** 2).mean(-1, keepdim=True)
        return (value - mean) / torch.sqrt(variance + 1e-5) * p[name + ".weight"] + p[name + ".bias"]

    squares = linear(x, "sm1")
    # Explicit square order rather than production's reshape/flatten helper.
    board = torch.cat(squares.unbind(dim=1), dim=-1)
    hidden = norm(gelu(linear(board, "sm2")), "ln1")
    coefficients = norm(gelu(linear(hidden, "sm3")), "ln2")
    heads = coefficients.split(32, dim=-1)
    outputs = [head @ p["bank.weight"].T for head in heads]
    return torch.stack(outputs, dim=1).reshape(x.shape[0], 8, 64, 64)


def pinned_reference(x, p):
    # Build a separate minimal wrapper for the verbatim pinned _sq_bias only.
    ref = SimpleNamespace(num_heads=8, gen_size=32, gab_weight=p["bank.weight"], sm_act=nn.GELU())
    for name, dims in {"sm1": (256, 8), "sm2": (512, 32), "sm3": (32, 256)}.items():
        layer = nn.Linear(*dims, dtype=x.dtype)
        layer.weight = nn.Parameter(p[name + ".weight"])
        layer.bias = nn.Parameter(p[name + ".bias"])
        setattr(ref, name, layer)
        p[name + ".weight"], p[name + ".bias"] = layer.weight, layer.bias
    for name, dim in (("ln1", 32), ("ln2", 256)):
        layer = nn.LayerNorm(dim, dtype=x.dtype)
        layer.weight = nn.Parameter(p[name + ".weight"])
        layer.bias = nn.Parameter(p[name + ".bias"])
        setattr(ref, name, layer)
        p[name + ".weight"], p[name + ".bias"] = layer.weight, layer.bias
    return PinnedOperation._sq_bias(ref, x)


def objective(output):
    i = torch.arange(output.numel(), dtype=torch.float64).reshape(output.shape)
    probe = (torch.sin(i * 0.031) + 0.4 * torch.cos(i * 0.011)).to(output.dtype)
    return ((output * probe).sum() + 0.03 * output.square().sum()) / (output.numel() ** 0.5)


def error(actual, expected, tolerance):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, **tolerance)
    return float((actual - expected).detach().abs().max()) if actual.numel() else 0.0


def numerical_case(batch, dtype, reference):
    model = specimen(dtype)
    x = asymmetric_input(batch, dtype)
    ref_x = x.detach().clone().requires_grad_()
    parameters = dict(model.named_parameters())
    ref_parameters = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    expected = reference(ref_x, ref_parameters)
    actual = model(x)
    assert actual.shape == (batch, 8, 64, 64)
    tol = TOLERANCES[dtype]
    forward_error = error(actual, expected, tol)
    actual_grads = torch.autograd.grad(objective(actual), (x, *parameters.values()))
    ref_grads = torch.autograd.grad(objective(expected), (ref_x, *ref_parameters.values()))
    names = ("input", *parameters.keys())
    gradients = {name: error(a, b, tol) for name, a, b in zip(names, actual_grads, ref_grads)}
    assert all(torch.count_nonzero(gradient) for gradient in actual_grads)
    # Off-diagonal entries cannot be accidentally transposed unnoticed.
    assert not torch.allclose(actual[:, :, 3, 41], actual[:, :, 41, 3])
    return {"batch_size": batch, "dtype": str(dtype), "reference": reference.__name__,
            "input_contiguous": x.is_contiguous(), "tolerance": tol,
            "output_max_abs_error": forward_error, "gradient_max_abs_errors": gradients,
            "gradient_max_abs_error": max(gradients.values()), "all_gradients_nonzero": True}


def sharing_case(dtype=torch.float64):
    torch.manual_seed(8721)
    bank = SharedTemplateBank(dtype=dtype)
    layers = nn.ModuleList([CompactGAB(bank), CompactGAB(bank)])
    assert layers[0].bank.weight is layers[1].bank.weight is bank.weight
    private = [{id(p) for n, p in layer.named_parameters() if not n.startswith("bank.")} for layer in layers]
    assert len(private[0]) == len(private[1]) == 10
    assert private[0].isdisjoint(private[1])
    optimizer = torch.optim.SGD(layers.parameters(), lr=0.01)
    entries = [p for group in optimizer.param_groups for p in group["params"]]
    count = sum(p is bank.weight for p in entries)
    assert count == 1 and len(entries) == len({id(p) for p in entries}) == 21
    losses = [objective(layer(asymmetric_input(batch, dtype))) for layer, batch in zip(layers, (2, 3))]
    individual = [torch.autograd.grad(loss, bank.weight, retain_graph=True)[0] for loss in losses]
    assert all(torch.count_nonzero(gradient) for gradient in individual)
    sum(losses).backward()
    accumulation_error = error(bank.weight.grad, individual[0] + individual[1], TOLERANCES[dtype])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in layers.parameters())
    before = bank.weight.detach().clone()
    expected = before - 0.01 * bank.weight.grad
    optimizer.step()
    step_error = error(bank.weight, expected, TOLERANCES[dtype])
    return {"dtype": str(dtype), "generator_count": 2, "bank_shape": list(bank.weight.shape),
            "same_bank_parameter_identity": True, "independent_generator_parameter_tensors": [10, 10],
            "optimizer_bank_entries": count, "optimizer_unique_parameter_tensors": len(entries),
            "unique_parameter_elements": sum(p.numel() for p in entries),
            "individual_bank_gradient_norms": [float(g.norm()) for g in individual],
            "accumulated_bank_gradient_norm": float(bank.weight.grad.norm()),
            "accumulation_max_abs_error": accumulation_error, "optimizer_step_max_abs_error": step_error}
