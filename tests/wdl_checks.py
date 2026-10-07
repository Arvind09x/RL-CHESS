"""M04 independent NumPy analytical forward/backward and diagnostic checks."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chess_rl.adapter import make_state, encode_position
from chess_rl.contracts import WDL_ORDER
from chess_rl.model import WDLHead, EncoderBackbone
from .backbone_checks import asymmetric_input
from .fixtures.wdl_reference import pinned_wdl
from .helpers import fingerprint

TOLERANCES = {torch.float32: {'atol': 2e-5, 'rtol': 2e-4},
              torch.float64: {'atol': 2e-12, 'rtol': 2e-11}}


def compare(actual, expected, dtype):
    expected = torch.as_tensor(expected, dtype=dtype).detach()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
    return float((actual.detach() - expected).abs().max())


def numpy_reference(x, p):
    """Float64 oracle with manual chain rule; no Torch layers or autograd.

    Population variance and its analytic derivative include epsilon. The
    objective is sum(c*y + .03*y**2), c unequal across rows and batches.
    """
    pooled = np.sum(x, axis=1) / 64
    centered = pooled - np.sum(pooled, axis=1, keepdims=True) / 256
    inverse = 1 / np.sqrt(np.sum(centered**2, axis=1, keepdims=True) / 256 + 1e-5)
    z = centered * inverse
    normalized = z * p['last_ln.weight'] + p['last_ln.bias']
    pre = normalized @ p['fc_value_hid.weight'].T + p['fc_value_hid.bias']
    hidden = np.maximum(pre, 0)
    y = hidden @ p['fc_value.weight'].T + p['fc_value.bias']
    c = np.arange(y.size).reshape(y.shape) * .17 - .4
    dy = c + .06*y
    dpre = (dy @ p['fc_value.weight']) * (pre > 0)
    dn = dpre @ p['fc_value_hid.weight']
    dz = dn * p['last_ln.weight']
    dp = inverse * (dz - dz.mean(axis=1, keepdims=True)
                    - z * (dz*z).mean(axis=1, keepdims=True))
    grads = {'output': dy, 'input': np.repeat(dp[:, None, :] / 64, 64, axis=1),
             'last_ln.weight': np.sum(dn*z, axis=0), 'last_ln.bias': np.sum(dn, axis=0),
             'fc_value_hid.weight': dpre.T @ normalized, 'fc_value_hid.bias': dpre.sum(axis=0),
             'fc_value.weight': dy.T @ hidden, 'fc_value.bias': dy.sum(axis=0)}
    return y, grads, pre


def numerical_case(batch, dtype, d_value=128):
    torch.manual_seed(1042026)
    head = WDLHead(d_value=d_value, dtype=dtype)
    with torch.no_grad():
        head.last_ln.weight.uniform_(.6, 1.4)
        head.last_ln.bias.uniform_(-.25, .3)
        head.fc_value_hid.weight.uniform_(-.07, .08)
        head.fc_value_hid.bias.copy_(torch.linspace(-.4, .5, d_value, dtype=dtype))
        head.fc_value.weight.uniform_(-.2, .15)
        head.fc_value.bias.copy_(torch.tensor([.2, -.3, .4], dtype=dtype))
    x = asymmetric_input(batch, dtype, 256)
    assert not x.is_contiguous()
    before = x.detach().clone()
    params = dict(head.named_parameters())
    old_params = {n: p.detach().clone() for n, p in params.items()}
    values = {n: p.detach().double().numpy() for n, p in params.items()}
    y_ref, g_ref, pre = numpy_reference(x.detach().double().numpy(), values)
    assert (pre > 0).any() and (pre < 0).any() and np.min(np.abs(pre)) > 1e-7
    y = head(x)
    assert y.shape == (batch, 3)
    y.retain_grad()
    output_error = compare(y, y_ref, dtype)
    pinned_error = compare(y, pinned_wdl(head, x), dtype)
    c = torch.arange(y.numel(), dtype=dtype).reshape(y.shape) * .17 - .4
    (c*y + .03*y.square()).sum().backward()
    actual_grads = {'output': y.grad, 'input': x.grad, **{n: p.grad for n, p in params.items()}}
    gradient_errors = {n: compare(g, g_ref[n], dtype) for n, g in actual_grads.items()}
    assert all(torch.count_nonzero(g) for g in actual_grads.values())
    assert torch.equal(x.detach(), before)
    assert all(torch.equal(p.detach(), old_params[n]) for n, p in params.items())
    perm = torch.randperm(64)
    permutation_error = compare(head(x[:, perm]), y.detach(), dtype)
    return {'batch': batch, 'dtype': str(dtype), 'd_value': d_value, 'noncontiguous_input': True,
            'tolerance': TOLERANCES[dtype], 'output_max_abs_error': output_error,
            'pinned_body_max_abs_error': pinned_error, 'gradient_max_abs_errors': gradient_errors,
            'square_permutation_max_abs_error': permutation_error,
            'all_gradient_tensors_nonzero_finite': True, 'input_and_parameters_unchanged': True,
            'relu_active_and_inactive': True, 'min_abs_preactivation': float(np.min(np.abs(pre)))}


def synthetic_case():
    head = WDLHead(dtype=torch.float64)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.last_ln.weight.fill_(1.5)
        head.last_ln.bias.fill_(.25)
        head.fc_value_hid.weight[0, 1] = 2
        head.fc_value_hid.bias[0] = .5
        head.fc_value_hid.weight[1, 0] = 1
        head.fc_value_hid.bias[1] = -.5
        head.fc_value_hid.weight[2, :2] = 1
        head.fc_value_hid.bias[2] = .25
        head.fc_value.weight[:, :3] = torch.tensor([[1., 9., 2.], [-1., 8., 1.], [.5, 7., -3.]])
        head.fc_value.bias.copy_(torch.tensor([.1, -.2, .3], dtype=torch.float64))
    channels = torch.tensor([-1., 1.], dtype=torch.float64).repeat(128)
    x = (channels[None, None, :] + (torch.arange(64, dtype=torch.float64)-31.5)[None, :, None]*.125)
    before = x.clone()
    stages = {}
    def capture(name):
        def hook(module, inputs, output):
            stages[name] = (inputs[0].detach().clone(), output.detach().clone())
        return hook
    hooks = [head.last_ln.register_forward_hook(capture('ln')),
             head.fc_value_hid.register_forward_hook(capture('hidden')),
             head.fc_value.register_forward_hook(capture('output'))]
    try:
        y = head(x)
    finally:
        for hook in hooks:
            hook.remove()
    s = 1/math.sqrt(1+1e-5)
    compare(stages['ln'][0], channels[None], torch.float64)
    compare(stages['ln'][1], (.25+1.5*s*channels)[None], torch.float64)
    pre = torch.zeros(1, 128, dtype=torch.float64)
    pre[0, :3] = torch.tensor([1+3*s, -.25-1.5*s, .75], dtype=torch.float64)
    compare(stages['hidden'][1], pre, torch.float64)
    compare(stages['output'][0], pre.clamp_min(0), torch.float64)
    expected = [[2.6+3*s, -.45-3*s, -1.45+1.5*s]]
    compare(y, expected, torch.float64)
    assert head.wdl_order is WDL_ORDER and WDL_ORDER == ('win', 'draw', 'loss')
    assert len(set(y.detach().flatten().tolist())) == 3 and y[0, 1] < 0
    assert torch.equal(x, before)
    return {'pooled_mean': 0, 'pooled_population_variance': 1, 'inverse_std': s,
            'order': list(WDL_ORDER), 'expected_logits': expected[0], 'actual_logits': y.detach().tolist()[0],
            'pool_norm_affine_relu_and_three_rows_independently_checked': True, 'input_unchanged': True}


def constructor_case():
    cases = []
    for dtype in (torch.float32, torch.float64):
        for width in (1, 7, 128):
            torch.manual_seed(404)
            head = WDLHead(d_value=width, dtype=dtype)
            torch.manual_seed(404)
            expected = [nn.LayerNorm(256, dtype=dtype), nn.Linear(256, width, dtype=dtype),
                        nn.Linear(width, 3, dtype=dtype)]
            assert list(head.children()) == [head.last_ln, head.fc_value_hid, head.fc_value]
            assert head.last_ln.normalized_shape == (256,)
            assert head.last_ln.eps == 1e-5 and head.last_ln.elementwise_affine
            assert head.last_ln.bias is not None
            assert torch.all(head.last_ln.weight == 1) and torch.count_nonzero(head.last_ln.bias) == 0
            for a, e in zip(head.children(), expected):
                for name, p in a.named_parameters():
                    assert p.device.type == 'cpu' and p.dtype == dtype and torch.isfinite(p).all()
                    torch.testing.assert_close(p, dict(e.named_parameters())[name], atol=0, rtol=0)
                if isinstance(a, nn.Linear):
                    assert a.bias is not None
                    bound = a.in_features**-.5
                    assert a.weight.abs().max() <= bound and a.bias.abs().max() <= bound
            assert head.d_value == width and not hasattr(head, 'd_policy')
            count = sum(p.numel() for p in head.parameters())
            assert count == 515+260*width and len(list(head.parameters())) == 6
            x = asymmetric_input(2, dtype, 256)
            assert torch.equal(head.train()(x), head.eval()(x))
            # Zero feature variance exercises epsilon; bias-driven output is finite.
            assert torch.isfinite(head(torch.ones(2, 64, 256, dtype=dtype))).all()
            cases.append({'dtype': str(dtype), 'd_value': width, 'parameter_elements': count})
    assert WDLHead().d_value == 128
    return {'seeded_default_constructor_parity_exact': True, 'eps': 1e-5,
            'elementwise_affine': True, 'layernorm_and_linear_biases': True,
            'default_parameter_elements': 33795, 'unique_parameter_tensors': 6,
            'train_eval_identical': True, 'constant_inputs_finite': True, 'cases': cases}


def backbone_diagnostic_case():
    records = json.loads((Path(__file__).parent/'fixtures/adapter_records.json').read_text())
    states = [make_state(r['start_fen'], r['moves_uci'], history_complete=r['history_complete']) for r in records]
    before = [fingerprint(s) for s in states]
    board_ids = [id(s.board) for s in states]
    features = np.stack([encode_position(s).features for s in states])
    x = torch.from_numpy(features.copy()).requires_grad_()
    original = x.detach().clone()
    torch.manual_seed(404)
    backbone = EncoderBackbone().eval()
    head = WDLHead().eval()
    assert head.last_ln is not backbone.norm
    tokens = backbone(x)
    tokens.retain_grad()
    saved_tokens = tokens.detach().clone()
    logits = head(tokens)
    assert logits.shape == (13, 3) and torch.isfinite(logits).all()
    c = torch.arange(logits.numel()).reshape(logits.shape)*.17-.4
    (c*logits + .03*logits.square()).sum().backward()
    for p in [x, tokens, *backbone.parameters(), *head.parameters()]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
    assert torch.equal(x.detach(), original) and torch.equal(tokens.detach(), saved_tokens)
    assert [fingerprint(s) for s in states] == before and [id(s.board) for s in states] == board_ids
    np.testing.assert_array_equal(np.stack([encode_position(s).features for s in states]), features)
    assert sum(p.numel() for p in backbone.parameters()) == 3567168
    return {'fixture_count': 13, 'names': [r['name'] for r in records],
            'contextual_shape': list(tokens.shape), 'logit_shape': list(logits.shape),
            'backbone_parameter_elements': 3567168, 'head_parameter_elements': 33795,
            'separate_pooled_layernorm': True, 'finite_nonzero_gradients_through_backbone_and_head': True,
            'states_all_prefixes_provenance_board_identity_features_and_tokens_unchanged': True,
            'terminal_fixtures_are_tensor_probes_only': True, 'assembled_network_added': False}
