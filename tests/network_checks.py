"""M05 joint checks reusing accepted independent M02/M03/M04 references."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chess_rl.adapter import encode_position, make_state
from chess_rl.contracts import WDL_ORDER
from chess_rl.model import ChessNetwork, EncoderBackbone, PairPolicyHead, WDLHead
from .backbone_checks import asymmetric_input, independent_reference, objective, TOLERANCES
from .policy_checks import component_reference
from .wdl_checks import numpy_reference
from .helpers import fingerprint


def compare(a, b, dtype):
    b = torch.as_tensor(b, dtype=dtype)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, **TOLERANCES[dtype])
    return float((a.detach() - b.detach()).abs().max())


def value_objective(y):
    c = torch.arange(y.numel(), dtype=y.dtype).reshape(y.shape) * .17 - .4
    return (c*y + .03*y.square()).sum()


def numerical_case(batch, dtype, d_policy=128, d_value=128):
    torch.manual_seed(1052026)
    model = ChessNetwork(dropout=0, dtype=dtype, d_policy=d_policy, d_value=d_value)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ('norm' in name or '.ln' in name or 'last_ln' in name) and name.endswith('weight'):
                p.uniform_(.7, 1.3)
            else:
                p.uniform_(-.04, .05)
        model.wdl_head.fc_value_hid.bias.copy_(torch.linspace(-.4, .5, d_value, dtype=dtype))
    x = asymmetric_input(batch, dtype)
    assert not x.is_contiguous()
    before = x.detach().clone()
    parameters = dict(model.named_parameters())
    saved = {n: p.detach().clone() for n, p in parameters.items()}
    ref_x = x.detach().clone().requires_grad_()
    # Deduplicated clone: exactly one bank is used throughout the independent path.
    ref = {n: p.detach().clone().requires_grad_() for n, p in parameters.items()}
    bp = {n.removeprefix('backbone.'): p for n, p in ref.items() if n.startswith('backbone.')}
    pp = {n.removeprefix('policy_head.'): p for n, p in ref.items() if n.startswith('policy_head.')}
    vp = {n.removeprefix('wdl_head.'): p.detach().double().numpy() for n, p in ref.items() if n.startswith('wdl_head.')}
    tokens, _ = independent_reference(ref_x, bp)
    policy_ref = component_reference(tokens, pp)
    value_ref, vg, pre = numpy_reference(tokens.detach().double().numpy(), vp)
    assert (pre > 0).any() and (pre < 0).any() and np.min(np.abs(pre)) > 1e-7
    outputs = model(x)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    policy, value = outputs
    assert policy.shape == (batch, 4352) and value.shape == (batch, 3)
    output_errors = {'policy': compare(policy, policy_ref, dtype), 'wdl': compare(value, value_ref, dtype)}
    # NumPy's manual WDL chain rule supplies the final-token cotangent.
    # The independent backbone propagates the sum of policy and WDL cotangents.
    ref_loss = objective(policy_ref) + (tokens * torch.as_tensor(vg['input'], dtype=dtype)).sum()
    differentiable_names = [n for n in ref if not n.startswith('wdl_head.')]
    ref_grads = torch.autograd.grad(ref_loss, (ref_x, *(ref[n] for n in differentiable_names)))
    expected = dict(zip(('input', *differentiable_names), ref_grads))
    expected.update({'wdl_head.'+n: torch.as_tensor(vg[n], dtype=dtype) for n in vp})
    actual = torch.autograd.grad(objective(policy) + value_objective(value), (x, *parameters.values()))
    errors = {n: compare(g, expected[n], dtype) for n, g in zip(('input', *parameters), actual)}
    assert all(torch.count_nonzero(g) for g in actual)
    assert torch.equal(x.detach(), before)
    assert all(torch.equal(p.detach(), saved[n]) for n, p in parameters.items())
    return {'batch': batch, 'dtype': str(dtype), 'd_policy': d_policy, 'd_value': d_value,
            'dropout': 0, 'noncontiguous_asymmetric_input': True, 'tolerance': TOLERANCES[dtype],
            'output_max_abs_errors': output_errors, 'gradient_max_abs_errors': errors,
            'gradient_max_abs_error': max(errors.values()), 'unique_parameter_tensors': len(parameters),
            'all_gradient_tensors_finite_nonzero': True, 'inputs_and_parameters_unchanged': True,
            'relu_active_and_inactive': True, 'min_abs_relu_preactivation': float(np.min(np.abs(pre)))}


def routing_and_joint_case(dtype):
    torch.manual_seed(505)
    model = ChessNetwork(dropout=0, dtype=dtype)
    x = asymmetric_input(2, dtype)
    seen = {'backbone': [], 'final_norm': [], 'policy': [], 'wdl': []}
    hooks = [model.backbone.register_forward_hook(lambda m, a, y: seen['backbone'].append(y)),
             model.backbone.norm.register_forward_hook(lambda m, a, y: seen['final_norm'].append(y)),
             model.policy_head.register_forward_pre_hook(lambda m, a: seen['policy'].append(a[0])),
             model.wdl_head.register_forward_pre_hook(lambda m, a: seen['wdl'].append(a[0]))]
    try:
        policy, value = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert all(len(v) == 1 for v in seen.values())
    tokens = seen['backbone'][0]
    assert tokens is seen['final_norm'][0] is seen['policy'][0] is seen['wdl'][0]
    saved_tokens = tokens.detach().clone()
    assert model.wdl_head.last_ln is not model.backbone.norm
    assert model.wdl_order is model.wdl_head.wdl_order is WDL_ORDER
    params = list(model.parameters())
    targets = (x, tokens, *params)
    gp = torch.autograd.grad(objective(policy), targets, retain_graph=True, allow_unused=True)
    gv = torch.autograd.grad(value_objective(value), targets, retain_graph=True, allow_unused=True)
    joint = torch.autograd.grad(objective(policy)+value_objective(value), targets)
    errors = [compare(j, (torch.zeros_like(j) if p is None else p)
                     + (torch.zeros_like(j) if v is None else v), dtype) for j, p, v in zip(joint, gp, gv)]
    bp_ids = {id(p) for p in model.backbone.parameters()}
    for target, p, v in zip(targets, gp, gv):
        if target is x or target is tokens or id(target) in bp_ids:
            assert p is not None and v is not None and torch.count_nonzero(p) and torch.count_nonzero(v)
    ph_ids = {id(p) for p in model.policy_head.parameters()}
    vh_ids = {id(p) for p in model.wdl_head.parameters()}
    for p, a, b in zip(params, gp[2:], gv[2:]):
        if id(p) in ph_ids:
            assert a is not None and torch.count_nonzero(a) and b is None
        if id(p) in vh_ids:
            assert b is not None and torch.count_nonzero(b) and a is None
    assert torch.equal(tokens.detach(), saved_tokens)
    generators = [b.self_attn.gab for b in model.backbone.layers]
    assert len({id(g) for g in generators}) == 8
    assert all(g.bank is model.backbone.bank for g in generators)
    private = [{id(p) for p in g.parameters() if p is not model.backbone.bank.weight} for g in generators]
    assert all(len(s) == 10 for s in private) and len(set.union(*private)) == 80
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    entries = [p for group in optimizer.param_groups for p in group['params']]
    assert len(entries) == len({id(p) for p in entries}) == 191
    assert sum(p is model.backbone.bank.weight for p in entries) == 1
    return {'dtype': str(dtype), 'backbone_evaluations': 1, 'same_encoder_final_token_object': True,
            'separate_pooled_norm': True, 'shared_wdl_order_identity': True, 'independent_generators': 8,
            'private_generator_tensor_counts': list(map(len, private)), 'optimizer_unique_tensors': 191,
            'optimizer_bank_entries': 1, 'both_branches_nonzero_through_input_tokens_all_backbone_parameters': True,
            'head_gradients_only_from_own_branch': True, 'joint_gradient_sum_max_abs_error': max(errors)}


def adapter_case(dtype):
    records = json.loads((Path(__file__).parent/'fixtures/adapter_records.json').read_text())
    states = [make_state(r['start_fen'], r['moves_uci'], history_complete=r['history_complete']) for r in records]
    before = [fingerprint(s) for s in states]
    boards = [id(s.board) for s in states]
    features = np.stack([encode_position(s).features for s in states])
    x = torch.from_numpy(features.copy()).to(dtype).requires_grad_()
    original = x.detach().clone()
    torch.manual_seed(505)
    model = ChessNetwork(dtype=dtype).eval()
    policy, value = model(x)
    assert policy.shape == (13, 4352) and value.shape == (13, 3)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    (objective(policy) + value_objective(value)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in (x, *model.parameters()))
    assert torch.equal(x.detach(), original)
    assert [fingerprint(s) for s in states] == before and [id(s.board) for s in states] == boards
    np.testing.assert_array_equal(features, np.stack([encode_position(s).features for s in states]))
    # Count actual allocated gradient tensors, independently of analytic count.
    return {'dtype': str(dtype), 'fixture_count': len(records), 'names': [r['name'] for r in records],
            'shapes': [list(policy.shape), list(value.shape)], 'all_outputs_and_gradients_finite_nonzero': True,
            'states_all_prefixes_provenance_boards_and_features_unchanged': True,
            'terminal_fixtures_are_tensor_probes_only': True,
            'allocated_parameter_tensor_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'allocated_gradient_tensor_bytes': sum(p.grad.numel()*p.grad.element_size() for p in model.parameters())}


def dropout_case():
    torch.manual_seed(505)
    model = ChessNetwork()
    for b in model.backbone.layers:
        assert b.self_attn.mha.dropout == .1
        assert all(d.p == .1 for d in (b.dropout, b.dropout1, b.dropout2))
    x = asymmetric_input(2, torch.float32)
    with torch.no_grad():
        model.train()
        assert all(m.training for m in model.modules())
        torch.manual_seed(1); a = model(x)
        torch.manual_seed(1); replay = model(x)
        torch.manual_seed(2); b = model(x)
        assert all(torch.equal(y, z) for y, z in zip(a, replay))
        assert all(not torch.equal(y, z) for y, z in zip(a, b))
        model.eval()
        assert all(not m.training for m in model.modules())
        e, repeated = model(x), model(x)
        assert all(torch.equal(y, z) for y, z in zip(e, repeated))
        assert all(not torch.equal(y, z) for y, z in zip(e, a))
        zero = ChessNetwork(dropout=0)
        assert all(torch.equal(y, z) for y, z in zip(zero.train()(x), zero.eval()(x)))
    return {'default_dropout': .1, 'all_module_train_eval_flags_propagate': True,
            'both_heads_stochastic_in_train': True, 'seed_replay_exact': True,
            'both_eval_outputs_repeat_exactly': True, 'zero_dropout_train_eval_equal': True,
            'train_max_abs_differences': [float((y-z).abs().max()) for y, z in zip(a, b)]}


def constructor_and_memory_case():
    cases = []
    for dtype in (torch.float32, torch.float64):
        for policy_width, value_width in ((128, 128), (7, 11)):
            torch.manual_seed(505)
            model = ChessNetwork(d_policy=policy_width, d_value=value_width, device='cpu', dtype=dtype)
            torch.manual_seed(505)
            components = [EncoderBackbone(dtype=dtype), PairPolicyHead(d_policy=policy_width, dtype=dtype),
                          WDLHead(d_value=value_width, dtype=dtype)]
            assert list(model.children()) == [model.backbone, model.policy_head, model.wdl_head]
            for actual, expected in zip(model.children(), components):
                assert type(actual) is type(expected)
                for p, q in zip(actual.parameters(), expected.parameters()):
                    assert p.dtype == dtype and p.device.type == 'cpu'
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
            params = list(model.parameters())
            assert len(params) == len({id(p) for p in params}) == 191
            # Shape arithmetic independent of the accepted component totals.
            backbone = (256*128+256) + 64*256 + 4096*32 + 2*256 + 8*(
                3*256*256+3*256 + 256*256+256 + 2*(256*256+256) + 4*256
                + 8*256+8 + 32*512+32 + 2*32 + 256*32+256 + 2*256)
            total = backbone + 2*256*policy_width + 4*policy_width + 2*256 + 257*value_width + 3*value_width+3
            assert total == sum(p.numel() for p in params)
            assert sum(p.numel() for p in model.backbone.parameters()) == backbone == 3567168
            assert model.policy_head.d_policy == policy_width and model.wdl_head.d_value == value_width
            byte_count = sum(p.numel()*p.element_size() for p in params)
            if policy_width == value_width == 128:
                assert total == 3667011
            cases.append({'dtype': str(dtype), 'd_policy': policy_width, 'd_value': value_width,
                          'parameter_elements': total, 'unique_parameter_tensors': len(params),
                          'parameter_tensor_bytes': byte_count, 'gradient_tensor_bytes_if_all_allocated': byte_count,
                          'parameter_plus_gradient_tensor_bytes': 2*byte_count})
    # Whole-model dtype movement also retains the shared bank identity.
    moved = ChessNetwork().to(dtype=torch.float64)
    assert all(p.dtype == torch.float64 for p in moved.parameters())
    assert all(b.self_attn.gab.bank.weight is moved.backbone.bank.weight for b in moved.backbone.layers)
    return {'seeded_direct_component_constructor_parity_exact': True, 'coherent_device_dtype': True,
            'whole_network_dtype_move_preserves_sharing': True,
            'independent_shape_arithmetic_matches_deduplicated_count': True,
            'default_components': {'backbone': 3567168, 'policy_head': 66048, 'wdl_head': 33795},
            'memory_scope': 'tensor payload only; excludes activations, input/output, optimizer state, allocator and process overhead',
            'cases': cases}
