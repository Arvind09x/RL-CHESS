"""M03 independent scalar, component-wise gradient and diagnostic checks."""
import math

import chess
import numpy as np
import torch
from torch import nn

from chess_rl.adapter import make_state, move_to_id, id_to_move, legal_mask
from chess_rl.adapter.moves import MOVE_VOCABULARY
from chess_rl.model import PairPolicyHead, EncoderBackbone
from .backbone_checks import asymmetric_input, objective
from .fixtures.policy_reference import pinned_policy
from .helpers import PROMOTION, BLACK_PROMOTION

TOLERANCES = {
    torch.float32: {"atol": 2e-5, "rtol": 2e-4},
    torch.float64: {"atol": 2e-12, "rtol": 2e-11},
}
SELECTED_PAIRS = [(0, 0), (0, 63), (1, 0), (12, 28), (28, 12), (48, 63), (52, 60), (63, 63)]


def compare(actual, expected, dtype):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
    return float((actual - expected).detach().abs().max())


def specimen(dtype, d_policy):
    torch.manual_seed(1032026)
    head = PairPolicyHead(d_policy=d_policy, dtype=dtype)
    with torch.no_grad():
        for p in head.parameters():
            p.uniform_(-0.05, 0.04)
            assert torch.all(p != 0)
    return head


def component_reference(x, parameters):
    """Independent differentiable path: scalar-depth products and slot lists.

    No production forward, Linear, einsum or batched pair matmul. Projection
    uses affine-free matrix products; pair depth is summed explicitly.
    """
    q = x @ parameters['proj_sq_from.weight'].T
    k = x @ parameters['proj_sq_to.weight'].T
    width = q.shape[-1]
    scores = sum(q[:, :, d, None] * k[:, None, :, d] for d in range(width)) / math.sqrt(width)
    adjustments = [sum(k[:, 56 + g, d] * parameters['promo_bias_proj.weight'][p, d]
                       for d in range(width)) * math.sqrt(width)
                   for g in range(8) for p in range(4)]
    ordinary = [scores[:, i, j] for i in range(64) for j in range(64)]
    promotions = [scores[:, 48 + f, 56 + g] + adjustments[4*g + p]
                  for f in range(8) for g in range(8) for p in range(4)]
    return torch.stack(ordinary + promotions, dim=1)


def scalar_reference(head, x):
    """Python math.fsum calculations in double precision, including projections.

    Check selected ordinary pairs and every promotion slot, using explicit
    adapter vocabulary tuples rather than any production reshape/slicing.
    """
    values = x.detach().double().tolist()
    weights = {n: p.detach().double().tolist() for n, p in head.named_parameters()}
    results = []
    ids = [64*i+j for i, j in SELECTED_PAIRS] + list(range(4096, 4352))
    for board in values:
        q = [[math.fsum(a*b for a, b in zip(token, row))
              for row in weights['proj_sq_from.weight']] for token in board]
        k = [[math.fsum(a*b for a, b in zip(token, row))
              for row in weights['proj_sq_to.weight']] for token in board]
        result = []
        for slot in ids:
            origin, destination, piece = MOVE_VOCABULARY[slot]
            base = math.fsum(a*b for a, b in zip(q[origin], k[destination])) / math.sqrt(head.d_policy)
            if piece is not None:
                p = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT].index(piece)
                base += math.sqrt(head.d_policy) * math.fsum(a*b for a, b in zip(
                    k[destination], weights['promo_bias_proj.weight'][p]))
            result.append(base)
        results.append(result)
    return ids, torch.tensor(results, dtype=x.dtype)


def numerical_case(batch, dtype, d_policy=128):
    head = specimen(dtype, d_policy)
    x = asymmetric_input(batch, dtype, 256)
    assert not x.is_contiguous()
    before = x.detach().clone()
    ref_x = x.detach().clone().requires_grad_()
    params = dict(head.named_parameters())
    ref_params = {n: p.detach().clone().requires_grad_() for n, p in params.items()}
    actual = head(x)
    assert actual.shape == (batch, 4352)
    expected = component_reference(ref_x, ref_params)
    output_error = compare(actual, expected, dtype)
    pinned_error = compare(actual, pinned_policy(head, x), dtype)
    ids, scalars = scalar_reference(head, x)
    scalar_error = compare(actual[:, ids], scalars, dtype)
    actual_grads = torch.autograd.grad(objective(actual), (x, *params.values()))
    ref_grads = torch.autograd.grad(objective(expected), (ref_x, *ref_params.values()))
    gradient_errors = {n: compare(a, b, dtype) for n, a, b in zip(
        ('input', *params), actual_grads, ref_grads)}
    assert all(torch.count_nonzero(g) for g in actual_grads)
    assert torch.equal(x.detach(), before)
    return {'batch_size': batch, 'dtype': str(dtype), 'd_policy': d_policy,
            'input_contiguous': False, 'tolerance': TOLERANCES[dtype],
            'output_max_abs_error': output_error, 'pinned_max_abs_error': pinned_error,
            'scalar_max_abs_error': scalar_error, 'scalar_ordinary_ids': ids[:len(SELECTED_PAIRS)],
            'scalar_promotion_slots': 256, 'gradient_max_abs_errors': gradient_errors,
            'all_gradients_nonzero_and_finite': True, 'input_unchanged': True,
            'parameter_elements': sum(p.numel() for p in params.values())}


def exact_order_and_dependence_case():
    head = PairPolicyHead(d_policy=4, dtype=torch.float64)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.proj_sq_from.weight[0, 0] = 1
        head.proj_sq_to.weight[0, 1] = 1
        head.promo_bias_proj.weight[:, 0] = torch.tensor([1., 2., 3., 4.])
    x = torch.zeros(2, 64, 256, dtype=torch.float64)
    for b in range(2):
        for square in range(64):
            x[b, square, 0] = square + 1 + b * 100
            x[b, square, 1] = square + 2 + b * 200
    actual = head(x)
    for b in range(2):
        for i in range(64):
            for j in range(64):
                assert actual[b, 64*i+j] == (i+1+b*100) * (j+2+b*200) / 2
        for f in range(8):
            for g in range(8):
                for p in range(4):
                    base = (49+f+b*100) * (58+g+b*200) / 2
                    adjustment = 2 * (58+g+b*200) * (p+1)
                    assert actual[b, 4096+4*(8*f+g)+p] == base + adjustment
    # Remove base scores to isolate destination-key promotion adjustments.
    with torch.no_grad():
        head.proj_sq_from.weight.zero_()
    adjustment_only = head(x)
    origin_change = x.clone()
    origin_change[:, 48:56] += 101
    assert torch.equal(head(origin_change), adjustment_only)
    irrelevant_change = x.clone()
    irrelevant_change[:, 56:64, 2:] += 203
    assert torch.equal(head(irrelevant_change), adjustment_only)
    destination_change = x.clone()
    destination_change[:, 59, 1] += 5
    delta = head(destination_change) - adjustment_only
    for f in range(8):
        for g in range(8):
            for p in range(4):
                assert torch.all(delta[:, 4096+4*(8*f+g)+p] == (10*(p+1) if g == 3 else 0))
    assert torch.count_nonzero(delta[:, :4096]) == 0
    return {'exact_ordinary_slots_per_batch': 4096, 'exact_promotion_slots_per_batch': 256,
            'batch_size': 2, 'piece_order': ['queen', 'rook', 'bishop', 'knight'],
            'destination_fastest': True, 'base_divisor': 2, 'adjustment_multiplier': 2,
            'origin_and_unprojected_extra_features_do_not_change_adjustment': True,
            'only_selected_destination_file_changes': True, 'impossible_geometry_retained': True}


def constructor_case():
    results = []
    for dtype in (torch.float32, torch.float64):
        for width in (1, 7, 128):
            torch.manual_seed(305)
            head = PairPolicyHead(d_policy=width, dtype=dtype)
            torch.manual_seed(305)
            expected = [nn.Linear(256, width, bias=False, dtype=dtype),
                        nn.Linear(256, width, bias=False, dtype=dtype),
                        nn.Linear(width, 4, bias=False, dtype=dtype)]
            assert head.d_policy == width
            assert len(list(head.parameters())) == 3
            for actual, reference in zip((head.proj_sq_from, head.proj_sq_to, head.promo_bias_proj), expected):
                assert actual.bias is None and actual.weight.device.type == 'cpu' and actual.weight.dtype == dtype
                assert actual.weight.abs().max() <= actual.in_features ** -0.5
                torch.testing.assert_close(actual.weight, reference.weight, rtol=0, atol=0)
            count = sum(p.numel() for p in head.parameters())
            assert count == 516*width
            assert torch.equal(head(torch.zeros(2, 64, 256, dtype=dtype)), torch.zeros(2, 4352, dtype=dtype))
            x = asymmetric_input(1, dtype, 256)
            assert torch.equal(head.train()(x), head.eval()(x))
            results.append({'dtype': str(dtype), 'd_policy': width, 'parameter_elements': count})
    assert PairPolicyHead().d_policy == 128
    return {'constructors_match_seeded_linears_exactly': True, 'biases_absent': True,
            'default_d_policy': 128, 'default_parameter_elements': 66048,
            'unique_parameter_tensors': 3, 'zero_input_zero_logits': True,
            'train_eval_identical': True, 'cases': results}


def adapter_ids_case():
    cases = [(make_state(), 'e2e4', 796),
             (make_state(moves_uci=['e2e4']), 'e7e5', 796)]
    for fen, prefix in ((PROMOTION, 'e7e8'), (BLACK_PROMOTION, 'e2e1')):
        cases += [(make_state(fen), prefix + suffix, 4240+p) for p, suffix in enumerate('qrbn')]
    head = specimen(torch.float64, 7)
    x = asymmetric_input(1, torch.float64, 256)
    output = head(x)
    ids, scalars = scalar_reference(head, x)
    checked = []
    for state, uci, slot in cases:
        before = state.board.fen(), list(state.board.move_stack)
        move = chess.Move.from_uci(uci)
        assert move_to_id(state, move) == slot and id_to_move(state, slot) == move
        assert legal_mask(state)[slot]
        compare(output[:, slot], scalars[:, ids.index(slot)], torch.float64)
        assert (state.board.fen(), state.board.move_stack) == before
        checked.append({'uci': uci, 'id': slot})
    assert not legal_mask(make_state(PROMOTION))[52*64+60]
    assert torch.isfinite(output[:, 52*64+60]).all()  # Head retains ordinary promotion pair.
    assert not legal_mask(make_state(PROMOTION))[4096+4*7]  # a7->h8 impossible geometry.
    assert torch.isfinite(output[:, 4096+4*7]).all()
    return {'checked': checked, 'ordinary_promotion_pair_and_impossible_slot_retained': True}


def backbone_diagnostic_case():
    # Direct module composition is a diagnostic only; no network/evaluator.
    import json
    from pathlib import Path
    from chess_rl.adapter import encode_position
    from .helpers import fingerprint
    records = json.loads((Path(__file__).parent/'fixtures/adapter_records.json').read_text())
    states = [make_state(r['start_fen'], r['moves_uci'], history_complete=r['history_complete']) for r in records]
    before = [fingerprint(s) for s in states]
    board_ids = [id(s.board) for s in states]
    features = np.stack([encode_position(s).features for s in states])
    x = torch.from_numpy(features).requires_grad_()
    torch.manual_seed(303)
    backbone = EncoderBackbone().eval()
    head = PairPolicyHead().eval()
    tokens = backbone(x)
    tokens.retain_grad()
    logits = head(tokens)
    assert logits.shape == (13, 4352) and torch.isfinite(logits).all()
    objective(logits).backward()
    for p in [x, tokens, *backbone.parameters(), *head.parameters()]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
    assert [fingerprint(s) for s in states] == before and [id(s.board) for s in states] == board_ids
    np.testing.assert_array_equal(np.stack([encode_position(s).features for s in states]), features)
    assert torch.equal(x.detach(), torch.from_numpy(features))
    assert sum(p.numel() for p in backbone.parameters()) == 3567168
    return {'fixture_count': 13, 'contextual_shape': list(tokens.shape), 'logit_shape': list(logits.shape),
            'backbone_parameter_elements': 3567168, 'head_parameter_elements': 66048,
            'finite_nonzero_gradients_through_head_and_backbone': True, 'states_and_features_unchanged': True,
            'terminal_fixtures_are_tensor_probes_only': True, 'assembled_network_added': False}
