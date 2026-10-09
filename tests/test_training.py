"""Independent numerical loss oracles and bounded combined-model updates."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.adapter import encode_position, legal_mask, make_state, move_to_id
from chess_rl.model import ChessNetwork
from chess_rl.supervised import SupervisedBatch
from chess_rl.tensor_ops import legal_policy_log_probs, legal_policy_probs
from chess_rl.training.losses import policy_class_cross_entropy, soft_wdl_cross_entropy
from chess_rl.training.trainer import LossWeights, supervised_step
import chess
from .fixtures.loss_reference import independent


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("scale", [1, 10000, 1e30])
def test_policy_matches_independent_loss_probabilities_and_gradient(dtype, batch, scale):
    # Noncontiguous logits and asymmetric support with illegal very large values.
    raw = torch.full((batch, 8704), 9e30, dtype=dtype)
    logits = raw[:, ::2].detach().requires_grad_()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    classes = torch.tensor([31 + row for row in range(batch)])
    with torch.no_grad():
        for row in range(batch):
            ids = [0, 31 + row, 4211, 4351]
            mask[row, ids] = True
            logits[row, ids] = torch.tensor([2.1, -.7, .3, -1.2], dtype=dtype) * scale
    before = logits.detach().clone()
    expected, gradient, probabilities = independent(logits.detach().numpy(), mask.numpy(), classes.numpy())
    loss = policy_class_cross_entropy(logits, mask, classes)
    loss.backward()
    np.testing.assert_allclose(loss.detach().numpy(), expected, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(logits.grad.numpy(), gradient, rtol=3e-6, atol=2e-7)
    priors = legal_policy_probs(logits, mask)
    np.testing.assert_allclose(priors.detach().numpy(), probabilities, rtol=3e-6, atol=2e-7)
    assert torch.equal(logits.detach(), before)
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(priors[~mask]) == 0
    assert torch.isneginf(legal_policy_log_probs(logits, mask)[~mask]).all()
    assert torch.allclose(priors.sum(dim=1), torch.ones(batch, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1, 10000, 1e30])
@pytest.mark.parametrize("targets", [[[.2, .3, .5], [0, .125, .875]], [[1, 0, 0], [0, 1, 0]]])
def test_wdl_matches_independent_soft_loss_and_gradient(dtype, scale, targets):
    logits = (torch.tensor([[1.2, -.7, .3], [-1, 2, .1]], dtype=dtype) * scale).requires_grad_()
    target = torch.tensor(targets, dtype=dtype)
    before = target.clone()
    expected, gradient, _ = independent(logits.detach().numpy(), np.ones((2, 3), dtype=bool), targets=target.numpy())
    loss = soft_wdl_cross_entropy(logits, target)
    loss.backward()
    np.testing.assert_allclose(loss.detach().numpy(), expected, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(logits.grad.numpy(), gradient, rtol=3e-6, atol=2e-7)
    assert torch.equal(target, before)
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_zero_wdl_mass_never_multiplies_negative_infinity():
    # Finite raw float32 logits whose subtraction exceeds the dtype range.
    logits = torch.tensor([[3e38, -3e38, -3e38]], requires_grad=True)
    target = torch.tensor([[1., 0., 0.]])
    loss = soft_wdl_cross_entropy(logits, target)
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError, match="finite"):
        soft_wdl_cross_entropy(logits, torch.tensor([[.5, .5, 0.]]))


def policy_inputs():
    return torch.zeros((2, 4352), requires_grad=True), torch.ones((2, 4352), dtype=torch.bool), torch.tensor([0, 1])


@pytest.mark.parametrize("bad", ["float", "bool", "rank", "batch", "negative", "large", "illegal"])
def test_invalid_policy_classes_fail(bad):
    logits, mask, classes = policy_inputs()
    if bad == "float": classes = classes.float()
    if bad == "bool": classes = classes.bool()
    if bad == "rank": classes = classes[:, None]
    if bad == "batch": classes = classes[:1]
    if bad == "negative": classes[0] = -1
    if bad == "large": classes[0] = 4352
    if bad == "illegal": mask[0, 0] = False
    with pytest.raises(ValueError): policy_class_cross_entropy(logits, mask, classes)
    assert logits.grad is None


@pytest.mark.parametrize("bad", ["empty_support", "empty_batch", "float_mask", "mask_rank", "int_logits", "rank", "width"])
def test_invalid_policy_support_or_shapes_fail(bad):
    logits, mask, classes = policy_inputs()
    if bad == "empty_support": mask[1] = False
    if bad == "empty_batch": logits, mask, classes = logits[:0], mask[:0], classes[:0]
    if bad == "float_mask": mask = mask.float()
    if bad == "mask_rank": mask = mask[0]
    if bad == "int_logits": logits = logits.detach().long()
    if bad == "rank": logits = logits[0]
    if bad == "width": logits, mask = logits[:, :4], mask[:, :4]
    with pytest.raises(ValueError): policy_class_cross_entropy(logits, mask, classes)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("legal", [True, False])
def test_nonfinite_policy_logits_fail_even_when_illegal(value, legal):
    logits, mask, classes = policy_inputs()
    with torch.no_grad(): logits[0, 10] = value
    mask[0, 10] = legal
    with pytest.raises(ValueError, match="finite"): legal_policy_probs(logits, mask)
    with pytest.raises(ValueError, match="finite"): policy_class_cross_entropy(logits, mask, classes)


def test_single_legal_class_is_zero_loss_and_gradient():
    logits, mask, classes = policy_inputs()
    mask[:] = False
    mask[torch.arange(2), classes] = True
    loss = policy_class_cross_entropy(logits, mask, classes)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_policy_range_overflow_fails_explicitly():
    logits, mask, classes = policy_inputs()
    with torch.no_grad(): logits[0, :2] = torch.tensor([3e38, -3e38])
    with pytest.raises(ValueError, match="nonfinite"): policy_class_cross_entropy(logits, mask, classes)


@pytest.mark.parametrize("bad", ["negative", "over_one", "zero_sum", "wrong_sum", "nan", "inf", "int", "rank", "width", "batch", "grad", "empty"])
def test_invalid_wdl_targets_fail(bad):
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([[.2, .3, .5], [0., .25, .75]])
    if bad == "negative": targets[0] = torch.tensor([-.1, .3, .8])
    if bad == "over_one": targets[0] = torch.tensor([1.1, 0, 0])
    if bad == "zero_sum": targets[0] = 0
    if bad == "wrong_sum": targets[0, 0] = .21
    if bad == "nan": targets[0, 0] = torch.nan
    if bad == "inf": targets[0, 0] = torch.inf
    if bad == "int": targets = targets.long()
    if bad == "rank": targets = targets[0]
    if bad == "width": targets = targets[:, :2]
    if bad == "batch": targets = targets[:1]
    if bad == "grad": targets.requires_grad_()
    if bad == "empty": logits, targets = logits[:0].detach().requires_grad_(), targets[:0]
    with pytest.raises(ValueError): soft_wdl_cross_entropy(logits, targets)
    assert logits.grad is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mass", [0., 1.])
def test_nonfinite_wdl_logits_fail_even_with_zero_mass(value, mass):
    logits = torch.zeros((1, 3))
    logits[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        soft_wdl_cross_entropy(logits, torch.tensor([[mass, 1 - mass, 0.]]))


def owned_batch():
    states = [make_state(), make_state(moves_uci=["e2e4"])]
    return SupervisedBatch((("controlled-white", 0), ("controlled-black", 1)), "train",
        np.stack([encode_position(s).features for s in states]), np.stack([legal_mask(s) for s in states]),
        np.array([move_to_id(states[0], chess.Move.from_uci("d2d4")),
                  move_to_id(states[1], chess.Move.from_uci("e7e5"))], dtype=np.int64),
        np.array([[.2, .3, .5], [.125, 0, .875]], dtype=np.float32))


def test_combined_step_analytic_sgd_update_and_batch_ownership():
    torch.manual_seed(20261009)
    model = ChessNetwork()  # all terms, default .1 dropout
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    batch = owned_batch()
    arrays = {name: getattr(batch, name).copy() for name in ("features", "legal_mask", "policy_classes", "wdl")}
    before = [p.detach().clone() for p in model.parameters()]
    captured = []
    def capture(module, args, output):
        for tensor in output: tensor.retain_grad()
        captured.extend(output)
    hook = model.register_forward_hook(capture)
    weights = LossWeights(.7, 1.3)
    metrics = supervised_step(model, optimizer, batch, weights=weights)
    hook.remove()
    assert metrics.batch_size == 2 and model.training
    expected_policy, _, _ = independent(captured[0].detach().numpy(), batch.legal_mask, batch.policy_classes)
    expected_wdl, _, _ = independent(captured[1].detach().numpy(), np.ones((2, 3), bool), targets=batch.wdl)
    assert metrics.policy_loss == pytest.approx(expected_policy, rel=2e-6)
    assert metrics.wdl_loss == pytest.approx(expected_wdl, rel=2e-6)
    assert metrics.total_loss == pytest.approx(.7 * expected_policy + 1.3 * expected_wdl, rel=2e-6)
    assert torch.count_nonzero(captured[0].grad[~torch.tensor(batch.legal_mask)]) == 0
    changed = 0
    for old, p in zip(before, model.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p, old - .001 * p.grad, atol=1e-7, rtol=1e-6)
        changed += not torch.equal(old, p)
    assert changed > 200
    for name, p in model.named_parameters():
        if ".relative." in name or name == "backbone.bank.weight":
            assert torch.count_nonzero(p.grad) > 0
    bank = model.backbone.bank.weight
    assert all(layer.self_attn.gab.bank is model.backbone.bank for layer in model.backbone.layers)
    assert sum(p is bank for p in optimizer.param_groups[0]["params"]) == 1
    for name, old in arrays.items(): np.testing.assert_array_equal(getattr(batch, name), old)


@pytest.mark.parametrize("bad", ["heldout", "empty", "features_nan", "illegal", "empty_support", "float_class", "hard_wdl", "wdl_nan", "feature_dtype"])
def test_bad_batch_never_updates_model(bad):
    model = ChessNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = owned_batch()
    if bad == "heldout": batch = replace(batch, split="validation")
    if bad == "empty": batch = replace(batch, references=())
    if bad == "features_nan": batch.features[0, 0, 0] = np.nan
    if bad == "illegal": batch.policy_classes[0] = 0
    if bad == "empty_support": batch.legal_mask[0] = False
    if bad == "float_class": batch = replace(batch, policy_classes=batch.policy_classes.astype(np.float32))
    if bad == "hard_wdl": batch = replace(batch, wdl=np.array([2, 0], dtype=np.int64))
    if bad == "wdl_nan": batch.wdl[0, 0] = np.nan
    if bad == "feature_dtype": batch = replace(batch, features=batch.features.astype(np.float64))
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError): supervised_step(model, optimizer, batch)
    assert not optimizer.state
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))


@pytest.mark.parametrize("failure", ["gradient", "output", "parameter", "gab_only", "foreign_optimizer"])
def test_model_failures_never_call_optimizer_step(failure):
    model = ChessNetwork(relative_enabled=failure != "gab_only")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    if failure == "foreign_optimizer": optimizer = torch.optim.SGD(ChessNetwork().parameters(), lr=.001)
    if failure == "gradient": model.backbone.input_projection.weight.register_hook(lambda g: g * torch.nan)
    if failure == "output":
        model.register_forward_hook(lambda module, args, outputs: (outputs[0] * torch.nan, outputs[1]))
    if failure == "parameter":
        with torch.no_grad(): model.backbone.square_embeddings[0, 0] = torch.inf
    calls = []
    optimizer.register_step_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError): supervised_step(model, optimizer, owned_batch())
    assert calls == []
    if failure in ("gradient", "output"): assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "1"])
@pytest.mark.parametrize("field", ["policy", "wdl"])
def test_invalid_weights(value, field):
    with pytest.raises(ValueError): LossWeights(**{field: value})
