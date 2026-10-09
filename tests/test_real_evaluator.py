"""T04 boundaries, independent conversion, real batching and bounded search."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.adapter import Evaluation, TerminalResult, encode_position, make_state, move_to_id
from chess_rl.search import real_evaluator as implementation
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.puct import Node
from chess_rl.search.runner import search
from chess_rl.training.checkpoints import CheckpointError, Progress, save_checkpoint
from chess_rl.training.trainer import LossWeights
from .test_checkpoints import saved, rng_snapshot, assert_rng_equal
from .test_evaluator import nonterminal_cases, terminal_cases, preserved

# Frozen before running checks: public float32 adapter outputs on pinned CPU.
BATCH_ATOL = 2e-6
BATCH_RTOL = 2e-5
ORACLE_ATOL = 1e-6
ORACLE_RTOL = 1e-5


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def real(saved):
    return RealEvaluator(saved["path"], expected_config=saved["config"])


def support(state, result):
    assert type(result) is Evaluation
    ids = [move_to_id(state, move) for move in state.board.generate_legal_moves()]
    assert ids and len(set(ids)) == len(ids)
    illegal = np.ones(4352, dtype=bool)
    illegal[ids] = False
    assert not result.priors[illegal].any()
    for probabilities in (result.priors, result.wdl):
        assert probabilities.dtype == np.float32 and probabilities.flags.owndata
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert abs(probabilities.sum(dtype=np.float64) - 1) < 1e-6
    assert result.value == float(result.wdl[0] - result.wdl[2])


def softmax(values):
    # Independent float64 NumPy oracle, with no production Torch normalizer.
    weights = np.exp(values.astype(np.float64) - np.max(values))
    return weights / weights.sum()


def test_shared_loader_identity_eval_and_rng(saved):
    before = rng_snapshot()
    evaluator = RealEvaluator(saved["path"], expected_config=saved["config"])
    checkpoint = evaluator.checkpoint
    assert checkpoint.checkpoint_id == "test-step-1"
    assert checkpoint.config == saved["config"]
    assert checkpoint.provenance == saved["kwargs"]["provenance"]
    assert all(not module.training for module in checkpoint.model.modules())
    bank = checkpoint.model.backbone.bank
    assert all(layer.self_attn.gab.bank is bank and layer.self_attn.relative is not None
               for layer in checkpoint.model.backbone.layers)
    evaluator([make_state()])
    assert_rng_equal(before, rng_snapshot())
    with pytest.raises(CheckpointError, match="incompatible"):
        RealEvaluator(saved["path"], expected_config=replace(saved["config"], dropout=0))
    assert_rng_equal(before, rng_snapshot())


def test_empty_and_all_terminal_bypass_every_nonterminal_operation(real, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Empty/terminal input reached nonterminal work")
    monkeypatch.setattr(implementation, "encode_position", forbidden)
    monkeypatch.setattr(implementation, "legal_mask", forbidden)
    monkeypatch.setattr(implementation, "legal_policy_probs", forbidden)
    monkeypatch.setattr(real.checkpoint.model, "forward", forbidden)
    monkeypatch.setattr(real.checkpoint.model, "eval", forbidden)
    cases = terminal_cases()
    states = [case[0] for case in cases]
    before = [preserved(state) for state in states]
    assert real.evaluate([]) == real(()) == []
    for (state, reason, winner), result in zip(cases, real(states)):
        assert type(result) is TerminalResult and result.priors is None
        assert result.ending_reason == reason and result.winner is winner
        assert result.history_complete == state.history_complete
        np.testing.assert_array_equal(result.wdl, [0, 1, 0] if winner is None else [0, 0, 1])
        assert result.value == (0.0 if winner is None else -1.0)
        if reason in {"THREEFOLD_REPETITION", "FIFTY_MOVES"}:
            assert list(state.board.legal_moves)
    assert [preserved(state) for state in states] == before


@pytest.mark.parametrize("index", range(13))
def test_batch_one_independent_logits_conversion_and_history(real, index):
    state = nonterminal_cases()[index]
    before = preserved(state)
    features = encode_position(state).features
    with torch.inference_mode():
        logits, wdl_logits = real.checkpoint.model(torch.from_numpy(features[None]))
    result, = real([state])
    support(state, result)
    ids = [move_to_id(state, move) for move in state.board.generate_legal_moves()]
    expected = np.zeros(4352)
    expected[ids] = softmax(logits[0].numpy()[ids])
    expected_wdl = softmax(wdl_logits[0].numpy())
    np.testing.assert_allclose(result.priors, expected, atol=ORACLE_ATOL, rtol=ORACLE_RTOL)
    np.testing.assert_allclose(result.wdl, expected_wdl, atol=ORACLE_ATOL, rtol=ORACLE_RTOL)
    assert abs(result.value - (expected_wdl[0] - expected_wdl[2])) <= ORACLE_ATOL
    assert preserved(state) == before
    np.testing.assert_array_equal(encode_position(state).features, features)


def test_mixed_partition_one_forward_order_batch_agreement(real, monkeypatch):
    nonterminal = nonterminal_cases()
    terminals = [case[0] for case in terminal_cases()]
    states = [terminals[4], nonterminal[0], terminals[6], nonterminal[1], terminals[0]]
    states += nonterminal + terminals + [nonterminal[0]]
    before = [preserved(state) for state in states]
    adjudicated, encoded, forwards = [], [], []
    original_terminal = implementation.terminal_result
    original_encode = implementation.encode_position
    original_forward = real.checkpoint.model.forward
    expected_terminals = [original_terminal(state) for state in states]
    pending = [state for state, result in zip(states, expected_terminals) if result is None]
    def terminal(state):
        adjudicated.append(state)
        return expected_terminals[len(adjudicated) - 1]
    def encode(state):
        assert len(adjudicated) == len(states)
        assert any(state is candidate for candidate in pending)
        encoded.append(state)
        return original_encode(state)
    def forward(features):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert all(not module.training for module in real.checkpoint.model.modules())
        np.testing.assert_array_equal(features.numpy(), np.stack([
            original_encode(state).features for state in pending]))
        forwards.append(tuple(features.shape))
        return original_forward(features)
    # Even accidental external mode changes are corrected before forward.
    real.checkpoint.model.train()
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "terminal_result", terminal)
        patch.setattr(implementation, "encode_position", encode)
        patch.setattr(real.checkpoint.model, "forward", forward)
        results = real(states)
    assert [id(state) for state in adjudicated] == [id(state) for state in states]
    assert [id(state) for state in encoded] == [id(state) for state in pending]
    assert forwards == [(len(pending), 64, 128)]
    for state, result, terminal_result in zip(states, results, expected_terminals):
        if terminal_result is not None:
            assert result is terminal_result
        else:
            support(state, result)
            individual, = real([state])
            for name in ("priors", "wdl", "value"):
                np.testing.assert_allclose(getattr(result, name), getattr(individual, name),
                                           atol=BATCH_ATOL, rtol=BATCH_RTOL)
    assert [preserved(state) for state in states] == before


def test_known_asymmetric_wdl_and_policy_use_root_player_order(real, monkeypatch):
    states = [make_state(), make_state(moves_uci=["e2e4"])]
    policy = torch.linspace(-9, 9, 4352).repeat(2, 1)
    wdl_logits = torch.tensor([[2., -1., .5], [-2., 0., 3.]])
    monkeypatch.setattr(real.checkpoint.model, "forward", lambda features: (policy, wdl_logits))
    for state, result, raw_wdl in zip(states, real(states), wdl_logits.numpy()):
        support(state, result)
        expected = softmax(raw_wdl)
        np.testing.assert_allclose(result.wdl, expected, atol=1e-7, rtol=1e-6)
        assert abs(result.value - (expected[0] - expected[2])) < 1e-7
    assert real(states)[0].value > 0 and real(states)[1].value < 0


def test_output_arrays_are_independent(real):
    state = make_state()
    a, b = real([state, state])
    reference, = real([state])
    assert not np.shares_memory(a.priors, b.priors) and not np.shares_memory(a.wdl, b.wdl)
    before_priors, before_wdl = b.priors.copy(), b.wdl.copy()
    a.priors[:] = 0
    a.wdl[:] = 0
    np.testing.assert_array_equal(b.priors, before_priors)
    np.testing.assert_array_equal(b.wdl, before_wdl)
    np.testing.assert_allclose(b.priors, reference.priors, atol=BATCH_ATOL, rtol=BATCH_RTOL)
    np.testing.assert_allclose(b.wdl, reference.wdl, atol=BATCH_ATOL, rtol=BATCH_RTOL)


@pytest.mark.parametrize("bad", ["policy_nan", "wdl_inf", "policy_width", "wdl_width", "wdl_batch"])
def test_invalid_outputs_fail_clearly(real, monkeypatch, bad):
    policy, wdl = torch.zeros(1, 4352), torch.zeros(1, 3)
    if bad == "policy_nan": policy[0, 0] = torch.nan
    elif bad == "wdl_inf": wdl[0, 0] = torch.inf
    elif bad == "policy_width": policy = policy[:, :-1]
    elif bad == "wdl_width": wdl = wdl[:, :-1]
    else: wdl = wdl.repeat(2, 1)
    monkeypatch.setattr(real.checkpoint.model, "forward", lambda features: (policy, wdl))
    state = make_state()
    before = preserved(state)
    with pytest.raises(ValueError): real([state])
    assert preserved(state) == before


def test_invalid_legal_support_fails_before_forward(real, monkeypatch):
    monkeypatch.setattr(implementation, "legal_mask", lambda state: np.zeros(4352, dtype=bool))
    monkeypatch.setattr(real.checkpoint.model, "forward", lambda _: pytest.fail("Unexpected forward"))
    with pytest.raises(ValueError, match="nonempty legal support"):
        real([make_state()])


def test_float64_checkpoint_public_float32_boundary(saved, tmp_path):
    config = replace(saved["config"], dtype="float64")
    model = config.build()
    path = tmp_path / "double.pt"
    save_checkpoint(path, model=model, optimizer=torch.optim.Adam(model.parameters()),
        checkpoint_id="double-fixture", provenance={"purpose": "T04 correctness"},
        weights=LossWeights(), progress=Progress(0, 0, 0),
        data_order=replace(saved["kwargs"]["data_order"], cursor=0))
    evaluator = RealEvaluator(path, expected_config=config)
    states = [make_state(), make_state(moves_uci=["e2e4"])]
    for state, batched in zip(states, evaluator(states)):
        support(state, batched)
        individual, = evaluator([state])
        np.testing.assert_allclose(batched.wdl, individual.wdl, atol=BATCH_ATOL, rtol=BATCH_RTOL)
        np.testing.assert_allclose(batched.priors, individual.priors, atol=BATCH_ATOL, rtol=BATCH_RTOL)


@pytest.mark.parametrize("index", [0, 1, 2, 4, 6, 7, 11, 12])
def test_bounded_search_compatibility(real, index):
    state = nonterminal_cases()[index]
    before = preserved(state)
    root = Node(state)
    result = search(root, 3, evaluator=real)
    assert result.completed_simulations == 3 and result.root_visits.sum() == 3
    assert result.move in state.board.legal_moves
    support(state, root.result)
    ids = {move_to_id(state, move) for move in state.board.legal_moves}
    assert not any(result.root_visits[i] for i in range(4352) if i not in ids)
    assert result.training_policy.sum() == pytest.approx(1)
    assert preserved(state) == before


def test_bounded_terminal_search_does_not_infer(real, monkeypatch):
    monkeypatch.setattr(real.checkpoint.model, "forward", lambda _: pytest.fail("Terminal forward"))
    for state, reason, _ in terminal_cases():
        before = preserved(state)
        result = search(Node(state), 3, evaluator=real)
        assert result.completed_simulations == 0 and result.terminal.ending_reason == reason
        assert result.move is None and preserved(state) == before
