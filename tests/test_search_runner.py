"""S04 budget, target, action, failure and exploration contract checks."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import (
    Evaluation, apply_move, encode_position, id_to_move, legal_mask, make_state,
    move_to_id, terminal_result,
)
from chess_rl.search import runner
from chess_rl.search.evaluator import evaluate
from chess_rl.search.puct import Node, SearchConfig, backup, expand, traverse
from chess_rl.search.runner import (
    SearchFailure, TrainingExploration, root_visits, search, select_move, training_policy,
)
from .helpers import BLACK_PROMOTION, CASTLING, CYCLE, LONG_CYCLE, PROMOTION, fingerprint
from .test_puct import Recorder, snapshot, stats


def assert_target(state, result, total):
    visits, policy = result.root_visits, result.training_policy
    assert visits.shape == policy.shape == (4352,)
    assert visits.dtype == np.int64 and policy.dtype == np.float64
    assert int(visits.sum()) == total
    assert np.isfinite(policy).all() and (policy >= 0).all()
    assert policy.sum() == pytest.approx(1, abs=1e-15)
    mask = legal_mask(state)
    assert not np.any(visits[~mask]) and not np.any(policy[~mask])
    np.testing.assert_array_equal(policy > 0, visits > 0)
    # Independent scalar count normalization, including every zero slot.
    for i, n in enumerate(visits):
        assert policy[i] == int(n) / total
    if result.move_id is not None:
        assert visits[result.move_id] > 0
        assert result.move in state.board.legal_moves
        assert id_to_move(state, result.move_id) == result.move
        assert move_to_id(state, result.move) == result.move_id


@pytest.mark.parametrize("budget", [1, 2, 3, 16, 32])
@pytest.mark.parametrize("black", [False, True])
def test_exact_completed_budget_and_pending_work(budget, black, monkeypatch):
    state = make_state(moves_uci=["h2h3"] if black else [])
    root = Node(state)
    before = snapshot(state), snapshot(root.state)
    rec = Recorder()
    completions, lengths = [], []
    def checked_backup(resolved):
        assert not resolved.completed and root.parent_visits == len(completions)
        count = backup(resolved)
        assert count == 1 and resolved.completed
        completions.append(count)
        lengths.append(len(resolved.path))
        assert root.parent_visits == len(completions)
        return count
    monkeypatch.setattr(runner, "backup", checked_backup)
    result = search(root, budget, evaluator=rec)
    assert result.requested_simulations == result.completed_simulations == budget
    assert len(completions) == budget and sum(lengths) >= budget
    assert len(rec.calls) == budget + 1  # Root preparation is extra, never counted.
    assert (snapshot(state), snapshot(root.state)) == before
    assert_target(state, result, budget)
    assert result.move_id == min(i for i, n in enumerate(result.root_visits) if n == max(result.root_visits))


def test_pending_paths_reuse_and_zero_additional_budget():
    root = Node(make_state())
    expand(root)
    pending = [traverse(root), traverse(root)]
    assert root.parent_visits == 0
    first = search(root, 3)
    assert first.completed_simulations == 3 and not any(p.completed for p in pending)
    for resolved in pending:
        assert backup(resolved) == 1
    second = search(root, 2)
    assert second.completed_simulations == 2
    assert_target(root.state, second, 7)
    before = stats(root)
    third = search(root, 0, evaluator=lambda _: pytest.fail("Unexpected evaluation"))
    assert third.completed_simulations == 0 and stats(root) == before
    assert_target(root.state, third, 7)
    assert first.root_visits.sum() == 3  # Independent result snapshots.


def test_zero_budget_prepares_without_fabricating_target_or_action():
    root = Node(make_state())
    rec = Recorder()
    rng = np.random.default_rng(7)
    before_rng = deepcopy(rng.bit_generator.state)
    result = search(root, 0, evaluator=rec, exploration=TrainingExploration(noise_fraction=.25), rng=rng)
    assert root.expanded and len(rec.calls) == 1
    assert result.completed_simulations == 0 and not result.root_visits.any()
    assert result.move_id is result.move is result.training_policy is result.terminal is None
    assert before_rng == rng.bit_generator.state
    assert all(edge.child is None for edge in root.edges.values())
    with pytest.raises(ValueError, match="positive"):
        training_policy(root)
    with pytest.raises(ValueError, match="positive"):
        select_move(root)


@pytest.mark.parametrize("bad", [-1, 1.5, True, np.int64(2), "2", None])
def test_invalid_budget_no_work(bad):
    root = Node(make_state())
    with pytest.raises(ValueError, match="budget"):
        search(root, bad, evaluator=lambda _: pytest.fail("Unexpected evaluation"))
    assert not root.expanded and not root.edges


@pytest.mark.parametrize("field,bad", [
    ("temperature", -1), ("temperature", np.nan), ("temperature", np.inf),
    ("temperature", True), ("temperature", "1"), ("noise_fraction", -.01),
    ("noise_fraction", 1.01), ("noise_fraction", np.nan), ("noise_fraction", False),
    ("dirichlet_alpha", 0), ("dirichlet_alpha", -1), ("dirichlet_alpha", np.inf),
    ("dirichlet_alpha", True), ("dirichlet_alpha", None),
])
def test_exploration_config_rejects_invalid(field, bad):
    with pytest.raises(ValueError, match=field):
        TrainingExploration(**{field: bad})


@pytest.mark.parametrize("exploration,rng", [
    (TrainingExploration(), None), (TrainingExploration(0, .25), None),
    (TrainingExploration(), 5), (None, np.random.RandomState(7)), ("training", None),
])
def test_explicit_rng_validation_precedes_mutation(exploration, rng):
    root = Node(make_state())
    with pytest.raises(ValueError):
        search(root, 1, exploration=exploration, rng=rng)
    assert not root.expanded


@pytest.mark.parametrize("budget", [0, 1, 32])
@pytest.mark.parametrize("state,reason", [
    (make_state(moves_uci="f2f3 e7e5 g2g4 d8h4".split()), "CHECKMATE"),
    (make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"), "STALEMATE"),
    (make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "INSUFFICIENT_MATERIAL"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 100 1"), "FIFTY_MOVES"),
    (make_state(moves_uci=LONG_CYCLE * 2), "THREEFOLD_REPETITION"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 b - - 150 1"), "SEVENTYFIVE_MOVES"),
    (make_state(moves_uci=CYCLE * 4), "FIVEFOLD_REPETITION"),
])
def test_terminal_roots_skip_everything(state, reason, budget):
    root = Node(state)
    before = snapshot(state), snapshot(root.state)
    rng = np.random.default_rng(3)
    before_rng = deepcopy(rng.bit_generator.state)
    result = search(root, budget, evaluator=lambda _: pytest.fail("Terminal evaluation"),
                    exploration=TrainingExploration(noise_fraction=.25), rng=rng)
    assert result.terminal is root.result and result.terminal.ending_reason == reason
    assert result.completed_simulations == 0 and not result.root_visits.any()
    assert result.training_policy is result.move_id is result.move is None
    assert training_policy(root) is select_move(root) is None
    assert (snapshot(state), snapshot(root.state)) == before
    assert rng.bit_generator.state == before_rng
    if reason in {"FIFTY_MOVES", "THREEFOLD_REPETITION", "FIVEFOLD_REPETITION", "SEVENTYFIVE_MOVES"}:
        assert legal_mask(state).any()


SPECIAL = [(CASTLING.format(turn), uci) for turn, ucis in [
    ("w", ["e1g1", "e1c1"]), ("b", ["e8g8", "e8c8"])] for uci in ucis]
SPECIAL += [(PROMOTION, f"e7{dest}{piece}") for dest in ("d8", "e8", "f8") for piece in "qrbn"]
SPECIAL += [(BLACK_PROMOTION, f"e2{dest}{piece}") for dest in ("d1", "e1", "f1") for piece in "qrbn"]
SPECIAL += [("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3")]


@pytest.mark.parametrize("fen,uci", SPECIAL)
def test_special_action_legality_roundtrip_full_history(fen, uci):
    state = make_state(fen)
    root = Node(state)
    before = snapshot(state), snapshot(root.state)
    result = search(root, 1, evaluator=Recorder([uci]))
    assert_target(state, result, 1)
    assert result.move.uci() == uci
    copied = state.copy()
    apply_move(copied, result.move_id)
    child = root.edges[result.move_id].child
    assert fingerprint(copied) == fingerprint(child.state)
    assert child.state.history_complete is False
    assert (snapshot(state), snapshot(root.state)) == before


@pytest.mark.parametrize("complete", [True, False])
def test_long_caller_history_and_every_retained_child_prefix(complete):
    line = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6".split()
    state = make_state(moves_uci=line, history_complete=complete)
    features = encode_position(state).features.copy()
    root = Node(state)
    before = snapshot(state), snapshot(root.state)
    result = search(root, 24, exploration=TrainingExploration(noise_fraction=.25), rng=np.random.default_rng(23))
    assert_target(state, result, 24)
    assert (snapshot(state), snapshot(root.state)) == before
    np.testing.assert_array_equal(encode_position(state).features, features)
    def inspect(node):
        assert node.state.initial_fen == state.initial_fen
        assert node.state.history_complete is complete
        assert node.state.board.move_stack[:len(line)] == state.board.move_stack
        for edge in node.edges.values():
            if edge.child is not None:
                replay = node.state.copy()
                apply_move(replay, edge.move_id)
                assert fingerprint(replay) == fingerprint(edge.child.state)
                inspect(edge.child)
    inspect(root)


def test_target_is_raw_visits_not_prior_or_temperature():
    root = Node(make_state())
    expand(root)
    ids = sorted(root.edges)
    for i, edge in enumerate(root.edges.values()):
        edge.prior = 1.0 if i == len(ids) - 1 else 0.0
    root.edges[ids[0]].visit_count = 1
    root.edges[ids[1]].visit_count = 4
    expected = np.zeros(4352)
    expected[ids[:2]] = [.2, .8]
    real_choose = runner._choose
    captured = []
    # Observe the distribution passed to a real Generator via a subclass.
    class CaptureRNG(np.random.Generator):
        def choice(self, a, **kwargs):
            captured.append((a.copy(), kwargs["p"].copy()))
            return super().choice(a, **kwargs)
    rng = CaptureRNG(np.random.PCG64(9))
    for temperature, expected_weights in [(1, [.2, .8]), (2, [1/3, 2/3]), (.5, [1/17, 16/17])]:
        result = search(root, 0, exploration=TrainingExploration(temperature), rng=rng)
        np.testing.assert_array_equal(result.training_policy, expected)
        np.testing.assert_array_equal(captured[-1][0], ids[:2])
        np.testing.assert_allclose(captured[-1][1], expected_weights, atol=1e-15)
        assert result.move_id in ids[:2]
    assert select_move(root) == ids[1]  # Highest prior is elsewhere.
    root.edges[ids[0]].visit_count = 4
    root.edges = dict(reversed(list(root.edges.items())))
    assert select_move(root) == ids[0]
    assert select_move(root, exploration=TrainingExploration(0)) == ids[0]
    for temperature in [1e-320, 1e308]:
        assert real_choose(root_visits(root), TrainingExploration(temperature), rng) in ids[:2]


@pytest.mark.parametrize("temperature,noise", [(1, 0), (0, .25), (1, .25), (2, 1)])
def test_seeded_search_reproducibility_and_evaluation_rng_isolation(temperature, noise):
    outputs = []
    for _ in range(2):
        root = Node(make_state(moves_uci=["e2e4"]))
        rng = np.random.default_rng(103)
        result = search(root, 16, exploration=TrainingExploration(temperature, noise), rng=rng)
        outputs.append((result, stats(root), deepcopy(rng.bit_generator.state)))
        assert_target(root.state, result, 16)
    a, b = outputs
    np.testing.assert_array_equal(a[0].root_visits, b[0].root_visits)
    np.testing.assert_array_equal(a[0].training_policy, b[0].training_policy)
    assert a[0].move == b[0].move and a[1:] == b[1:]
    rng = np.random.default_rng(24)
    initial_rng = deepcopy(rng.bit_generator.state)
    evaluation = search(Node(make_state()), 16, rng=rng)
    reference = search(Node(make_state()), 16)
    assert rng.bit_generator.state == initial_rng
    np.testing.assert_array_equal(evaluation.root_visits, reference.root_visits)
    assert evaluation.move == reference.move


def test_root_noise_independent_oracle_and_restoration(monkeypatch):
    root = Node(make_state())
    expand(root, Recorder(["e2e4"]))
    priors = [(root.edges[i], root.edges[i].prior) for i in sorted(root.edges)]
    original_array = root.result.priors.tobytes()
    noise = np.random.default_rng(42).dirichlet(np.full(len(priors), .4))
    expected = np.array([.75 * p + .25 * z for (_, p), z in zip(priors, noise)])
    original_traverse = runner.traverse
    seen = []
    def observe(root, config, evaluator):
        actual = np.array([edge.prior for edge, _ in priors])
        np.testing.assert_array_equal(actual, expected)
        assert root.result.priors.tobytes() == original_array
        assert actual.sum() == pytest.approx(1)
        resolved = original_traverse(root, config, evaluator)
        for step in resolved.path:
            if step.child.edges:
                assert all(e.prior == float(step.child.result.priors[i]) for i, e in step.child.edges.items())
        seen.append(resolved)
        return resolved
    monkeypatch.setattr(runner, "traverse", observe)
    result = search(root, 8, exploration=TrainingExploration(0, .25, .4), rng=np.random.default_rng(42))
    assert len(seen) == 8 and result.completed_simulations == 8
    assert [(edge.prior) for edge, _ in priors] == [p for _, p in priors]
    assert root.result.priors.tobytes() == original_array
    # Reused root: evaluation begins with original P, no residual noise.
    monkeypatch.setattr(runner, "traverse", original_traverse)
    control = Node(make_state())
    expand(control, Recorder(["e2e4"]))
    for move_id, edge in root.edges.items():
        control.edges[move_id].visit_count = edge.visit_count
        control.edges[move_id].value_sum = edge.value_sum
        control.edges[move_id].child = edge.child  # Read-only score oracle; no traversal.
    from chess_rl.search.puct import select_edge
    assert select_edge(root).move_id == select_edge(control).move_id
    search(root, 1)
    assert [edge.prior for edge, _ in priors] == [p for _, p in priors]


@pytest.mark.parametrize("prior_completed", [0, 3])
@pytest.mark.parametrize("fail_after", [0, 1, 4])
def test_leaf_failure_reports_exact_partial_preserves_earlier_work(prior_completed, fail_after):
    state = make_state(moves_uci=["e2e4"])
    root = Node(state)
    expand(root)
    if prior_completed:
        search(root, prior_completed)
    before = snapshot(state), snapshot(root.state)
    calls = 0
    failure = RuntimeError("Controlled leaf failure")
    def evaluator(states):
        nonlocal calls
        if calls == fail_after:
            raise failure
        calls += 1
        return evaluate(states)
    with pytest.raises(SearchFailure) as caught:
        search(root, fail_after + 2, evaluator=evaluator,
               exploration=TrainingExploration(0, .25), rng=np.random.default_rng(37))
    exc = caught.value
    assert exc.stage == "simulation" and exc.__cause__ is failure
    partial = exc.partial
    assert partial.completed_simulations == fail_after
    assert root.parent_visits == prior_completed + fail_after
    assert partial.move is partial.move_id is None
    if root.parent_visits:
        assert_target(state, partial, root.parent_visits)
    else:
        assert partial.training_policy is None
    assert (snapshot(state), snapshot(root.state)) == before
    assert all(e.prior == float(root.result.priors[i]) for i, e in root.edges.items())
    repaired = search(root, 2)
    assert repaired.completed_simulations == 2
    assert_target(state, repaired, prior_completed + fail_after + 2)


@pytest.mark.parametrize("bad_result", [None, [], "exception"])
def test_root_preparation_failure_counts_zero_and_retries(bad_result):
    root = Node(make_state())
    def broken(states):
        if bad_result == "exception":
            raise RuntimeError("Preparation failure")
        return bad_result
    with pytest.raises(SearchFailure) as caught:
        search(root, 2, evaluator=broken)
    assert caught.value.stage == "preparation"
    partial = caught.value.partial
    assert partial.completed_simulations == 0 and not partial.root_visits.any()
    assert partial.training_policy is partial.move is None
    assert not root.expanded
    assert search(root, 2).completed_simulations == 2


@pytest.mark.parametrize("interrupt", [False, True])
def test_actual_backup_publication_failure_preserves_completed_stats_and_noise(monkeypatch, interrupt):
    root = Node(make_state())
    search(root, 3)
    expected_priors = {i: e.prior for i, e in root.edges.items()}
    root_edge_ids = {id(e) for e in root.edges.values()}
    original_setattr = runner.backup.__globals__["Edge"].__setattr__
    count, failed = 0, False
    error_type = KeyboardInterrupt if interrupt else RuntimeError
    def fail_on_value_sum(edge, name, value):
        nonlocal count, failed
        if name == "value_sum" and id(edge) in root_edge_ids and not failed:
            count += 1
            if count == 2:
                failed = True
                original_setattr(edge, name, value)
                raise error_type("Injected S03 publication failure")
        original_setattr(edge, name, value)
    monkeypatch.setattr(runner.backup.__globals__["Edge"], "__setattr__", fail_on_value_sum)
    with pytest.raises(KeyboardInterrupt if interrupt else SearchFailure) as caught:
        search(root, 5, exploration=TrainingExploration(0, .25), rng=np.random.default_rng(6))
    assert root.parent_visits == 4  # One new completion, then atomic failure.
    assert {i: e.prior for i, e in root.edges.items()} == expected_priors
    if not interrupt:
        assert caught.value.partial.completed_simulations == 1
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert caught.value.partial.root_visits.sum() == 4
    assert search(root, 2).completed_simulations == 2
    assert root.parent_visits == 6


def test_selection_failure_counts_completed_budget_and_restores_noise(monkeypatch):
    root = Node(make_state())
    def bad_choice(*args):
        raise RuntimeError("Action failure")
    monkeypatch.setattr(runner, "_choose", bad_choice)
    with pytest.raises(SearchFailure) as caught:
        search(root, 3, exploration=TrainingExploration(noise_fraction=.25), rng=np.random.default_rng(6))
    assert caught.value.stage == "selection"
    assert caught.value.partial.completed_simulations == 3
    assert_target(root.state, caught.value.partial, 3)
    assert all(e.prior == float(root.result.priors[i]) for i, e in root.edges.items())


def test_noise_failure_zero_completions_no_publication():
    class BadNoise(np.random.Generator):
        def dirichlet(self, alpha):
            return np.full(len(alpha), np.nan)
    root = Node(make_state())
    with pytest.raises(SearchFailure) as caught:
        search(root, 3, exploration=TrainingExploration(0, .25), rng=BadNoise(np.random.PCG64(1)))
    assert caught.value.stage == "noise" and caught.value.partial.completed_simulations == 0
    assert root.parent_visits == 0
    assert all(e.prior == float(root.result.priors[i]) and e.child is None for i, e in root.edges.items())


def test_terminal_child_repeat_counts_without_leaf_evaluation():
    root = Node(make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"))
    recorder = Recorder(["g6g7"])
    result = search(root, 12, evaluator=recorder)
    assert len(recorder.calls) == 1 and result.completed_simulations == 12
    assert result.move.uci() == "g6g7"
    edge = root.edges[result.move_id]
    assert edge.visit_count == edge.value_sum == 12
    assert terminal_result(edge.child.state).ending_reason == "CHECKMATE"
    assert_target(root.state, result, 12)


def test_multi_ply_simulations_count_once_and_keep_parent_values():
    root = Node(make_state())
    rec = Recorder("e2e4 e7e5 g1f3 b8c6".split(), value=.6)
    expand(root, rec)
    # Pending expansion builds three plies without completing a simulation.
    for _ in range(2):
        traverse(root, evaluator=rec)
    result = search(root, 1, evaluator=rec)
    assert result.completed_simulations == root.parent_visits == 1
    edge = root.edges[result.move_id]
    assert edge.value_sum == pytest.approx(-.6)
    child_edge = next(e for e in edge.child.edges.values() if e.visit_count)
    assert child_edge.value_sum == pytest.approx(.6)
    leaf_edge = next(e for e in child_edge.child.edges.values() if e.visit_count)
    assert leaf_edge.value_sum == pytest.approx(-.6)


def test_result_snapshots_and_helper_validation():
    root = Node(make_state())
    with pytest.raises(ValueError, match="expanded"):
        root_visits(root)
    result = search(root, 2)
    expected = root_visits(root)
    result.root_visits[:] = 0
    result.training_policy[:] = 0
    np.testing.assert_array_equal(root_visits(root), expected)
    assert training_policy(root).sum() == pytest.approx(1)
    i = next(iter(root.edges))
    root.edges[i].visit_count = -1
    with pytest.raises(ValueError, match="counts"):
        root_visits(root)


def test_cpu_boundary_no_model_training_engine_or_encoding():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("torch.", "chess.engine", "chess_rl.model", "chess_rl.training", "chess_rl.data")):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, Block())
import chess_rl.adapter as adapter
def forbidden(*args):
    raise AssertionError("S04 encoded features")
adapter.encode_position = forbidden
from chess_rl.search.puct import Node
from chess_rl.search.runner import search
result = search(Node(adapter.make_state()), 3)
assert result.completed_simulations == 3 and result.root_visits.sum() == 3
print("CPU S04 boundary passed")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
