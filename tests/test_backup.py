"""S03 independent value/sign oracles and completion/failure checks."""
import copy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, TerminalResult, encode_position, make_state, move_to_id
from chess_rl.search.evaluator import evaluate
from chess_rl.search.puct import (
    Edge, Node, PathStep, SearchConfig, TraversalResult, backup, expand,
    puct_score, select_edge, traverse,
)
from .helpers import CASTLING, CYCLE, LONG_CYCLE, PROMOTION, BLACK_PROMOTION, fingerprint


WHITE_LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"]
BLACK_LINE = WHITE_LINE[1:] + ["g8f6"]
# Hand-written root-to-leaf signs, independent of the reverse-path algorithm.
SIGNS = {
    1: [-1], 2: [1, -1], 3: [-1, 1, -1],
    4: [1, -1, 1, -1], 7: [-1, 1, -1, 1, -1, 1, -1],
}


class Scripted:
    """Controlled nonzero evaluations; no production backup in the oracle."""
    def __init__(self, moves=(), value=0.6):
        self.moves = moves
        self.value = value
        self.calls = 0

    def __call__(self, states):
        state, = states
        ordinary, = evaluate(states)
        assert isinstance(ordinary, Evaluation)
        priors = ordinary.priors.copy()
        if self.calls < len(self.moves):
            move_id = move_to_id(state, chess.Move.from_uci(self.moves[self.calls]))
            priors[:] = 0
            priors[move_id] = 1
        self.calls += 1
        return [Evaluation(priors, [(1 + self.value) / 2, 0, (1 - self.value) / 2], self.value)]


def forbidden(*args):
    raise AssertionError("Evaluator invoked for an already resolved terminal child")


def controlled_path(state=None, depth=4, value=0.6, line=None):
    state = make_state() if state is None else state
    if line is None:
        line = WHITE_LINE if state.board.turn == chess.WHITE else BLACK_LINE
    root = Node(state)
    evaluator = Scripted(line[:depth], value)
    expand(root, evaluator)
    attempts = [traverse(root, evaluator=evaluator) for _ in range(depth)]
    assert [len(attempt.path) for attempt in attempts] == list(range(1, depth + 1))
    return root, evaluator, attempts


def tree_stats(resolved):
    nodes = [step.parent for step in resolved.path] + [resolved.leaf]
    return tuple(tuple((id(edge), edge.visit_count, edge.value_sum)
                       for edge in node.edges.values()) for node in nodes)


def state_snapshot(state):
    board = state.board
    return (id(board), id(board.move_stack), id(board._stack),
            tuple(id(move) for move in board.move_stack),
            tuple((id(item), dict(vars(item))) for item in board._stack),
            {key: value for key, value in vars(board).items() if key not in {"move_stack", "_stack"}},
            fingerprint(state), encode_position(state).features.tobytes())


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("depth", SIGNS)
@pytest.mark.parametrize("value", [0.6, -0.75])
def test_independent_nonzero_signs_both_colors_and_depths(color, depth, value):
    state = make_state(moves_uci=[] if color == chess.WHITE else ["e2e4"])
    root, evaluator, attempts = controlled_path(state, depth, value)
    resolved = attempts[-1]
    assert all(not attempt.completed for attempt in attempts)
    assert root.parent_visits == 0  # All earlier resolved work is still pending.
    # Nonzero preexisting statistics exercise accumulation, not assignment.
    for i, step in enumerate(resolved.path):
        step.edge.visit_count, step.edge.value_sum = i + 2, 0.25
    before = tree_stats(resolved)
    assert resolved.result.value == value
    assert backup(resolved) == 1 and resolved.completed
    assert evaluator.calls == depth + 1
    assert all(not attempt.completed for attempt in attempts[:-1])
    for i, (step, sign) in enumerate(zip(resolved.path, SIGNS[depth], strict=True)):
        assert step.edge.visit_count == i + 3
        assert step.edge.value_sum == pytest.approx(0.25 + sign * value, abs=1e-15)
        q = (0.25 + sign * value) / (i + 3)
        expected = q + 1.5 * step.edge.prior * np.sqrt(i + 3) / (i + 4)
        assert puct_score(step.edge, step.parent.parent_visits) == pytest.approx(expected)
        assert step.parent.parent_visits == i + 3
        for entry in before[i]:
            edge = next(edge for edge in step.parent.edges.values() if id(edge) == entry[0])
            if edge is not step.edge:
                assert (edge.visit_count, edge.value_sum) == entry[1:]
    assert resolved.leaf.parent_visits == 0


def test_each_success_counts_one_simulation_regardless_of_path_length():
    root, evaluator, attempts = controlled_path(depth=4)
    assert root.parent_visits == 0 and all(not attempt.completed for attempt in attempts)
    # These are four distinct pending simulations that share some edges.
    completions = [backup(attempt) for attempt in attempts]
    assert completions == [1, 1, 1, 1]
    assert root.parent_visits == sum(completions) == 4
    deepest = attempts[-1]
    assert [step.edge.visit_count for step in deepest.path] == [4, 3, 2, 1]
    assert [step.edge.value_sum for step in deepest.path] == pytest.approx([0, -0.6, 0, -0.6])
    assert [step.parent.parent_visits for step in deepest.path] == [4, 3, 2, 1]
    assert deepest.leaf.parent_visits == 0 and evaluator.calls == 5


@pytest.mark.parametrize("state,line,winner,reason", [
    (make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"), ["g6g7"], chess.WHITE, "CHECKMATE"),
    (make_state("8/8/8/8/8/5kq1/8/7K b - - 0 1"), ["g3g2"], chess.BLACK, "CHECKMATE"),
    (make_state(moves_uci=WHITE_LINE[:0]), ["f2f3", "e7e5", "g2g4", "d8h4"], chess.BLACK, "CHECKMATE"),
    (make_state(moves_uci=["f2f3", "e7e5"]), ["g2g4", "d8h4"], chess.BLACK, "CHECKMATE"),
    (make_state("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"), ["g5g6"], None, "STALEMATE"),
    (make_state("8/8/8/8/6q1/8/5k2/7K b - - 0 1"), ["g4g3"], None, "STALEMATE"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1"), ["a1a2"], None, "FIFTY_MOVES"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 b - - 99 1"), ["e8e7"], None, "FIFTY_MOVES"),
    (make_state(moves_uci=CYCLE + CYCLE[:3]), ["f6g8"], None, "THREEFOLD_REPETITION"),
    (make_state(moves_uci=CYCLE + CYCLE[:2]), CYCLE[2:], None, "THREEFOLD_REPETITION"),
    (make_state("4k3/8/8/8/8/8/3n4/2B1K3 w - - 0 1"), ["c1d2"], None, "INSUFFICIENT_MATERIAL"),
])
def test_exact_terminal_values_and_repeated_visits_without_evaluator(state, line, winner, reason):
    root, evaluator, attempts = controlled_path(state, len(line), line=line)
    resolved = attempts[-1]
    terminal = resolved.result
    assert isinstance(terminal, TerminalResult)
    assert terminal.winner is winner and terminal.ending_reason == reason
    assert terminal.value == (0 if winner is None else -1)
    np.testing.assert_array_equal(terminal.wdl, [0, 1, 0] if winner is None else [0, 0, 1])
    assert terminal.priors is None and terminal.history_complete == state.history_complete
    wdl = terminal.wdl.tobytes()
    calls = evaluator.calls
    completed = 0
    for i in range(5):
        # Keep the deliberately forced losing line selected during repeated
        # terminal probes; default PUCT would rightly explore other root moves.
        attempt = resolved if i == 0 else traverse(root, SearchConfig(c_puct=10), evaluator=forbidden)
        assert attempt is resolved or attempt._completion is not resolved._completion
        assert attempt.leaf is resolved.leaf and attempt.result is terminal
        assert len(attempt.path) == len(line)
        completed += backup(attempt)
        # Exact score oracle: each parent whose color eventually won gets +1.
        for step in attempt.path:
            expected = 0 if winner is None else (1 if step.parent.state.board.turn == winner else -1)
            assert step.edge.visit_count == i + 1
            assert step.edge.value_sum == (i + 1) * expected
        assert root.parent_visits == completed == i + 1
    assert evaluator.calls == calls and terminal.wdl.tobytes() == wdl
    assert resolved.leaf.parent_visits == 0 and resolved.leaf.edges == {}


@pytest.mark.parametrize("state", [
    make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 150 1"),
    make_state("8/8/8/8/8/5k2/6q1/7K w - - 0 1"),
    make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
    make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    make_state(moves_uci=LONG_CYCLE * 2),
    make_state("4k3/8/8/8/8/8/8/R3K3 b - - 100 1"),
])
def test_terminal_root_empty_paths_never_complete(state):
    root = Node(state)
    before = state_snapshot(state), state_snapshot(root.state)
    expand(root, forbidden)
    for _ in range(3):
        resolved = traverse(root, evaluator=forbidden)
        assert resolved.path == () and not resolved.completed
        assert backup(resolved) == backup(resolved) == 0
        assert not resolved.completed and root.parent_visits == 0
    assert (state_snapshot(state), state_snapshot(root.state)) == before


def test_duplicate_and_shallow_copy_share_completion_and_cannot_add_visits():
    root, _, attempts = controlled_path()
    resolved = attempts[-1]
    alias = copy.copy(resolved)
    assert alias._completion is resolved._completion
    assert backup(resolved) == 1 and alias.completed
    before = tree_stats(resolved)
    for duplicate in [resolved, alias, copy.copy(resolved)]:
        with pytest.raises(ValueError, match="already completed"):
            backup(duplicate)
        assert tree_stats(resolved) == before and root.parent_visits == 1


def test_constructed_or_replaced_records_are_not_new_completion_tickets():
    _, _, attempts = controlled_path()
    resolved = attempts[-1]
    for invalid in [None, resolved.leaf, TraversalResult(resolved.path, resolved.leaf, resolved.result),
                    replace(resolved), replace(resolved, path=()),
                    TraversalResult((PathStep(resolved.leaf, resolved.path[0].edge, resolved.leaf),),
                                    resolved.leaf, resolved.result)]:
        before = tree_stats(resolved)
        with pytest.raises(ValueError, match="issued by traverse"):
            backup(invalid)
        assert tree_stats(resolved) == before and not resolved.completed
    assert backup(resolved) == 1


@pytest.mark.parametrize("index", [0, 1, 3])
@pytest.mark.parametrize("field,bad", [
    ("visit_count", -1), ("visit_count", True), ("visit_count", 0.5),
    ("value_sum", 2.1), ("value_sum", float("nan")), ("value_sum", float("inf")),
    ("value_sum", "0"), ("value_sum", False),
])
def test_invalid_edge_statistics_fail_atomically_and_are_retryable(index, field, bad):
    root, _, attempts = controlled_path()
    resolved = attempts[-1]
    for step in resolved.path:
        step.edge.visit_count, step.edge.value_sum = 2, 0.25
    edge = resolved.path[index].edge
    original = getattr(edge, field)
    setattr(edge, field, bad)
    # NaN needs an identity/byte-safe snapshot for failure comparisons.
    before = repr(tree_stats(resolved))
    with pytest.raises(ValueError, match="statistics"):
        backup(resolved)
    assert repr(tree_stats(resolved)) == before and not resolved.completed
    setattr(edge, field, original)
    assert backup(resolved) == 1 and root.parent_visits == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.01, -1.01, True, "0", None])
def test_invalid_leaf_scalar_never_changes_statistics(bad):
    _, _, attempts = controlled_path()
    resolved = attempts[-1]
    original = resolved.result.value
    object.__setattr__(resolved.result, "value", bad)
    before = tree_stats(resolved)
    with pytest.raises(ValueError, match="Leaf value"):
        backup(resolved)
    assert tree_stats(resolved) == before and not resolved.completed
    object.__setattr__(resolved.result, "value", original)
    assert backup(resolved) == 1


@pytest.mark.parametrize("corruption", ["detached", "wrong-child", "unresolved", "wrong-result",
                                      "wrong-turn", "disconnected", "cycle", "ticket"])
def test_invalid_topology_or_resolution_is_atomic(corruption):
    root, _, attempts = controlled_path()
    resolved = attempts[-1]
    step = resolved.path[1]
    before = tree_stats(resolved)
    if corruption == "detached":
        root.edges[resolved.path[0].edge.move_id] = Edge(resolved.path[0].edge.move_id, 1)
    elif corruption == "wrong-child":
        step.edge.child = root
    elif corruption == "unresolved":
        step.child.result = None
    elif corruption == "wrong-result":
        resolved.leaf.result = evaluate([resolved.leaf.state])[0]
    elif corruption == "wrong-turn":
        step.child.state.board.turn = step.parent.state.board.turn
    elif corruption == "disconnected":
        object.__setattr__(step, "parent", root)
    elif corruption == "cycle":
        step.edge.child = root
        object.__setattr__(step, "child", root)
    else:
        object.__setattr__(resolved, "path", resolved.path[:-1])
    corrupted = tree_stats(resolved)
    with pytest.raises(ValueError):
        backup(resolved)
    assert tree_stats(resolved) == corrupted and not resolved.completed
    assert all(n == 0 and w == 0 for node in before for _, n, w in node)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failure_site", ["edge", "completion"])
def test_publication_exception_rolls_back_even_after_partial_writes(monkeypatch, error, failure_site):
    root, _, attempts = controlled_path()
    resolved = attempts[-1]
    for i, step in enumerate(resolved.path):
        step.edge.visit_count, step.edge.value_sum = i + 2, 0.25
    before = tree_stats(resolved)
    target = resolved.path[0].edge if failure_site == "edge" else resolved._completion
    field = "value_sum" if failure_site == "edge" else "completed"
    def interrupted_setattr(self, name, value):
        if self is target and name == field:
            raise error("controlled publication interruption")
        object.__setattr__(self, name, value)
    with monkeypatch.context() as patch:
        patch.setattr(type(target), "__setattr__", interrupted_setattr, raising=False)
        with pytest.raises(error, match="publication interruption"):
            backup(resolved)
    assert tree_stats(resolved) == before and not resolved.completed
    assert backup(resolved) == 1 and root.parent_visits == 3


def test_failed_traversal_pending_work_and_root_preparation_have_zero_visits():
    root = Node(make_state())
    evaluator = Scripted(WHITE_LINE)
    expand(root, evaluator)
    assert root.parent_visits == 0
    def interrupted(states):
        raise RuntimeError("controlled evaluation failure")
    with pytest.raises(RuntimeError, match="evaluation failure"):
        traverse(root, evaluator=interrupted)
    edge = select_edge(root)
    child = edge.child
    assert child is not None and not child.expanded
    assert edge.visit_count == edge.value_sum == root.parent_visits == 0
    resolved = traverse(root, evaluator=evaluator)
    assert resolved.leaf is child and not resolved.completed and root.parent_visits == 0
    assert backup(resolved) == 1 and root.parent_visits == 1


def test_parent_visit_sums_include_siblings_and_zero_prior_edges():
    root, _, attempts = controlled_path(depth=2, value=0.5)
    resolved = attempts[-1]
    parent = root
    path_edge = resolved.path[0].edge
    siblings = [edge for edge in parent.edges.values() if edge is not path_edge]
    siblings[0].visit_count, siblings[0].value_sum = 9, -4.5
    siblings[1].visit_count, siblings[1].value_sum = 3, 0.3
    path_edge.visit_count, path_edge.value_sum = 4, 2
    assert parent.parent_visits == 16 and siblings[0].prior == siblings[1].prior == 0
    assert backup(resolved) == 1
    assert parent.parent_visits == 17
    assert path_edge.value_sum / path_edge.visit_count == 0.5
    assert puct_score(path_edge, parent.parent_visits) == pytest.approx(0.5 + 1.5 * np.sqrt(17) / 6)
    assert (siblings[0].visit_count, siblings[0].value_sum) == (9, -4.5)
    assert (siblings[1].visit_count, siblings[1].value_sum) == (3, 0.3)
    # Accepted defaults, exact ties and smallest-ID behavior are unchanged.
    assert SearchConfig() == SearchConfig(1.5, 0)
    a, b = siblings[:2]
    a.visit_count = b.visit_count = 2
    a.value_sum = b.value_sum = 1
    parent.edges = {b.move_id: b, a.move_id: a}
    assert select_edge(parent).move_id == min(a.move_id, b.move_id)


@pytest.mark.parametrize("state", [
    make_state(moves_uci=LONG_CYCLE + ["d2d4", "d7d5"]),
    make_state(chess.STARTING_FEN, LONG_CYCLE + ["d2d4", "d7d5"], history_complete=False),
    make_state(chess.STARTING_FEN, ["e2e4"], history_complete=False),
])
def test_backup_preserves_every_state_prefix_features_and_record_identity(state):
    root, evaluator, attempts = controlled_path(state, depth=4)
    resolved = attempts[-1]
    states = [state] + [step.parent.state for step in resolved.path] + [resolved.leaf.state]
    before = [state_snapshot(item) for item in states]
    nodes = [step.parent for step in resolved.path] + [resolved.leaf]
    records = [(id(node.edges), id(node.result), node.result.priors.tobytes(), node.result.wdl.tobytes())
               for node in nodes]
    path_refs = [(id(step), id(step.parent), id(step.edge), id(step.child), step.edge.prior)
                 for step in resolved.path]
    assert backup(resolved) == 1
    with pytest.raises(ValueError, match="already completed"):
        backup(resolved)
    assert [state_snapshot(item) for item in states] == before
    assert [(id(node.edges), id(node.result), node.result.priors.tobytes(), node.result.wdl.tobytes())
            for node in nodes] == records
    assert [(id(step), id(step.parent), id(step.edge), id(step.child), step.edge.prior)
            for step in resolved.path] == path_refs
    assert evaluator.calls == 5 and root.parent_visits == 1


def special_moves():
    cases = [(make_state(CASTLING.format(color)), f"e{rank}{dest}{rank}")
             for color, rank in [("w", "1"), ("b", "8")] for dest in "cg"]
    cases += [(make_state(moves_uci=["e2e4", "a7a6", "e4e5", "d7d5"]), "e5d6"),
              (make_state(moves_uci=["a2a3", "e7e5", "d2d4", "e5e4", "f2f4"]), "e4f3")]
    cases += [(make_state(fen), origin + dest + rank + piece)
              for fen, origin, rank in [(PROMOTION, "e7", "8"), (BLACK_PROMOTION, "e2", "1")]
              for dest in "def" for piece in "qrbn"]
    return cases


@pytest.mark.parametrize("state,uci", special_moves())
def test_special_moves_keep_histories_through_backup(state, uci):
    root, _, attempts = controlled_path(state, depth=1, value=0.375, line=[uci])
    resolved = attempts[0]
    states = [state, root.state, resolved.leaf.state]
    before = [state_snapshot(item) for item in states]
    reference = state.board.copy(stack=True)
    reference.push_uci(uci)
    assert resolved.leaf.state.board.fen(en_passant="fen") == reference.fen(en_passant="fen")
    assert backup(resolved) == 1
    expected = -resolved.result.value
    assert resolved.path[0].edge.value_sum == expected
    assert root.parent_visits == 1 and resolved.leaf.parent_visits == 0
    assert [state_snapshot(item) for item in states] == before


def test_cpu_backup_import_boundary_and_no_evaluation_or_encoding():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("torch.", "chess.engine", "chess_rl.model", "chess_rl.training", "chess_rl.data")):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, Block())
import chess_rl.adapter as adapter
from chess_rl.search import puct
root = puct.Node(adapter.make_state())
puct.expand(root)
resolved = puct.traverse(root)
def forbidden(*args):
    raise AssertionError("Backup invoked adapter/evaluator work")
adapter.encode_position = puct.evaluate = puct.expand = puct.apply_move = puct.legal_mask = puct.terminal_result = forbidden
assert puct.backup(resolved) == 1 and root.parent_visits == 1
assert resolved.completed
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
