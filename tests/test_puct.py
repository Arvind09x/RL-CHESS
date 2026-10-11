"""S02 controlled statistics and real legal/history traversal checks."""
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, TerminalResult, encode_position, make_state, move_to_id, terminal_result
from chess_rl.search import puct
from chess_rl.search.evaluator import evaluate
from chess_rl.search.puct import Edge, Node, SearchConfig, expand, puct_score, select_edge, traverse
from .helpers import BLACK_PROMOTION, CASTLING, CYCLE, LONG_CYCLE, PROMOTION, fingerprint


def snapshot(state):
    board = state.board
    return (id(board), id(board.move_stack), id(board._stack),
            tuple(id(item) for item in board.move_stack),
            tuple((id(item), dict(vars(item))) for item in board._stack),
            {key: value for key, value in vars(board).items() if key not in {"move_stack", "_stack"}},
            fingerprint(state))


class Recorder:
    def __init__(self, preferred=(), value=0.0):
        self.preferred = preferred
        self.value = value
        self.calls = []

    def __call__(self, states):
        assert len(states) == 1
        state, = states
        assert terminal_result(state) is None
        self.calls.append(fingerprint(state))
        uniform, = evaluate(states)
        priors = uniform.priors.copy()
        depth = len(self.calls) - 1
        if depth < len(self.preferred):
            move_id = move_to_id(state, chess.Move.from_uci(self.preferred[depth]))
            priors[:] = 0
            priors[move_id] = 1
        return [Evaluation(priors, [(1 + self.value) / 2, 0, (1 - self.value) / 2], self.value)]


def stats(node):
    return tuple((edge.move_id, edge.visit_count, edge.value_sum) for edge in node.edges.values())


@pytest.mark.parametrize("field,bad", [
    ("c_puct", 0), ("c_puct", -1), ("c_puct", float("nan")), ("c_puct", float("inf")),
    ("c_puct", True), ("c_puct", "1.5"), ("c_puct", None),
    ("unvisited_value", -1.01), ("unvisited_value", 1.01), ("unvisited_value", float("nan")),
    ("unvisited_value", float("inf")), ("unvisited_value", False), ("unvisited_value", "0"),
])
def test_config_rejects_invalid(field, bad):
    with pytest.raises(ValueError, match=field):
        SearchConfig(**{field: bad})


def test_config_defaults_and_valid_overrides():
    assert SearchConfig().c_puct == 1.5 and SearchConfig().unvisited_value == 0
    assert SearchConfig(c_puct=2, unvisited_value=-1).unvisited_value == -1
    assert SearchConfig(c_puct=0.001, unvisited_value=1).unvisited_value == 1
    with pytest.raises(AttributeError):
        SearchConfig().c_puct = 2


@pytest.mark.parametrize("fields", [
    dict(move_id=-1), dict(move_id=4352), dict(move_id=True), dict(prior=-0.1),
    dict(prior=1.1), dict(prior=float("nan")), dict(visit_count=-1),
    dict(visit_count=True), dict(visit_count=1.5), dict(value_sum=0.1),
    dict(visit_count=2, value_sum=2.1), dict(value_sum=float("inf")),
])
def test_edge_validates_initial_statistics(fields):
    with pytest.raises(ValueError):
        Edge(**(dict(move_id=1, prior=0.1) | fields))


def test_independent_nonzero_scores_and_selection():
    node = Node(make_state())
    expand(node)
    # Controlled completed statistics; no production backup is invoked.
    node.edges = {796: Edge(796, 0.2, 4, 2), 731: Edge(731, 0.5, 9, -4.5),
                  82: Edge(82, 0.2, 3, 0.3), 405: Edge(405, 0.1)}
    assert node.parent_visits == 16
    # sqrt(16)=4: .5+.24, -.5+.30, .1+.30, 0+.60.
    expected = {796: 0.74, 731: -0.2, 82: 0.4, 405: 0.6}
    before = stats(node)
    for move_id, edge in node.edges.items():
        assert puct_score(edge, 16) == pytest.approx(expected[move_id], abs=1e-14)
    assert select_edge(node) is node.edges[796]
    assert select_edge(node, SearchConfig(c_puct=3)) is node.edges[405]
    assert stats(node) == before


def test_unvisited_value_and_zero_parent_behavior():
    edge = Edge(1, 0.4)
    assert puct_score(edge, 0) == pytest.approx(0.6)
    assert puct_score(edge, 0, SearchConfig(unvisited_value=-0.2)) == pytest.approx(0.4)
    assert puct_score(edge, 9) == pytest.approx(1.8)
    assert puct_score(Edge(2, 0), 0) == 0
    # Visited Q overrides the configured first-play value.
    assert puct_score(Edge(3, 0.4, 2, -1), 9, SearchConfig(unvisited_value=1)) == pytest.approx(0.1)


@pytest.mark.parametrize("visits", [-1, True, 1.5, 2])
def test_invalid_parent_visits(visits):
    with pytest.raises(ValueError, match="parent_visits"):
        puct_score(Edge(1, 0.2, 3, 0.5), visits)


def test_initial_selection_uses_highest_prior_and_exact_ties_ignore_insertion():
    recorder = Recorder(["e2e4"])
    root = Node(make_state())
    expand(root, recorder)
    wanted = move_to_id(root.state, chess.Move.from_uci("e2e4"))
    assert select_edge(root).move_id == wanted
    uniform = Node(make_state())
    expand(uniform)
    smallest = min(uniform.edges)
    uniform.edges = dict(reversed(list(uniform.edges.items())))
    assert select_edge(uniform).move_id == smallest
    # Exact nonzero score tie: both Q=.5, P=.2, N=2.
    uniform.edges = {9: Edge(9, 0.2, 2, 1), 3: Edge(3, 0.2, 2, 1)}
    assert select_edge(uniform).move_id == 3
    # No approximate-tie tolerance: a representably higher score wins.
    uniform.edges[9].value_sum = 1.000000000001
    assert select_edge(uniform).move_id == 9


@pytest.mark.parametrize("state", [
    make_state(), make_state(moves_uci=["e2e4"]),
    make_state(CASTLING.format("w")), make_state(CASTLING.format("b")),
    make_state(PROMOTION), make_state(BLACK_PROMOTION),
    make_state(moves_uci=["e2e4", "a7a6", "e4e5", "d7d5"]),
])
def test_legal_expansion_retains_priors_zero_edges_and_value_perspective(state):
    root = Node(state)
    before = snapshot(state), snapshot(root._state)
    recorder = Recorder(value=0.6)
    result = expand(root, recorder)
    expected_ids = {move_to_id(state, move) for move in state.board.generate_legal_moves()}
    assert set(root.edges) == expected_ids
    for move_id, edge in root.edges.items():
        assert edge.prior == float(result.priors[move_id])
        assert edge.visit_count == 0 and edge.value_sum == 0 and edge.child is None
    assert root.result.value == 0.6  # Positive means root player, independent of color.
    assert root.parent_visits == 0
    edges = root.edges
    assert expand(root, recorder) is result and root.edges is edges
    assert len(recorder.calls) == 1
    assert (snapshot(state), snapshot(root._state)) == before


def terminal_fixtures():
    return [
        (make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 150 1"), "CHECKMATE", chess.WHITE),
        (make_state("8/8/8/8/8/5k2/6q1/7K w - - 0 1"), "CHECKMATE", chess.BLACK),
        (make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"), "STALEMATE", None),
        (make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "INSUFFICIENT_MATERIAL", None),
        (make_state(moves_uci=LONG_CYCLE * 2), "THREEFOLD_REPETITION", None),
        (make_state(chess.STARTING_FEN, CYCLE * 2), "THREEFOLD_REPETITION", None),
        (make_state("4k3/8/8/8/8/8/8/R3K3 b - - 100 1"), "FIFTY_MOVES", None),
        (make_state(moves_uci=CYCLE * 4), "FIVEFOLD_REPETITION", None),
        (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 150 1"), "SEVENTYFIVE_MOVES", None),
    ]


@pytest.mark.parametrize("state,reason,winner", terminal_fixtures())
def test_terminal_bypasses_evaluator_and_mask_preserves_exact_adapter_result(state, reason, winner, monkeypatch):
    root = Node(state)
    before = snapshot(state), snapshot(root._state)
    expected = terminal_result(root.state)
    def forbidden(*args):
        raise AssertionError("Terminal entered evaluator/legal expansion")
    monkeypatch.setattr(puct, "terminal_result", lambda candidate: expected if candidate is root._state else forbidden())
    monkeypatch.setattr(puct, "legal_mask", forbidden)
    assert expand(root, forbidden) is expected
    assert expand(root, forbidden) is expected
    leaf = traverse(root, evaluator=forbidden)
    assert leaf.path == () and leaf.leaf is root and leaf.result is expected
    assert expected.ending_reason == reason and expected.winner is winner
    assert expected.history_complete == state.history_complete
    assert expected.value == (0 if winner is None else -1)
    np.testing.assert_array_equal(expected.wdl, [0, 1, 0] if winner is None else [0, 0, 1])
    assert expected.priors is None and root.edges == {} and root.parent_visits == 0
    assert (snapshot(state), snapshot(root._state)) == before
    with pytest.raises(ValueError, match="expanded nonterminal"):
        select_edge(root)


@pytest.mark.parametrize("state", [
    make_state(moves_uci=["d2d4", "d7d5"]),
    make_state(moves_uci=LONG_CYCLE + ["d2d4", "d7d5"]),
    make_state(chess.STARTING_FEN, LONG_CYCLE + ["d2d4", "d7d5"], history_complete=False),
])
def test_root_preparation_is_separate_from_traversals_no_statistics_writes(state):
    root = Node(state)
    with pytest.raises(ValueError, match="Expand the root"):
        traverse(root)
    with pytest.raises(ValueError, match="expanded nonterminal"):
        select_edge(root)
    recorder = Recorder(["e2e4", "e7e5", "g1f3"], value=0.6)
    expand(root, recorder)
    assert root.parent_visits == 0 and len(recorder.calls) == 1
    before = snapshot(state), snapshot(root._state), stats(root)
    first = traverse(root, evaluator=recorder)
    child_before = snapshot(first.leaf._state), stats(first.leaf)
    second = traverse(root, evaluator=recorder)
    assert len(first.path) == 1 and len(second.path) == 2
    assert second.path[0].edge is first.path[0].edge
    assert second.path[0].child is first.leaf
    assert second.path[1].parent is first.leaf
    assert first.result.value == second.result.value == 0.6  # Leaf player's perspective.
    assert (snapshot(first.leaf._state), stats(first.leaf)) == child_before
    assert (snapshot(state), snapshot(root._state), stats(root)) == before
    for step in second.path:
        assert step.edge.visit_count == step.edge.value_sum == 0
        assert step.child.state.board.move_stack[:-1] == step.parent.state.board.move_stack
        assert step.child.state.initial_fen == state.initial_fen
    assert len(recorder.calls) == 3


def test_repeatable_traversal_sequences_with_unchanged_config():
    sequences = []
    for _ in range(2):
        root = Node(make_state(moves_uci=["e2e4", "e7e5"]))
        expand(root)
        sequences.append([tuple(step.edge.move_id for step in traverse(root).path) for _ in range(5)])
    assert sequences[0] == sequences[1]
    assert [len(path) for path in sequences[0]] == [1, 2, 3, 4, 5]


def special_cases():
    cases = []
    for color, rank in [("w", "1"), ("b", "8")]:
        for destination in ("g", "c"):
            cases.append((make_state(CASTLING.format(color)), f"e{rank}{destination}{rank}"))
    cases += [(make_state(moves_uci=["e2e4", "a7a6", "e4e5", "d7d5"]), "e5d6"),
              (make_state(moves_uci=["a2a3", "e7e5", "d2d4", "e5e4", "f2f4"]), "e4f3")]
    for fen, origin, rank in [(PROMOTION, "e7", "8"), (BLACK_PROMOTION, "e2", "1")]:
        for destination in "def":
            for piece in "qrbn":
                cases.append((make_state(fen, history_complete=True), origin + destination + rank + piece))
    return cases


@pytest.mark.parametrize("state,uci", special_cases())
def test_special_move_traversal_and_every_prefix_preserved(state, uci):
    before = snapshot(state)
    features = encode_position(state).features.copy()
    root = Node(state)
    root_before = snapshot(root._state)
    recorder = Recorder([uci])
    expand(root, recorder)
    selected = traverse(root, evaluator=recorder)
    assert len(selected.path) == 1
    step, = selected.path
    assert step.parent is root and step.child is selected.leaf and step.edge.child is selected.leaf
    assert step.edge.move_id == move_to_id(state, chess.Move.from_uci(uci))
    reference = state.board.copy(stack=True)
    reference.push_uci(uci)  # Independent actual-coordinate reference application.
    leaf = selected.leaf.state
    assert leaf.board.fen(en_passant="fen") == reference.fen(en_passant="fen")
    assert leaf.board.promoted == reference.promoted
    assert leaf.initial_fen == state.initial_fen and leaf.history_complete == state.history_complete
    assert tuple(move.uci() for move in leaf.board.move_stack) == tuple(move.uci() for move in reference.move_stack)
    prefix = leaf.copy()
    prefix.board.pop()
    assert fingerprint(prefix) == fingerprint(state)
    assert snapshot(state) == before and snapshot(root._state) == root_before
    np.testing.assert_array_equal(encode_position(state).features, features)
    assert len(root.edges) == len(list(state.board.legal_moves))  # Includes zero-prior legal edges.
    assert all(edge.visit_count == edge.value_sum == 0 for edge in root.edges.values())


@pytest.mark.parametrize("state,uci,reason", [
    (make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"), "g6g7", "CHECKMATE"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1"), "a1a2", "FIFTY_MOVES"),
    (make_state(moves_uci=CYCLE + CYCLE[:3]), "f6g8", "THREEFOLD_REPETITION"),
])
def test_terminal_leaf_bypass_repeat_and_no_visit_completion(state, uci, reason):
    root = Node(state)
    recorder = Recorder([uci])
    expand(root, recorder)
    before = snapshot(root._state)
    first = traverse(root, evaluator=recorder)
    second = traverse(root, evaluator=recorder)
    assert first.leaf is second.leaf and first.result is second.result
    assert len(first.path) == len(second.path) == 1
    assert isinstance(first.result, TerminalResult) and first.result.ending_reason == reason
    assert len(recorder.calls) == 1 and first.leaf.edges == {}
    assert root.parent_visits == 0 and snapshot(root._state) == before
    if reason != "CHECKMATE":
        assert list(first.leaf.state.board.legal_moves)


def test_same_current_fen_different_histories_remain_separate():
    states = [make_state(moves_uci=CYCLE), make_state(moves_uci=["b1c3", "b8c6", "c3b1", "c6b8"])]
    assert states[0].board.fen(en_passant="fen") == states[1].board.fen(en_passant="fen")
    assert not np.array_equal(encode_position(states[0]).features, encode_position(states[1]).features)
    recorder = Recorder()
    roots = [Node(state) for state in states]
    for root in roots:
        expand(root, recorder)
    leaves = [traverse(root, evaluator=recorder) for root in roots]
    assert len(recorder.calls) == 4 and recorder.calls[0] != recorder.calls[1]
    assert roots[0].result is not roots[1].result
    assert leaves[0].leaf is not leaves[1].leaf
    assert leaves[0].leaf.state.board.fen() == leaves[1].leaf.state.board.fen()
    assert leaves[0].leaf.state.board.move_stack != leaves[1].leaf.state.board.move_stack
    # Identical exact FEN can also have different adjudication due to history.
    drawn = make_state(moves_uci=LONG_CYCLE * 2)
    isolated = make_state(drawn.board.fen(en_passant="fen"))
    draw_root, isolated_root = Node(drawn), Node(isolated)
    assert isinstance(expand(draw_root, recorder), TerminalResult)
    assert isinstance(expand(isolated_root, recorder), Evaluation)
    assert len(recorder.calls) == 5


def test_distinct_paths_with_same_fen_within_one_tree_are_not_merged():
    root = Node(make_state())
    expand(root)
    paths = [("g1f3", "g8f6", "b1c3", "b8c6"), ("b1c3", "b8c6", "g1f3", "g8f6")]
    recorder = Recorder()
    leaves = []
    for moves in paths:
        node = root
        for uci in moves:
            wanted = move_to_id(node.state, chess.Move.from_uci(uci))
            for edge in node.edges.values():
                edge.prior = 1.0 if edge.move_id == wanted else 0.0
            resolved = traverse(root, evaluator=recorder)
            node = resolved.leaf
        leaves.append(node)
    assert leaves[0].state.board.fen() == leaves[1].state.board.fen()
    assert leaves[0] is not leaves[1] and leaves[0].result is not leaves[1].result
    assert leaves[0].state.board.move_stack != leaves[1].state.board.move_stack
    assert len(recorder.calls) == 8


def test_failed_leaf_evaluation_never_completes_visits_and_retry_is_safe():
    root = Node(make_state())
    recorder = Recorder(["e2e4"])
    expand(root, recorder)
    selected = select_edge(root)
    selected.visit_count, selected.value_sum = 2, 1.0
    before = snapshot(root._state), stats(root)
    def interrupted(states):
        raise RuntimeError("controlled evaluation interruption")
    with pytest.raises(RuntimeError, match="interruption"):
        traverse(root, evaluator=interrupted)
    assert (snapshot(root._state), stats(root)) == before
    child = select_edge(root).child
    assert child is not None and not child.expanded and child.edges == {}
    result = traverse(root, evaluator=recorder)
    assert result.leaf is child and len(result.path) == 1
    assert (snapshot(root._state), stats(root)) == before


@pytest.mark.parametrize("bad", ["empty", "extra", "terminal", "illegal"])
def test_failed_expansion_is_unpublished_and_retryable(bad):
    root = Node(make_state())
    before = snapshot(root._state)
    valid = evaluate([root.state])[0]
    if bad == "empty":
        result = []
    elif bad == "extra":
        result = [valid, valid]
    elif bad == "terminal":
        result = [terminal_result(make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))]
    else:
        priors = np.zeros(4352, dtype=np.float32)
        priors[0] = 1
        result = [Evaluation(priors, [0.25, 0.5, 0.25], 0)]
    with pytest.raises(ValueError):
        expand(root, lambda states: result)
    assert not root.expanded and root.edges == {} and root.parent_visits == 0
    assert snapshot(root._state) == before
    assert expand(root) is not None


def test_node_owns_state_copy_even_when_caller_later_moves():
    state = make_state(moves_uci=["e2e4"])
    root = Node(state)
    expected = fingerprint(root.state)
    state.board.push_uci("e7e5")
    assert fingerprint(root.state) == expected
    assert root.state.board is not state.board


def test_cpu_import_boundary_and_no_encoding():
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
    raise AssertionError("S02 encoded features")
adapter.encode_position = forbidden
from chess_rl.search.puct import Node, expand, traverse
root = Node(adapter.make_state())
expand(root)
assert len(traverse(root).path) == 1
assert root.parent_visits == 0
print("CPU search boundary passed")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
