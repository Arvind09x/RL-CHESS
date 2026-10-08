import json
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import (apply_move, id_to_move, make_state, move_to_id,
                              terminal_result)
from chess_rl.search.evaluator import evaluate
from chess_rl.search.game import run_game
from chess_rl.search.puct import Node, SearchConfig
from chess_rl.search.runner import search
from .helpers import CYCLE, fingerprint

WHITE_MATE = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"
BLACK_MATE = chess.Board(WHITE_MATE).mirror().fen()
LONG_OPENING = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6".split()


def replay(record):
    # Independently rebuild and apply the actual trace, rather than re-running
    # the runner or reconstructing its final FEN as a new rule state.
    state = make_state(record.initial_fen, record.opening_prefix,
                       history_complete=record.history_complete)
    for uci in record.moves_uci[len(record.opening_prefix):]:
        assert terminal_result(state) is None
        move = chess.Move.from_uci(uci)
        assert move in state.board.legal_moves
        apply_move(state, move_to_id(state, move))
    assert state.board.fen(en_passant="fen") == record.final_fen
    assert state.board.promoted == record.final_promoted
    assert [m.uci() for m in state.board.move_stack] == list(record.moves_uci)
    result = terminal_result(state)
    if record.status == "completed":
        assert result is not None and result.ending_reason == record.ending_reason
        assert record.outcome == ("1/2-1/2" if result.winner is None else
                                  "1-0" if result.winner else "0-1")
    else:
        assert record.outcome is record.winner is None
    return state


@pytest.mark.parametrize("fen,color", [(WHITE_MATE, chess.WHITE), (BLACK_MATE, chess.BLACK)])
def test_uniform_dummy_finds_immediate_mate_and_parent_winning_values(fen, color):
    state = make_state(fen)
    before = fingerprint(state)
    # 64 is deliberate and recorded: uniform priors, no mate-biased evaluator.
    root = Node(state)
    result = search(root, 64)
    assert result.completed_simulations == root.parent_visits == 64
    assert result.move in state.board.legal_moves
    assert id_to_move(state, result.move_id) == result.move
    edge = root.edges[result.move_id]
    assert edge.visit_count > 1 and edge.value_sum == edge.visit_count
    assert edge.child.result.value == -1  # Child loss; parent W/N=+1.
    assert terminal_result(edge.child.state).winner is color
    record = run_game(state, 64, ply_cap=1)
    assert record.status == "completed" and record.ending_reason == "CHECKMATE"
    assert record.winner == ("white" if color else "black")
    assert record.moves_uci == (result.move.uci(),)
    assert record.attempts[0].completed_simulations == 64
    assert fingerprint(state) == before
    replay(record)


TERMINALS = [
    (make_state(moves_uci="f2f3 e7e5 g2g4 d8h4".split()), "CHECKMATE"),
    (make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"), "STALEMATE"),
    (make_state(moves_uci=CYCLE * 2), "THREEFOLD_REPETITION"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 100 1"), "FIFTY_MOVES"),
]


@pytest.mark.parametrize("state,reason", TERMINALS)
@pytest.mark.parametrize("cap", [0, 1, 100])
def test_terminal_precedes_search_and_cap(state, reason, cap):
    before = fingerprint(state)
    def forbidden(states):
        pytest.fail("Terminal states must never reach evaluator")
    record = run_game(state, 4, ply_cap=cap, evaluator=forbidden,
                      evaluator_id="must-not-evaluate")
    assert record.status == "completed" and record.ending_reason == reason
    assert not record.attempts
    assert fingerprint(state) == before
    replay(record)


def line_evaluator(uci):
    def controlled(states):
        results = evaluate(states)
        for state, result in zip(states, results, strict=True):
            move = chess.Move.from_uci(uci)
            if move in state.board.legal_moves and terminal_result(state) is None:
                result.priors[:] = 0
                result.priors[move_to_id(state, move)] = 1
        return results
    return controlled


@pytest.mark.parametrize("state,uci,reason", [
    (make_state(moves_uci=(CYCLE * 2)[:-1]), "f6g8", "THREEFOLD_REPETITION"),
    (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1"), "a1b1", "FIFTY_MOVES"),
    (make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"), "g6g7", "CHECKMATE"),
    (make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"), "g6f7", "STALEMATE"),
])
def test_final_allowed_move_terminal_wins_over_cap(state, uci, reason):
    assert terminal_result(state) is None
    record = run_game(state, 4, ply_cap=len(state.board.move_stack) + 1,
                      evaluator=line_evaluator(uci), evaluator_id="controlled-onehot-" + uci)
    assert record.status == "completed" and record.ending_reason == reason
    assert record.moves_uci[-1] == uci
    replay(record)


@pytest.mark.parametrize("state", [
    make_state(moves_uci=(CYCLE * 2)[:-1]),
    make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1"),
])
def test_intended_claim_alone_is_not_terminal(state):
    assert state.board.can_claim_draw() and terminal_result(state) is None
    at_cap = run_game(state, 4, ply_cap=len(state.board.move_stack))
    assert at_cap.status == "truncated" and at_cap.outcome is None
    replay(at_cap)
    # Choose a pawn move for repetition, king move still reaches fifty-move
    # draw: the key boundary is that search executes, not a pre-search draw.
    active = run_game(state, 1, ply_cap=len(state.board.move_stack) + 1)
    assert len(active.attempts) == 1 and active.attempts[0].completed_simulations == 1
    replay(active)


@pytest.mark.parametrize("complete", [True, False])
def test_full_history_ownership_opening_cap_settings_and_json(complete):
    state = make_state(moves_uci=LONG_OPENING, history_complete=complete)
    before = fingerprint(state)
    config = SearchConfig(c_puct=2, unvisited_value=.2)
    record = run_game(state, 4, ply_cap=len(LONG_OPENING) + 3, config=config)
    assert record.status == "truncated" and record.ending_reason == "ply_limit"
    assert record.opening_prefix == tuple(LONG_OPENING)
    assert len(record.moves_uci) == len(LONG_OPENING) + 3
    assert [a.ply for a in record.attempts] == list(range(14, 17))
    assert all(a.completed_simulations == a.requested_simulations == 4 for a in record.attempts)
    assert record.search_config == config and record.history_complete is complete
    assert fingerprint(state) == before
    rebuilt = replay(record)
    assert rebuilt.history_complete is complete
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["search_config"] == {"c_puct": 2, "unvisited_value": .2}
    assert payload["rng"] is None and payload["exploration_noise"] is False
    assert payload["outcome"] is None
    state.board.pop()  # Returned history is independent of subsequent caller edits.
    assert record.opening_prefix == tuple(LONG_OPENING)


def fault_evaluator(error_type, fail_at):
    calls = 0
    def fault(states):
        nonlocal calls
        assert len(states) == 1
        calls += 1
        if calls == fail_at:
            raise error_type("injected S05 evaluator fault")
        return evaluate(states)
    return fault


@pytest.mark.parametrize("error_type,status", [(RuntimeError, "failed"),
                                              (KeyboardInterrupt, "interrupted"),
                                              (SystemExit, "interrupted")])
@pytest.mark.parametrize("fail_at,moves,counts", [(1, 0, [0]), (4, 0, [2]), (9, 1, [4, 2])])
def test_injected_search_fault_preserves_exact_completions_and_actual_history(error_type, status,
                                                                           fail_at, moves, counts):
    state = make_state()
    before = fingerprint(state)
    record = run_game(state, 4, ply_cap=10, evaluator=fault_evaluator(error_type, fail_at),
                      evaluator_id=f"fault-{error_type.__name__}-call-{fail_at}")
    assert record.status == status and record.outcome is record.winner is None
    assert record.error_type == error_type.__name__
    assert len(record.moves_uci) == moves
    assert [a.completed_simulations for a in record.attempts] == counts
    assert record.attempts[-1].status == status and record.attempts[-1].move_id is None
    assert all(sum(n for _, n, _ in a.visited_edges) == a.completed_simulations
               for a in record.attempts)
    assert fingerprint(state) == before
    replay(record)


@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("error_type,status", [(RuntimeError, "failed"),
                                              (KeyboardInterrupt, "interrupted")])
def test_fault_retains_long_opening_and_provenance(complete, error_type, status):
    state = make_state(moves_uci=LONG_OPENING, history_complete=complete)
    record = run_game(state, 4, ply_cap=30, evaluator=fault_evaluator(error_type, 4),
                      evaluator_id="long-opening-injected-fault")
    assert record.status == status and record.opening_prefix == tuple(LONG_OPENING)
    assert record.moves_uci == tuple(LONG_OPENING)
    assert record.attempts[0].completed_simulations == 2
    assert replay(record).history_complete is complete


@pytest.mark.parametrize("error_type,status", [(RuntimeError, "failed"),
                                              (KeyboardInterrupt, "interrupted")])
def test_apply_fault_does_not_publish_partial_history(monkeypatch, error_type, status):
    import chess_rl.search.game as game
    real_apply = game.apply_move
    def broken(state, move_id):
        real_apply(state, move_id)
        raise error_type("after candidate push")
    monkeypatch.setattr(game, "apply_move", broken)
    state = make_state(moves_uci=LONG_OPENING)
    before = fingerprint(state)
    record = run_game(state, 4, ply_cap=30)
    assert record.status == status and record.moves_uci == tuple(LONG_OPENING)
    assert record.attempts[0].completed_simulations == 4
    assert fingerprint(state) == before
    replay(record)


@pytest.mark.parametrize("budget,cap", [(0, 1), (-1, 1), (True, 1), (np.int64(1), 1),
                                       (1, -1), (1, True), (1, np.int64(1))])
def test_invalid_budget_or_cap_rejected_without_work(budget, cap):
    with pytest.raises(ValueError):
        run_game(make_state(), budget, ply_cap=cap)


def test_custom_evaluator_requires_identity():
    with pytest.raises(ValueError, match="evaluator_id"):
        run_game(make_state(), 1, ply_cap=1, evaluator=lambda states: evaluate(states))


def test_cpu_boundary():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("torch.", "chess.engine", "chess_rl.model", "chess_rl.training", "chess_rl.data")):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, Block())
from chess_rl.adapter import make_state
from chess_rl.search.game import run_game
r = run_game(make_state(), 4, ply_cap=3)
assert r.status == "truncated" and len(r.moves_uci) == 3
assert sum(a.completed_simulations for a in r.attempts) == 12
print("CPU S05 boundary passed")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
