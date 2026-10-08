"""S01 checks against enumerated chess moves, exact outcomes and full history."""
import os
from pathlib import Path
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, TerminalResult, encode_position, make_state, move_to_id, terminal_result
from chess_rl.search import evaluator
from chess_rl.search.evaluator import evaluate
from .helpers import BLACK_PROMOTION, CASTLING, CYCLE, LONG_CYCLE, PROMOTION, fingerprint


def preserved(state):
    """Include raw board fields, stack object identities and every prefix."""
    board = state.board
    raw = {key: value for key, value in vars(board).items() if key not in {"move_stack", "_stack"}}
    return (
        id(board), id(board.move_stack), id(board._stack),
        tuple(id(move) for move in board.move_stack),
        tuple((id(saved), dict(vars(saved))) for saved in board._stack),
        raw, fingerprint(state),
    )


def check_nonterminal(state, result):
    assert type(result) is Evaluation
    # Enumerate python-chess moves independently of evaluator's legal_mask.
    moves = list(state.board.generate_legal_moves())
    ids = [move_to_id(state, move) for move in moves]
    assert len(ids) == len(set(ids)) > 0
    priors = result.priors
    assert priors.shape == (4352,) and priors.dtype == np.float32
    assert np.isfinite(priors).all() and (priors >= 0).all()
    assert set(np.flatnonzero(priors)) == set(ids)
    np.testing.assert_array_equal(priors[ids], np.full(len(ids), 1 / len(ids), dtype=np.float32))
    assert abs(float(priors.sum(dtype=np.float64)) - 1) < 1e-6
    assert result.wdl.shape == (3,) and result.wdl.dtype == np.float32
    assert np.isfinite(result.wdl).all() and (result.wdl >= 0).all()
    np.testing.assert_array_equal(result.wdl, [0.25, 0.5, 0.25])
    assert result.wdl.sum() == 1
    assert result.wdl[0] == result.wdl[2]
    assert result.value == float(result.wdl[0] - result.wdl[2]) == 0.0
    assert priors.flags.owndata and result.wdl.flags.owndata


def nonterminal_cases():
    return [
        make_state(), make_state(moves_uci=["e2e4"]),
        make_state(CASTLING.format("w")), make_state(CASTLING.format("b")),
        make_state(moves_uci=["e2e4", "a7a6", "e4e5", "d7d5"]),
        make_state(moves_uci=["a2a3", "e7e5", "d2d4", "e5e4", "f2f4"]),
        make_state(PROMOTION), make_state(BLACK_PROMOTION),
        make_state(moves_uci=LONG_CYCLE + ["e2e4", "e7e5", "g1f3"]),
        make_state(chess.STARTING_FEN, LONG_CYCLE + ["e2e4"], history_complete=False),
        make_state(CASTLING.format("w"), ["e1g1"], history_complete=True),
        make_state(moves_uci=CYCLE + CYCLE[:3]),
        make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1"),
    ]


@pytest.mark.parametrize("index", range(13))
def test_batch_of_one_legal_support_values_and_preservation(index):
    state = nonterminal_cases()[index]
    assert terminal_result(state) is None
    before = preserved(state)
    encoded = encode_position(state)
    result, = evaluate((state,))
    check_nonterminal(state, result)
    assert preserved(state) == before
    after = encode_position(state)
    np.testing.assert_array_equal(after.features, encoded.features)
    assert (after.history_complete, after.available_snapshots) == (encoded.history_complete, encoded.available_snapshots)


@pytest.mark.parametrize("color", ["w", "b"])
def test_special_moves_really_are_present(color):
    castling = make_state(CASTLING.format(color))
    moves = {move.uci() for move in castling.board.legal_moves}
    assert ({"e1g1", "e1c1"} if color == "w" else {"e8g8", "e8c8"}) <= moves
    check_nonterminal(castling, evaluate([castling])[0])
    promotion = make_state(PROMOTION if color == "w" else BLACK_PROMOTION)
    moves = list(promotion.board.legal_moves)
    assert {move.promotion for move in moves if move.promotion} == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
    assert len([move for move in moves if move.promotion]) == 12  # forward and both captures
    check_nonterminal(promotion, evaluate([promotion])[0])
    ep = nonterminal_cases()[4 if color == "w" else 5]
    assert ep.board.has_legal_en_passant()
    assert any(ep.board.is_en_passant(move) for move in ep.board.legal_moves)
    check_nonterminal(ep, evaluate([ep])[0])


def terminal_cases():
    return [
        (make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 150 1"), "CHECKMATE", chess.WHITE),
        (make_state("8/8/8/8/8/5k2/6q1/7K w - - 0 1"), "CHECKMATE", chess.BLACK),
        (make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"), "STALEMATE", None),
        (make_state("8/8/8/8/8/6q1/5k2/7K w - - 0 1"), "STALEMATE", None),
        (make_state(moves_uci=LONG_CYCLE * 2), "THREEFOLD_REPETITION", None),
        (make_state(chess.STARTING_FEN, CYCLE * 2), "THREEFOLD_REPETITION", None),
        (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1", ["a1a2"]), "FIFTY_MOVES", None),
        (make_state("4k3/8/8/8/8/8/8/R3K3 b - - 100 1", history_complete=True), "FIFTY_MOVES", None),
        (make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "INSUFFICIENT_MATERIAL", None),
        (make_state(moves_uci=CYCLE * 4), "FIVEFOLD_REPETITION", None),
        (make_state("4k3/8/8/8/8/8/8/R3K3 w - - 150 1"), "SEVENTYFIVE_MOVES", None),
        (make_state(moves_uci=CYCLE * 2 + CYCLE[:1]), "THREEFOLD_REPETITION", None),
    ]


@pytest.mark.parametrize("index", range(12))
def test_terminal_exact_adapter_objects_bypass_nonterminal_path(index, monkeypatch):
    state, reason, winner = terminal_cases()[index]
    before = preserved(state)
    expected = terminal_result(state)
    assert expected.ending_reason == reason and expected.winner is winner
    assert expected.history_complete is state.history_complete
    if reason in {"THREEFOLD_REPETITION", "FIFTY_MOVES"}:
        assert list(state.board.legal_moves)  # terminal despite available actions
    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal entered nonterminal evaluation/encoding")
    monkeypatch.setattr(evaluator, "terminal_result", lambda candidate: expected if candidate is state else forbidden())
    monkeypatch.setattr(evaluator, "legal_mask", forbidden)
    monkeypatch.setattr(evaluator, "Evaluation", forbidden)
    monkeypatch.setattr("chess_rl.adapter.encode_position", forbidden)
    result, = evaluate([state])
    assert result is expected
    assert type(result) is TerminalResult and result.priors is None
    np.testing.assert_array_equal(result.wdl, [0, 1, 0] if winner is None else [0, 0, 1])
    assert result.value == (0.0 if winner is None else -1.0)
    assert preserved(state) == before


def test_empty_bypasses_adapter(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Empty input called adapter")
    monkeypatch.setattr(evaluator, "terminal_result", forbidden)
    monkeypatch.setattr(evaluator, "legal_mask", forbidden)
    assert evaluate([]) == evaluate(()) == []


def test_repeated_inputs_and_calls_own_arrays():
    state = make_state(moves_uci=["e2e4", "e7e5"])
    before = preserved(state)
    results = evaluate([state, state]) + evaluate([state])
    for result in results:
        check_nonterminal(state, result)
    for i, first in enumerate(results):
        for second in results[i + 1:]:
            assert first is not second
            assert not np.shares_memory(first.priors, second.priors)
            assert not np.shares_memory(first.wdl, second.wdl)
    results[0].priors[:] = 0
    results[0].wdl[:] = 0
    for result in results[1:] + evaluate([state]):
        check_nonterminal(state, result)
    assert preserved(state) == before


def test_mixed_batch_order_adjudication_first_and_preservation(monkeypatch):
    nonterminal = nonterminal_cases()
    terminals = [item[0] for item in terminal_cases()]
    states = [terminals[0], nonterminal[0], terminals[4], nonterminal[5], terminals[6], nonterminal[7]] + nonterminal + terminals
    before = [preserved(state) for state in states]
    expected = {id(state): terminal_result(state) for state in states}
    calls = []
    real_terminal, real_mask = evaluator.terminal_result, evaluator.legal_mask
    def adjudicate(state):
        calls.append(("terminal", id(state)))
        actual = real_terminal(state)
        reference = expected[id(state)]
        assert (actual is None) == (reference is None)
        if actual is not None:
            assert actual.ending_reason == reference.ending_reason
            np.testing.assert_array_equal(actual.wdl, reference.wdl)
        return reference
    def mask(state):
        assert calls[-1] == ("terminal", id(state)) and expected[id(state)] is None
        calls.append(("mask", id(state)))
        return real_mask(state)
    monkeypatch.setattr(evaluator, "terminal_result", adjudicate)
    monkeypatch.setattr(evaluator, "legal_mask", mask)
    results = evaluate(states)
    assert len(results) == len(states)
    assert [ident for kind, ident in calls if kind == "terminal"] == [id(state) for state in states]
    for state, result in zip(states, results):
        if expected[id(state)] is not None:
            assert result is expected[id(state)]
        else:
            check_nonterminal(state, result)
    assert [preserved(state) for state in states] == before


def test_intended_move_draw_claims_are_nonterminal():
    repetition, fifty = nonterminal_cases()[-2:]
    assert repetition.board.can_claim_threefold_repetition() and not repetition.board.is_repetition(3)
    assert fifty.board.can_claim_fifty_moves() and not fifty.board.is_fifty_moves()
    for state, result in zip([repetition, fifty], evaluate([repetition, fifty])):
        check_nonterminal(state, result)


def test_zero_legal_moves_fails_clearly_if_adapter_contract_is_broken(monkeypatch):
    monkeypatch.setattr(evaluator, "legal_mask", lambda state: np.zeros(4352, dtype=bool))
    with pytest.raises(ValueError, match="at least one legal move"):
        evaluate([make_state()])


def test_cpu_import_boundary_and_no_feature_encoding():
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
    raise AssertionError("Dummy evaluator encoded features")
adapter.encode_position = forbidden
from chess_rl.search.evaluator import evaluate
states = [adapter.make_state(), adapter.make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
assert len(evaluate(states)) == 2
assert "torch" not in sys.modules
assert "chess.engine" not in sys.modules
print("CPU dependency boundary and encoding bypass passed")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
