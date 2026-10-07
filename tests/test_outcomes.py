import chess
import numpy as np
import pytest

from chess_rl.adapter import make_state, terminal_result, legal_mask, apply_move, move_to_id
from .helpers import CYCLE, CASTLING, fingerprint


@pytest.mark.parametrize("fen,reason,winner,wdl,value", [
    ("7k/6Q1/5K2/8/8/8/8/8 b - - 150 1", "CHECKMATE", chess.WHITE, [0,0,1], -1),
    ("8/8/8/8/8/5k2/6q1/7K w - - 0 1", "CHECKMATE", chess.BLACK, [0,0,1], -1),
    ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "STALEMATE", None, [0,1,0], 0),
    ("4k3/8/8/8/8/8/8/4K3 w - - 150 1", "INSUFFICIENT_MATERIAL", None, [0,1,0], 0),
    ("4k3/8/8/8/8/8/8/R3K3 w - - 150 1", "SEVENTYFIVE_MOVES", None, [0,1,0], 0),
    ("4k3/8/8/8/8/8/8/R3K3 w - - 100 1", "FIFTY_MOVES", None, [0,1,0], 0),
])
def test_exact_outcomes_and_precedence(fen, reason, winner, wdl, value):
    state = make_state(fen)
    before = fingerprint(state)
    result = terminal_result(state)
    assert result.ending_reason == reason and result.winner is winner
    np.testing.assert_array_equal(result.wdl, wdl)
    assert result.value == value and result.priors is None
    assert not result.history_complete
    assert fingerprint(state) == before
    with pytest.raises(ValueError, match="termination"):
        apply_move(state, 0)
    assert fingerprint(state) == before


def test_actual_repetition_and_automatic_fivefold():
    for repetitions, reason in [(2, "THREEFOLD_REPETITION"), (4, "FIVEFOLD_REPETITION")]:
        state = make_state(moves_uci=CYCLE * repetitions)
        before = fingerprint(state)
        result = terminal_result(state)
        assert result.ending_reason == reason and result.history_complete
        assert result.value == 0
        assert legal_mask(state).sum() == 20
        move_id = move_to_id(state, chess.Move.from_uci("e2e4"))
        with pytest.raises(ValueError, match="termination"):
            apply_move(state, move_id)
        assert fingerprint(state) == before


def test_repetition_precedes_reached_fifty_move_claim():
    state = make_state(chess.STARTING_FEN.replace("0 1", "92 1"), CYCLE * 2)
    assert state.board.halfmove_clock == 100 and state.board.is_repetition(3)
    assert terminal_result(state).ending_reason == "THREEFOLD_REPETITION"
    assert not terminal_result(state).history_complete


def test_intended_move_only_repetition_remains_nonterminal():
    state = make_state(moves_uci=CYCLE + CYCLE[:3])
    assert state.board.can_claim_threefold_repetition()
    assert not state.board.is_repetition(3)
    assert terminal_result(state) is None
    apply_move(state, move_to_id(state, chess.Move.from_uci("f6g8")))
    assert terminal_result(state).ending_reason == "THREEFOLD_REPETITION"


def test_intended_move_only_fifty_moves_remains_nonterminal():
    state = make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1")
    assert state.board.can_claim_fifty_moves() and not state.board.is_fifty_moves()
    assert terminal_result(state) is None
    apply_move(state, move_to_id(state, chess.Move.from_uci("a1a2")))
    assert terminal_result(state).ending_reason == "FIFTY_MOVES"
    assert legal_mask(state).any()


def test_castling_increments_clock_and_pawn_capture_reset():
    state = make_state(CASTLING.format("w"))
    apply_move(state, move_to_id(state, chess.Move.from_uci("e1g1")))
    assert state.board.halfmove_clock == 18
    state = make_state(moves_uci=["g1f3", "d7d5"])
    apply_move(state, move_to_id(state, chess.Move.from_uci("e2e4")))
    assert state.board.halfmove_clock == 0
    apply_move(state, move_to_id(state, chess.Move.from_uci("d5e4")))
    assert state.board.halfmove_clock == 0


@pytest.mark.parametrize("move_id", [-1, 4352, 2.0, True, None, 0, 4096])
def test_failed_application_is_atomic(move_id):
    state = make_state(moves_uci=["e2e4", "e7e5", "g1f3"])
    before = fingerprint(state)
    with pytest.raises(ValueError):
        apply_move(state, move_id)
    assert fingerprint(state) == before


def test_incomplete_provenance_survives_application():
    state = make_state(chess.STARTING_FEN)
    apply_move(state, 796)
    assert not state.history_complete and len(state.board.move_stack) == 1
