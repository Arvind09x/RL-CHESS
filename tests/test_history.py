import chess
import numpy as np

from chess_rl.adapter import make_state
from chess_rl.adapter.history import history_blocks

from .helpers import CYCLE


def test_start_and_two_ply_padding():
    start, count = history_blocks(make_state())
    assert count == 1 and start.shape == (64, 104)
    for t in range(8):
        np.testing.assert_array_equal(start[:, 13*t:13*(t+1)], start[:, :13])
    assert not start[:, 12::13].any()
    state = make_state(moves_uci=["e2e4", "e7e5"])
    blocks, count = history_blocks(state)
    assert count == 3
    assert blocks[chess.E4, 0] == blocks[chess.E5, 6] == 1
    assert blocks[chess.E4, 13] == blocks[chess.E7, 19] == 1
    for t in range(2, 8):
        np.testing.assert_array_equal(blocks[:, 13*t:13*(t+1)], start[:, :13])


def test_history_uses_current_black_ownership_and_reflection():
    state = make_state(moves_uci=["e2e4"])
    blocks, count = history_blocks(state)
    assert count == 2
    assert blocks[chess.E2, 0] == 1  # Black e7 is ours at network e2.
    assert blocks[chess.E5, 6] == 1  # White e4 is theirs at network e5.
    assert blocks[chess.E7, 19] == 1  # Historical White e2 -> e7; ownership unchanged.
    assert blocks[chess.E1, 5] == 1


def test_prefix_flags_padding_and_later_moves():
    state = make_state(moves_uci=CYCLE)
    blocks, count = history_blocks(state)
    assert count == 5
    assert list(blocks[0, 12::13]) == [1, 0, 0, 0, 0, 0, 0, 0]
    longer, count = history_blocks(make_state(moves_uci=CYCLE * 3))
    assert count == 8
    assert list(longer[0, 12::13]) == [1] * 8
    at_eight, _ = history_blocks(make_state(moves_uci=CYCLE * 2))
    np.testing.assert_array_equal(at_eight[:, :13], longer[:, 4*13:5*13])


def test_isolated_fen_and_known_incomplete_prefix():
    fen = chess.STARTING_FEN
    state = make_state(fen)
    blocks, count = history_blocks(state)
    assert count == 1 and not state.history_complete
    assert not blocks[:, 12::13].any()
    state = make_state(fen, CYCLE * 5)
    before = state.board.fen(en_passant="fen"), list(state.board.move_stack), state.initial_fen, state.history_complete
    blocks, count = history_blocks(state)
    assert count == 8 and not state.history_complete
    assert blocks[0, 12] == 1
    assert (state.board.fen(en_passant="fen"), state.board.move_stack, state.initial_fen, state.history_complete) == before


def test_history_survives_capture_and_pawn_moves():
    blocks, count = history_blocks(make_state(moves_uci=["e2e4", "d7d5", "e4d5"]))
    assert count == 4
    # Black's perspective across every snapshot after the white capture.
    assert blocks[chess.D4, 6] == 1
    assert blocks[chess.D4, 13] == 1
    assert blocks[chess.E5, 19] == 1
    assert blocks[chess.E7, 3*13+6] == 1
