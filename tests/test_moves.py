import chess
import numpy as np
import pytest

from chess_rl.adapter import make_state, move_to_id, id_to_move
from chess_rl.adapter.coordinates import transform_square
from chess_rl.adapter.moves import MOVE_VOCABULARY

from .helpers import CASTLING, PROMOTION, BLACK_PROMOTION


def test_coordinate_exhaustion_and_vocabulary():
    assert len(MOVE_VOCABULARY) == len(set(MOVE_VOCABULARY)) == 4352
    for square in range(64):
        assert transform_square(square, chess.WHITE) == square
        black = transform_square(square, chess.BLACK)
        assert black % 8 == square % 8
        assert black // 8 == 7 - square // 8
        assert transform_square(black, chess.BLACK) == square
    for move_id, (origin, destination, piece) in enumerate(MOVE_VOCABULARY):
        if move_id < 4096:
            assert move_id == origin * 64 + destination and piece is None
        else:
            offset = move_id - 4096
            assert origin == 48 + offset // 32
            assert destination == 56 + (offset // 4) % 8
            assert piece == [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][offset % 4]


@pytest.mark.parametrize("fen,uci,expected", [
    (None, "e2e4", 796),
    (CASTLING.format("w"), "e1g1", 262), (CASTLING.format("w"), "e1c1", 258),
    (CASTLING.format("b"), "e8g8", 262), (CASTLING.format("b"), "e8c8", 258),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 2347),
    ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3", 2284),
] + [(fen, uci + suffix, 4240 + index) for fen, uci in [(PROMOTION, "e7e8"), (BLACK_PROMOTION, "e2e1")]
     for index, suffix in enumerate("qrbn")])
def test_documented_and_special_ids(fen, uci, expected):
    state = make_state(fen)
    move = chess.Move.from_uci(uci)
    assert move_to_id(state, move) == expected
    assert id_to_move(state, expected).uci() == uci


def test_black_pawn_and_legal_roundtrips():
    state = make_state(moves_uci=["e2e4"])
    assert move_to_id(state, chess.Move.from_uci("e7e5")) == 796
    for fen in [None, CASTLING.format("w"), CASTLING.format("b"), PROMOTION, BLACK_PROMOTION]:
        state = make_state(fen)
        ids = [move_to_id(state, move) for move in state.board.legal_moves]
        assert len(ids) == len(set(ids))
        for move in state.board.legal_moves:
            assert id_to_move(state, move_to_id(state, move)) == move


@pytest.mark.parametrize("move_id", [-1, 4352, 1.5, True, np.bool_(False), "796", None, 0, 4096])
def test_reject_bad_or_illegal_ids(move_id):
    state = make_state()
    before = state.board.fen(), list(state.board.move_stack)
    with pytest.raises(ValueError):
        id_to_move(state, move_id)
    assert (state.board.fen(), state.board.move_stack) == before


def test_ordinary_slot_is_never_queen_promotion():
    state = make_state(PROMOTION)
    with pytest.raises(ValueError):
        id_to_move(state, 52 * 64 + 60)
    for move in [chess.Move.null(), chess.Move.from_uci("e7e6"), "e7e8q"]:
        with pytest.raises(ValueError):
            move_to_id(state, move)
    assert id_to_move(make_state(), np.int64(796)).uci() == "e2e4"


@pytest.mark.parametrize("uci", ["e1h1", "e1a1"])
def test_castling_alias_is_rejected(uci):
    state = make_state(CASTLING.format("w"))
    move = chess.Move.from_uci(uci)
    with pytest.raises(ValueError, match="noncanonical"):
        move_to_id(state, move)
    with pytest.raises(ValueError):
        id_to_move(state, 64 * move.from_square + move.to_square)
    with pytest.raises(ValueError):
        make_state(CASTLING.format("w"), [uci])


@pytest.mark.parametrize("move", [chess.Move(12.0, 28), chess.Move(12, 28.0), chess.Move(True, 18)])
def test_malformed_move_fields_are_value_errors(move):
    with pytest.raises(ValueError):
        move_to_id(make_state(), move)
