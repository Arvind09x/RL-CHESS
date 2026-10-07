"""Canonical pair4352-v1 table and chess legality (independent of endings)."""
from numbers import Integral

import chess
import numpy as np

from chess_rl.contracts import MOVE_COUNT
from .coordinates import transform_square
from .state import ChessState, require_legal_move

PROMOTION_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
MOVE_VOCABULARY = tuple((origin, destination, None)
    for origin in range(64) for destination in range(64)) + tuple(
    (48 + origin_file, 56 + destination_file, piece)
    for origin_file in range(8) for destination_file in range(8)
    for piece in PROMOTION_PIECES)
_MOVE_IDS = {entry: index for index, entry in enumerate(MOVE_VOCABULARY)}


def move_to_id(state: ChessState, move: chess.Move) -> int:
    require_legal_move(state.board, move)
    origin = transform_square(move.from_square, state.board.turn)
    destination = transform_square(move.to_square, state.board.turn)
    try:
        return _MOVE_IDS[(origin, destination, move.promotion)]
    except KeyError as exc:
        raise ValueError("Move is outside pair4352-v1") from exc


def id_to_move(state: ChessState, move_id: int) -> chess.Move:
    if isinstance(move_id, (bool, np.bool_)) or not isinstance(move_id, Integral) or not 0 <= move_id < MOVE_COUNT:
        raise ValueError("move_id must be an integer in 0..4351")
    origin, destination, promotion = MOVE_VOCABULARY[int(move_id)]
    move = chess.Move(transform_square(origin, state.board.turn),
                      transform_square(destination, state.board.turn), promotion=promotion)
    require_legal_move(state.board, move)
    return move


def legal_mask(state: ChessState) -> np.ndarray:
    """Chess-legal moves, including in adjudicated states with legal moves."""
    mask = np.zeros(MOVE_COUNT, dtype=np.bool_)
    for move in state.board.legal_moves:
        mask[move_to_id(state, move)] = True
    return mask
