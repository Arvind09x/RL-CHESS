"""The complete chess128-v2 CPU feature representation."""
import chess
import numpy as np

from .coordinates import transform_square
from .history import history_blocks
from .results import EncodedPosition
from .state import ChessState


def encode_position(state: ChessState) -> EncodedPosition:
    board = state.board
    player = board.turn
    features = np.zeros((64, 128), dtype=np.float32)
    features[:, :104], available = history_blocks(state)
    features[:, 104:108] = [
        board.has_queenside_castling_rights(player),
        board.has_kingside_castling_rights(player),
        board.has_queenside_castling_rights(not player),
        board.has_kingside_castling_rights(not player),
    ]
    features[:, 108] = player == chess.BLACK
    features[:, 109] = board.halfmove_clock / 100
    features[:, 111] = 1
    for channel, color in ((112, player), (113, not player)):
        for square in chess.SquareSet(board.occupied_co[color]):
            features[transform_square(square, player), channel] = 1
    features[:, 114] = [(square // 8 + square % 8) % 2 for square in range(64)]
    our_counts = np.array([len(board.pieces(piece, player)) for piece in range(1, 6)])
    their_counts = np.array([len(board.pieces(piece, not player)) for piece in range(1, 6)])
    features[:, 115:120] = (our_counts - their_counts) / 8
    features[:, 122:127] = our_counts / 8
    our_bishops = list(board.pieces(chess.BISHOP, player))
    their_bishops = list(board.pieces(chess.BISHOP, not player))
    if len(our_bishops) == len(their_bishops) == 1:
        def parity(square):
            return (square // 8 + square % 8) % 2
        features[:, 120] = parity(our_bishops[0]) != parity(their_bishops[0])
    for square in board.checkers():
        features[transform_square(square, player), 121] = 1
    if board.has_legal_en_passant():
        features[transform_square(board.ep_square, player), 127] = 1
    return EncodedPosition(features, state.history_complete, available)
