"""Newest-first placement/repetition blocks from retained real prefixes."""
import chess
import numpy as np

from .coordinates import transform_square
from .state import ChessState


def history_blocks(state: ChessState) -> tuple[np.ndarray, int]:
    """Copy full rule history before walking back; padding adds no occurrences."""
    board = state.board.copy(stack=True)
    player = board.turn
    blocks = []
    while len(blocks) < 8:
        block = np.zeros((64, 13), dtype=np.float32)
        for offset, color in ((0, player), (6, not player)):
            for piece_type in chess.PIECE_TYPES:
                for square in board.pieces(piece_type, color):
                    block[transform_square(square, player), offset + piece_type - 1] = 1
        block[:, 12] = board.is_repetition(2)
        blocks.append(block)
        if not board.move_stack:
            break
        board.pop()
    available = len(blocks)
    blocks.extend([blocks[-1]] * (8 - available))
    return np.concatenate(blocks, axis=1), available
