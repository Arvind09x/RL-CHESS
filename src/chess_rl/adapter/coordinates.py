"""One involution for actual and network squares: preserve files for Black."""
from numbers import Integral

import chess


def transform_square(square: int, player: chess.Color) -> int:
    if isinstance(square, bool) or not isinstance(square, Integral) or not 0 <= square < 64:
        raise ValueError("square must be an integer in 0..63")
    if type(player) is not bool:
        raise ValueError("player must be White or Black")
    return int(square) if player == chess.WHITE else int(square) ^ 56
