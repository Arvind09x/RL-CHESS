import chess
import numpy as np
import pytest

from chess_rl.adapter import make_state, legal_mask, id_to_move, move_to_id
from .helpers import CASTLING, PROMOTION, BLACK_PROMOTION


@pytest.mark.parametrize("fen", [None, CASTLING.format("w"), CASTLING.format("b"), PROMOTION, BLACK_PROMOTION,
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"])
def test_mask_matches_every_decodable_slot(fen):
    state = make_state(fen)
    mask = legal_mask(state)
    assert mask.shape == (4352,) and mask.dtype == np.bool_
    moves = list(state.board.legal_moves)
    ids = [move_to_id(state, move) for move in moves]
    assert mask.sum() == len(moves) == len(set(ids))
    assert set(np.flatnonzero(mask)) == set(ids)
    assert {id_to_move(state, int(i)) for i in np.flatnonzero(mask)} == set(moves)
    for i in np.flatnonzero(~mask):
        with pytest.raises(ValueError):
            id_to_move(state, int(i))
    for origin in range(8):
        for destination in range(8):
            if abs(origin - destination) > 1:
                assert not mask[4096 + 4 * (8 * origin + destination):4096 + 4 * (8 * origin + destination) + 4].any()
    if fen in [PROMOTION, BLACK_PROMOTION]:
        assert not mask[52 * 64 + 60]


def test_chess_legality_in_drawn_position():
    state = make_state("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert state.board.is_insufficient_material()
    assert legal_mask(state).sum() == 5
