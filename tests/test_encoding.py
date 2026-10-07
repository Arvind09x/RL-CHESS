import chess
import numpy as np
import pytest

from chess_rl.adapter import make_state, encode_position


@pytest.mark.parametrize("color", ["w", "b"])
def test_every_start_channel_group(color):
    fen = chess.STARTING_FEN.replace(" w ", f" {color} ")
    encoded = encode_position(make_state(fen, history_complete=True))
    x = encoded.features
    assert x.dtype == np.float32 and x.shape == (64, 128) and np.isfinite(x).all()
    assert encoded.history_complete and encoded.available_snapshots == 1
    # Start is symmetric under rank reflection with ownership exchanged.
    expected_squares = [[8+i for i in range(8)], [1, 6], [2, 5], [0, 7], [3], [4],
                        [48+i for i in range(8)], [57, 62], [58, 61], [56, 63], [59], [60]]
    for t in range(8):
        for channel, squares in enumerate(expected_squares):
            expected = np.zeros(64)
            expected[squares] = 1
            np.testing.assert_array_equal(x[:, 13*t+channel], expected)
        assert not x[:, 13*t+12].any()
    np.testing.assert_array_equal(x[:, 104:108], np.ones((64, 4)))
    assert (x[:, 108] == (color == "b")).all()
    assert not x[:, 109:111].any()
    assert (x[:, 111] == 1).all()
    assert set(np.flatnonzero(x[:, 112])) == set(range(16))
    assert set(np.flatnonzero(x[:, 113])) == set(range(48, 64))
    np.testing.assert_array_equal(x[:, 114], [(r+f) % 2 for r in range(8) for f in range(8)])
    assert x[chess.A1, 114] == 0 and x[chess.B1, 114] == 1
    assert not x[:, 115:122].any()
    np.testing.assert_array_equal(x[:, 122:127], np.tile([1, .25, .25, .25, .125], (64, 1)))
    assert not x[:, 127].any()


@pytest.mark.parametrize("color,expected_rights", [("w", [1,0,0,1]), ("b", [0,1,1,0])])
def test_castling_order_and_unclipped_clock(color, expected_rights):
    state = make_state(f"r3k2r/8/8/8/8/8/8/R3K2R {color} Qk - 151 1")
    x = encode_position(state).features
    np.testing.assert_array_equal(x[0, 104:108], expected_rights)
    assert (x[:, 109] == np.float32(1.51)).all()


@pytest.mark.parametrize("color,sign,count", [("w", 1, 9), ("b", -1, 0)])
def test_promoted_material_is_not_clipped(color, sign, count):
    state = make_state(f"7k/8/8/8/8/8/NNNNNNNN/NK6 {color} - - 0 1")
    x = encode_position(state).features
    assert (x[:, 116] == sign * 9/8).all()
    assert (x[:, 123] == count/8).all()


@pytest.mark.parametrize("color", ["w", "b"])
def test_opposite_bishops_with_other_material(color):
    x = encode_position(make_state(f"4k3/p7/8/8/8/8/P7/1B2K1bR {color} - - 37 1")).features
    assert (x[:, 120] == 1).all()
    assert x[:, 112].sum() == (4 if color == "w" else 3)
    assert x[:, 113].sum() == (3 if color == "w" else 4)
    assert x[0, 118] == (1/8 if color == "w" else -1/8)
    assert x[0, 125] == (1/8 if color == "w" else 0)
    assert x[0, 109] == np.float32(.37)


@pytest.mark.parametrize("fen", [
    "4k3/8/8/8/8/8/8/1B2K2b w - - 0 1",  # Same colors.
    "4k3/8/8/8/8/8/8/1BB1K1b1 w - - 0 1", # Two bishops on one side.
    "4k3/8/8/8/8/8/8/4K1b1 w - - 0 1",   # No bishop on one side.
])
def test_bishop_condition_is_exact(fen):
    assert not encode_position(make_state(fen)).features[:, 120].any()


@pytest.mark.parametrize("fen,target", [
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", chess.D6),
    ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", chess.E6),
    ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", None), # Pinned pawn.
    ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1", None), # No capturer.
])
def test_legal_en_passant_mask(fen, target):
    x = encode_position(make_state(fen)).features
    assert list(np.flatnonzero(x[:, 127])) == ([] if target is None else [target])


@pytest.mark.parametrize("fen,target", [
    ("k3r3/8/8/8/8/8/8/4K3 w - - 0 1", chess.E8),
    ("4k3/8/8/8/8/8/8/K3R3 b - - 0 1", chess.E8),
])
def test_checkers_reflected_for_current_player(fen, target):
    x = encode_position(make_state(fen)).features
    assert list(np.flatnonzero(x[:, 121])) == [target]


def test_encoded_provenance_and_input_preservation():
    state = make_state(chess.STARTING_FEN, ["e2e4", "e7e5"])
    before = state.board.fen(en_passant="fen"), list(state.board.move_stack), state.initial_fen, state.history_complete
    encoded = encode_position(state)
    assert not encoded.history_complete and encoded.available_snapshots == 3
    encoded.features[:] = 99  # Returned arrays cannot corrupt rule state.
    assert (state.board.fen(en_passant="fen"), state.board.move_stack, state.initial_fen, state.history_complete) == before
