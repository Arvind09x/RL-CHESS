import json
import random
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_rl.adapter import apply_move, encode_position, id_to_move, legal_mask, make_state, move_to_id, terminal_result
from .helpers import CASTLING, CYCLE, LONG_CYCLE, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_replay_in_two_fresh_processes():
    command = [sys.executable, "scripts/check_adapter.py"]
    first = subprocess.check_output(command, cwd=ROOT, text=True)
    second = subprocess.check_output(command, cwd=ROOT, text=True)
    assert first == second
    rows = json.loads(first)
    assert len(rows) == 13
    assert max(row["retained_plies"] for row in rows) == 20


def test_full_repetition_history_exceeds_network_window():
    state = make_state(moves_uci=LONG_CYCLE * 2)
    assert len(state.board.move_stack) == 16
    assert terminal_result(state).ending_reason == "THREEFOLD_REPETITION"
    x = encode_position(state).features
    assert x[0, 12] == 1
    assert not any(np.array_equal(x[:, :12], x[:, 13*t:13*t+12]) for t in range(1, 8))
    fen_only = make_state(state.board.fen(en_passant="fen"))
    assert terminal_result(fen_only) is None
    assert not encode_position(fen_only).features[:, 12::13][:, :8].any()
    assert not fen_only.history_complete


def test_repetition_equivalence_ep_rights_and_counters():
    # A raw EP square with no legal capturer does not distinguish rule positions.
    state = make_state(moves_uci=["e2e4"] + ["g8f6", "g1f3", "f6g8", "f3g1"])
    assert state.board.is_repetition(2)
    assert encode_position(state).features[0, 12] == 1
    # Available legal EP at the root DOES distinguish otherwise equal pieces.
    fen = "4k1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 1"
    state = make_state(fen, CYCLE)
    assert not state.board.is_repetition(2)
    assert encode_position(state).features[0, 12] == 0
    # Lost castling rights distinguish positions even after pieces return.
    state = make_state(CASTLING.format("w"), ["h1h2", "h8h7", "h2h1", "h7h8"])
    assert not state.board.is_repetition(2)
    assert encode_position(state).features[0, 12] == 0


@pytest.mark.parametrize("fen,moves", [(None, CYCLE*2), (None, ["e2e4", "d7d5", "e4d5"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", []),
    ("k7/8/8/8/8/8/4p3/3R1R1K b - - 0 1", [])])
def test_all_inspection_and_failed_operations_preserve_history(fen, moves):
    state = make_state(fen, moves)
    before = fingerprint(state)
    encode_position(state)
    mask = legal_mask(state)
    terminal_result(state)
    for move_id in np.flatnonzero(mask):
        move = id_to_move(state, int(move_id))
        assert move_to_id(state, move) == move_id
    for bad in [-1, 4352, True, 1.5]:
        with pytest.raises(ValueError):
            apply_move(state, bad)
    assert fingerprint(state) == before
    clone = state.copy()
    assert fingerprint(clone) == before
    if clone.board.move_stack:
        clone.board.pop()
        assert fingerprint(state) == before


@pytest.mark.parametrize("seed", [7, 23, 101])
def test_deterministic_legal_playouts(seed):
    rng = random.Random(seed)
    state = make_state()
    for _ in range(100):
        before = fingerprint(state)
        result = terminal_result(state)
        x = encode_position(state)
        mask = legal_mask(state)
        moves = list(state.board.legal_moves)
        assert x.available_snapshots == min(8, len(state.board.move_stack)+1)
        assert x.history_complete
        assert mask.sum() == len(moves)
        assert len(set(move_to_id(state, move) for move in moves)) == len(moves)
        assert fingerprint(state) == before
        if result is not None:
            break
        move = rng.choice(moves)
        move_id = move_to_id(state, move)
        assert id_to_move(state, move_id) == move
        apply_move(state, move_id)
    replay = make_state(state.initial_fen, [m.uci() for m in state.board.move_stack], history_complete=True)
    assert fingerprint(replay) == fingerprint(state)
    np.testing.assert_array_equal(encode_position(replay).features, encode_position(state).features)
    np.testing.assert_array_equal(legal_mask(replay), legal_mask(state))


def test_import_boundary_is_cpu_only():
    command = [sys.executable, "-c", "import sys; import chess_rl.adapter; assert not any(n in sys.modules for n in ('torch', 'chess.engine', 'chess_rl.search')); print('CPU-only imports OK')"]
    assert subprocess.check_output(command, cwd=ROOT, text=True).strip() == "CPU-only imports OK"
