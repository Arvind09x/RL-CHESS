from dataclasses import asdict
import json

import chess
import numpy as np
import pytest

from chess_rl.contracts import AdapterConfig
from chess_rl.adapter import ChessState, EncodedPosition, Evaluation, TerminalResult


def test_config_and_serialization(tmp_path):
    config = AdapterConfig()
    metadata = json.loads(json.dumps(asdict(config)))
    metadata["wdl_order"] = tuple(metadata["wdl_order"])
    assert AdapterConfig(**metadata) == config
    sample = EncodedPosition(np.arange(8192).reshape(64, 128), False, 3)
    path = tmp_path / "sample.npz"
    np.savez(path, features=sample.features, history_complete=sample.history_complete,
             available_snapshots=sample.available_snapshots, metadata=json.dumps(metadata))
    with np.load(path, allow_pickle=False) as stored:
        restored = EncodedPosition(stored["features"], bool(stored["history_complete"]), int(stored["available_snapshots"]))
        assert json.loads(str(stored["metadata"])) == json.loads(json.dumps(metadata))
    np.testing.assert_array_equal(restored.features, sample.features)
    assert restored.features.dtype == np.float32
    assert (restored.history_complete, restored.available_snapshots) == (False, 3)


@pytest.mark.parametrize("field,value", [("encoding_version", "old"), ("move_vocab_version", "old"),
    ("ending_policy", "claim-draw"), ("wdl_order", ("loss", "draw", "win"))])
def test_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        AdapterConfig(**{field: value})


def test_results_validation():
    priors = np.zeros(4352, np.float32)
    priors[796] = 1
    result = Evaluation(priors, [0.6, 0.1, 0.3], 0.3)
    assert result.priors.shape == (4352,)
    assert result.wdl.dtype == np.float32
    terminal = TerminalResult(chess.BLACK, "CHECKMATE", [0, 0, 1], -1, True)
    assert terminal.priors is None
    with pytest.raises(ValueError):
        Evaluation(priors, [0.6, 0.1, 0.3], -0.3)
    with pytest.raises(ValueError):
        Evaluation(np.zeros(4352), [0, 1, 0], 0)
    with pytest.raises(ValueError):
        EncodedPosition(np.full((64, 128), np.nan), False, 1)
    with pytest.raises(ValueError):
        EncodedPosition(np.zeros((128, 64)), False, 1)
    with pytest.raises(ValueError):
        EncodedPosition(np.zeros((64, 128)), False, 9)


def test_state_owns_board():
    board = chess.Board()
    state = ChessState(board, chess.STARTING_FEN, True)
    board.push_uci("e2e4")
    assert not state.board.move_stack
