import chess
import chess.variant
import pytest

from chess_rl.adapter import ChessState, make_state


def test_initial_provenance_and_replay():
    assert make_state().history_complete
    assert not make_state(chess.STARTING_FEN).history_complete
    setup = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert make_state(setup, history_complete=True).history_complete
    state = make_state(moves_uci=["e2e4", "e7e5", "g1f3"])
    assert state.initial_fen == chess.STARTING_FEN
    assert state.board.fen() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    copied = state.copy()
    assert copied.board.move_stack == state.board.move_stack
    copied.board.pop()
    assert len(state.board.move_stack) == 3
    assert copied.history_complete
    assert not make_state(chess.STARTING_FEN, ["e2e4"]).history_complete


@pytest.mark.parametrize("fen", ["bad", "8/8/8/8/8/8/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w K - 0 1", "4k3/8/8/8/8/8/8/P3K3 w - - 0 1"])
def test_bad_fen(fen):
    with pytest.raises(ValueError):
        make_state(fen)


@pytest.mark.parametrize("moves", [["0000"], ["e2e5"], ["e2e4", "e2e3"], ["nonsense"], [None], "e2e4", None])
def test_bad_replay(moves):
    with pytest.raises(ValueError):
        make_state(moves_uci=moves)


def test_reject_variants_and_bad_provenance():
    for board in (chess.Board(chess960=True), chess.variant.AtomicBoard()):
        with pytest.raises(ValueError, match="standard"):
            ChessState(board, chess.STARTING_FEN, True)
    with pytest.raises(ValueError, match="history_complete"):
        make_state(history_complete=1)
    with pytest.raises(ValueError, match="initial_fen"):
        ChessState(chess.Board(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1", True)


def test_replay_continues_through_reached_claim():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    state = make_state(moves_uci=cycle * 3)
    assert len(state.board.move_stack) == 12
    assert state.board.is_repetition(3)


@pytest.mark.parametrize("uci", ["0000", "e2e5"])
def test_direct_state_rejects_illegal_history(uci):
    board = chess.Board()
    board.push(chess.Move.from_uci(uci))
    with pytest.raises(ValueError):
        ChessState(board, chess.STARTING_FEN, True)


def test_direct_state_rejects_unrecorded_board_edits():
    board = chess.Board()
    board.push_uci("e2e4")
    board.halfmove_clock = 42
    with pytest.raises(ValueError, match="retained legal history"):
        ChessState(board, chess.STARTING_FEN, True)
