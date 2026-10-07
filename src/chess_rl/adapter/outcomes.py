"""actual-occurrence-v1: reached claims only, after library outcomes."""
import chess
import numpy as np

from .moves import id_to_move
from .results import TerminalResult
from .state import ChessState


def terminal_result(state: ChessState) -> TerminalResult | None:
    # Rule queries may temporarily pop/push in python-chess. Isolate them from
    # the owned board so even an interrupted inspection cannot change the state.
    board = state.board.copy(stack=True)
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        if board.is_repetition(3):
            outcome = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
        elif board.is_fifty_moves():
            outcome = chess.Outcome(chess.Termination.FIFTY_MOVES, None)
        else:
            return None
    draw = outcome.winner is None
    wdl = np.array([0, 1, 0] if draw else [0, 0, 1], dtype=np.float32)
    return TerminalResult(outcome.winner, outcome.termination.name, wdl,
                          0.0 if draw else -1.0, state.history_complete)


def apply_move(state: ChessState, move_id: int) -> None:
    """Mutate only after termination and canonical legality checks succeed."""
    result = terminal_result(state)
    if result is not None:
        raise ValueError(f"Cannot apply a move after termination: {result.ending_reason}")
    move = id_to_move(state, move_id)
    state.board.push(move)
