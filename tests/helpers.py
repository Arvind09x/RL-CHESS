"""Shared hand-written acceptance positions and preservation fingerprint."""
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R {} KQkq - 17 1"
PROMOTION = "3r1r1k/4P3/8/8/8/8/8/K7 w - - 0 1"
BLACK_PROMOTION = "k7/8/8/8/8/8/4p3/3R1R1K b - - 0 1"
CYCLE = ["g1f3", "g8f6", "f3g1", "f6g8"]
LONG_CYCLE = ["g1f3", "g8f6", "f3h4", "f6h5", "h4f3", "h5f6", "f3g1", "f6g8"]


def fingerprint(state):
    """Every retained prefix, counters, raw EP, promoted bits and provenance."""
    board = state.board.copy(stack=True)
    prefixes = []
    while True:
        prefixes.append((board.fen(en_passant="fen"), board.promoted))
        if not board.move_stack:
            break
        board.pop()
    return state.initial_fen, state.history_complete, tuple(m.uci() for m in state.board.move_stack), prefixes
