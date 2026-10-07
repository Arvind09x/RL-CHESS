"""Public CPU/NumPy chess adapter. Consumers use this single implementation."""
from .encoding import encode_position
from .moves import id_to_move, legal_mask, move_to_id
from .outcomes import apply_move, terminal_result
from .results import EncodedPosition, Evaluation, TerminalResult
from .state import ChessState, make_state

__all__ = [
    "ChessState", "EncodedPosition", "Evaluation", "TerminalResult",
    "make_state", "encode_position", "move_to_id", "id_to_move", "legal_mask",
    "terminal_result", "apply_move",
]
