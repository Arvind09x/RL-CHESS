"""Actual-coordinate board ownership and history provenance."""
from dataclasses import dataclass
import copy
from numbers import Integral

import chess


@dataclass
class ChessState:
    board: chess.Board
    initial_fen: str
    history_complete: bool

    def __post_init__(self):
        if type(self.board) is not chess.Board or self.board.chess960:
            raise ValueError("Only standard chess.Board positions are supported")
        if type(self.history_complete) is not bool:
            raise ValueError("history_complete must be a boolean")
        if not self.board.is_valid():
            raise ValueError(f"Invalid chess board (status={self.board.status()})")
        try:
            initial = chess.Board(self.initial_fen)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid initial FEN") from exc
        if not initial.is_valid() or initial.fen(en_passant="fen") != self.board.root().fen(en_passant="fen"):
            raise ValueError("initial_fen must match the retained history's valid root")
        replay = self.board.root()
        for move in self.board.move_stack:
            require_legal_move(replay, move)
            replay.push(move)
        if (replay.fen(en_passant="fen") != self.board.fen(en_passant="fen")
                or replay.promoted != self.board.promoted):
            raise ValueError("Current board must match its retained legal history")
        self.initial_fen = initial.fen(en_passant="fen")
        self.board = self.board.copy(stack=True)
        # python-chess copies the stack list but shares mutable _BoardState
        # entries. Own these too, so later caller edits cannot alter provenance.
        self.board._stack = [copy.copy(item) for item in self.board._stack]

    def copy(self) -> "ChessState":
        """Independent board copy retaining every move and provenance."""
        return ChessState(self.board, self.initial_fen, self.history_complete)


def make_state(start_fen=None, moves_uci=(), *, history_complete=None) -> ChessState:
    """Replay a retained prefix. Explicit true provenance is a caller attestation.

    Replay checks chess legality, not the episode's draw-adjudication policy.
    An explicit FEN is incomplete by default, even if it looks like game start.
    """
    if start_fen is not None and not isinstance(start_fen, str):
        raise ValueError("start_fen must be a FEN string or None")
    if history_complete is not None and type(history_complete) is not bool:
        raise ValueError("history_complete must be a boolean or None")
    try:
        board = chess.Board(chess.STARTING_FEN if start_fen is None else start_fen)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed starting FEN: {start_fen!r}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid starting board (status={board.status()})")
    initial_fen = board.fen(en_passant="fen")
    if isinstance(moves_uci, (str, bytes)):
        raise ValueError("moves_uci must be a sequence of UCI strings")
    try:
        moves = iter(moves_uci)
    except TypeError as exc:
        raise ValueError("moves_uci must be iterable") from exc
    for ply, uci in enumerate(moves, 1):
        if not isinstance(uci, str):
            raise ValueError(f"Move at ply {ply} must be a UCI string")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"Malformed UCI at ply {ply}: {uci!r}") from exc
        try:
            require_legal_move(board, move)
        except ValueError as exc:
            raise ValueError(f"Illegal or null move at ply {ply}: {uci!r}") from exc
        board.push(move)
    complete = start_fen is None if history_complete is None else history_complete
    return ChessState(board, initial_fen, complete)


def require_legal_move(board: chess.Board, move: chess.Move) -> None:
    """Require exact standard legal UCI, excluding python-chess castling aliases."""
    if isinstance(move, chess.Move):
        fields = (move.from_square, move.to_square)
        if move.promotion is not None:
            fields += (move.promotion,)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in fields):
            raise ValueError("Move squares and promotion must be integer chess fields")
    if (not isinstance(move, chess.Move) or not move or move.drop is not None
            or not any(move == legal for legal in board.generate_legal_moves())):
        raise ValueError(f"Illegal or noncanonical standard chess move: {move!r}")
