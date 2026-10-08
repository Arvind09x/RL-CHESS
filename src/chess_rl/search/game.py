"""Bounded CPU smoke games; no learning, opponent protocol or resume loop."""
from dataclasses import asdict, dataclass

from chess_rl.adapter import ChessState, apply_move, terminal_result
from .evaluator import evaluate
from .puct import Evaluator, Node, SearchConfig
from .runner import SearchFailure, search


@dataclass(frozen=True)
class SearchAttempt:
    """One fresh root: counts include successful backups even on interruption.

    Selected actions describe search; only moves_uci in GameRecord describes
    actual application. Sparse root edges store (ID, N, W) in parent view.
    """
    ply: int
    requested_simulations: int
    completed_simulations: int
    status: str
    move_id: int | None
    move_uci: str | None
    visited_edges: tuple[tuple[int, int, float], ...]
    error_type: str | None = None
    error_message: str | None = None
    failure_stage: str | None = None


@dataclass(frozen=True)
class GameRecord:
    initial_fen: str
    opening_prefix: tuple[str, ...]
    moves_uci: tuple[str, ...]
    history_complete: bool
    final_fen: str
    final_promoted: int
    simulation_budget: int
    ply_cap: int
    search_config: SearchConfig
    evaluator_id: str
    attempts: tuple[SearchAttempt, ...]
    status: str
    ending_reason: str
    outcome: str | None
    winner: str | None
    error_type: str | None = None
    error_message: str | None = None
    ending_policy: str = "actual-occurrence-v1"
    move_vocabulary: str = "pair4352-v1"
    selection: str = "evaluation-max-visits-smallest-id"
    exploration_noise: bool = False
    rng: None = None

    def to_dict(self) -> dict:
        """JSON-ready snapshot. No unfinished game gets an outcome label."""
        record = asdict(self)
        record["search_config"] = {"c_puct": float(self.search_config.c_puct),
                                   "unvisited_value": float(self.search_config.unvisited_value)}
        record["schema"] = "s05-game-v1"
        return record


def _attempt(root, ply, budget, status, result=None, error=None):
    # Every call owns a fresh root: outgoing completed N is this call's count.
    edges = () if root is None else tuple(
        (i, e.visit_count, e.value_sum) for i, e in sorted(root.edges.items())
        if e.visit_count
    )
    cause = error.__cause__ if isinstance(error, SearchFailure) else error
    return SearchAttempt(
        ply, budget, sum(n for _, n, _ in edges), status,
        None if result is None else result.move_id,
        None if result is None or result.move is None else result.move.uci(),
        edges, None if cause is None else type(cause).__name__,
        None if cause is None else str(cause),
        error.stage if isinstance(error, SearchFailure) else None,
    )


def run_game(starting_state: ChessState, simulation_budget: int, *, ply_cap: int,
             config: SearchConfig = SearchConfig(), evaluator: Evaluator = evaluate,
             evaluator_id: str = "s01-uniform-dummy-v1") -> GameRecord:
    """Run from an owned full-history copy, with deterministic S04 selection.

    The supplied state's retained moves are the explicit opening prefix.
    ply_cap counts ALL retained plies from initial_fen, including that prefix;
    a prefix already at/above the cap is adjudicated before truncation.
    Each move uses a fresh Node and exactly simulation_budget completions.
    Terminal outcomes are checked before search and before every cap check.
    KeyboardInterrupt/SystemExit become interrupted records; ordinary runtime
    errors become failed records. Invalid arguments raise before execution.
    Full history survives every returned status. This is not a resumable
    learning episode, checkpoint/target store or process-crash journal.
    Custom evaluators must honor the accepted read-only batch-of-one contract;
    callers must give them a descriptive evaluator_id.
    """
    if not isinstance(starting_state, ChessState):
        raise ValueError("starting_state must be ChessState")
    if type(simulation_budget) is not int or simulation_budget <= 0:
        raise ValueError("simulation_budget must be a positive integer")
    if type(ply_cap) is not int or ply_cap < 0:
        raise ValueError("ply_cap must be a nonnegative integer")
    if not isinstance(config, SearchConfig) or not callable(evaluator):
        raise ValueError("config must be SearchConfig and evaluator must be callable")
    if not isinstance(evaluator_id, str) or not evaluator_id.strip():
        raise ValueError("evaluator_id must be a nonempty string")
    if evaluator is not evaluate and evaluator_id == "s01-uniform-dummy-v1":
        raise ValueError("Custom evaluator requires an explicit evaluator_id")
    state = starting_state.copy()
    opening = tuple(m.uci() for m in state.board.move_stack)
    attempts = []

    def finish(status, reason, terminal=None, error=None):
        winner = None if terminal is None or terminal.winner is None else (
            "white" if terminal.winner else "black")
        outcome = None if terminal is None else (
            "1/2-1/2" if winner is None else "1-0" if winner == "white" else "0-1")
        cause = error.__cause__ if isinstance(error, SearchFailure) else error
        return GameRecord(
            state.initial_fen, opening, tuple(m.uci() for m in state.board.move_stack),
            state.history_complete, state.board.fen(en_passant="fen"),
            int(state.board.promoted), simulation_budget, ply_cap, config,
            evaluator_id, tuple(attempts), status, reason, outcome, winner,
            None if cause is None else type(cause).__name__,
            None if cause is None else str(cause),
        )

    try:
        while True:
            terminal = terminal_result(state)
            if terminal is not None:
                return finish("completed", terminal.ending_reason, terminal)
            ply = len(state.board.move_stack)
            if ply >= ply_cap:
                return finish("truncated", "ply_limit")
            root = None
            try:
                root = Node(state)
                result = search(root, simulation_budget, config=config,
                                evaluator=evaluator, exploration=None)
            except (KeyboardInterrupt, SystemExit) as exc:
                attempts.append(_attempt(root, ply, simulation_budget, "interrupted", error=exc))
                raise
            except Exception as exc:
                attempts.append(_attempt(root, ply, simulation_budget, "failed", error=exc))
                raise
            attempts.append(_attempt(root, ply, simulation_budget, "completed", result))
            if result.completed_simulations != simulation_budget or result.move_id is None:
                raise RuntimeError("Nonterminal game search must complete its budget and select a move")
            # Publish only a successfully applied owned copy. Caller and the
            # current game history survive an interrupted/failed application.
            next_state = state.copy()
            apply_move(next_state, result.move_id)
            state = next_state
    except (KeyboardInterrupt, SystemExit) as exc:
        return finish("interrupted", "execution_interrupted", error=exc)
    except Exception as exc:
        return finish("failed", "execution_failed", error=exc)
