"""Injectable E01 players. No checkpoint loader, Torch or annotation teacher."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Protocol

import chess

from chess_rl.adapter import ChessState, apply_move, move_to_id, terminal_result
from chess_rl.search.game import SearchAttempt
from chess_rl.search.puct import Evaluator, Node, SearchConfig
from chess_rl.search.runner import SearchFailure, search


def snapshot(value):
    """Independent, finite JSON metadata; callers cannot mutate saved records."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def text_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Checkpoint:
    """Byte identity only, never proof of training or model compatibility.

    T02/T04 will supply a verified loader. from_file hashes actual bytes;
    fixture artifacts must be labelled fixture, not trained.
    """
    checkpoint_id: str
    path: str
    sha256: str
    kind: str

    def __post_init__(self):
        text_id(self.checkpoint_id, "checkpoint_id")
        text_id(self.path, "checkpoint path")
        if self.kind not in ("fixture", "pretrained-P", "rl-R"):
            raise ValueError("Unknown checkpoint kind")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("checkpoint sha256 must be a lowercase SHA-256")

    @classmethod
    def from_file(cls, checkpoint_id, path, *, kind):
        path = Path(path).resolve()
        return cls(checkpoint_id, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), kind)

    def verify(self):
        if hashlib.sha256(Path(self.path).read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Checkpoint bytes differ from recorded identity")


@dataclass(frozen=True)
class Decision:
    move_id: int
    details: dict


class Player(Protocol):
    """One decision on an owned state copy with an explicit per-turn seed.

    metadata() returns finite JSON including player_id/kind/version/settings.
    choose must be read-only, return a legal shared ID and reset any per-game
    mutable state itself. Stateless choices make fault/reorder replay stable.
    Evaluators passed to SearchPlayer use the accepted batched Evaluator type.
    """
    def metadata(self) -> dict: ...
    def choose(self, state: ChessState, *, seed: int) -> Decision: ...


class DecisionFailure(Exception):
    def __init__(self, cause, details):
        super().__init__(str(cause))
        self.cause, self.details = cause, snapshot(details)


@dataclass(frozen=True)
class SearchSettings:
    simulation_budget: int = 16
    config: SearchConfig = SearchConfig()

    def __post_init__(self):
        integer(self.simulation_budget, "simulation_budget", 1)
        if not isinstance(self.config, SearchConfig):
            raise ValueError("config must be SearchConfig")

    def to_dict(self):
        return dict(simulation_budget=self.simulation_budget, c_puct=float(self.config.c_puct),
                    unvisited_value=float(self.config.unvisited_value),
                    exploration_noise=False, exploration=None,
                    selection="evaluation-max-visits-smallest-id", fresh_root=True)


@dataclass(frozen=True)
class SearchPlayer:
    player_id: str
    role: str
    evaluator_id: str
    evaluator: Evaluator
    settings: SearchSettings = SearchSettings()
    checkpoint: Checkpoint | None = None

    def __post_init__(self):
        text_id(self.player_id, "player_id")
        text_id(self.evaluator_id, "evaluator_id")
        if self.role not in ("P", "R", "fixture") or not callable(self.evaluator):
            raise ValueError("Explicit role and callable evaluator required")
        if not isinstance(self.settings, SearchSettings):
            raise ValueError("settings must be SearchSettings")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, Checkpoint):
                raise ValueError("checkpoint must be Checkpoint or None")
            expected = {"P": "pretrained-P", "R": "rl-R"}.get(self.role)
            if self.checkpoint.kind != "fixture" and self.checkpoint.kind != expected:
                raise ValueError("Checkpoint kind does not match player role")

    def metadata(self):
        return dict(player_id=self.player_id, kind="search", version="e01-search-v1",
                    role=self.role, evaluator_id=self.evaluator_id,
                    checkpoint=None if self.checkpoint is None else asdict(self.checkpoint),
                    checkpoint_status="unavailable" if self.checkpoint is None else self.checkpoint.kind,
                    settings=self.settings.to_dict())

    def choose(self, state, *, seed):
        root = Node(state)
        def details(status, result=None, error=None):
            edges = [(i, e.visit_count, e.value_sum) for i, e in sorted(root.edges.items()) if e.visit_count]
            cause = error.__cause__ if isinstance(error, SearchFailure) else error
            trace = SearchAttempt(len(state.board.move_stack), self.settings.simulation_budget,
                sum(n for _, n, _ in edges), status, None if result is None else result.move_id,
                None if result is None or result.move is None else result.move.uci(), tuple(edges),
                None if cause is None else type(cause).__name__,
                None if cause is None else str(cause), error.stage if isinstance(error, SearchFailure) else None)
            return dict(**asdict(trace), search_status=status, selected_id=trace.move_id)
        try:
            result = search(root, self.settings.simulation_budget, config=self.settings.config,
                            evaluator=self.evaluator, exploration=None)
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            cause = exc.__cause__ if isinstance(exc, SearchFailure) else exc
            raise DecisionFailure(cause or exc, details("interrupted" if isinstance(
                exc, (KeyboardInterrupt, SystemExit)) else "failed", error=exc)) from exc
        if result.move_id is None or result.completed_simulations != self.settings.simulation_budget:
            raise DecisionFailure(RuntimeError("Incomplete nonterminal search"), details("failed", result))
        return Decision(result.move_id, details("completed", result))


@dataclass(frozen=True)
class RandomLegalPlayer:
    player_id: str = "random-legal"

    def metadata(self):
        return dict(player_id=self.player_id, kind="random-legal", version="e01-random-v1",
                    settings=dict(order="sorted-shared-id", rng="python-random-MT19937",
                                  seed="per-turn-sha256-v1", distribution="uniform-legal"))

    def choose(self, state, *, seed):
        ids = sorted(move_to_id(state, m) for m in state.board.legal_moves)
        return Decision(random.Random(seed).choice(ids), {})


# King has no material contribution; legal chess prevents king captures.
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True)
class GreedyPlayer:
    player_id: str = "material-greedy"

    def metadata(self):
        return dict(player_id=self.player_id, kind="material-greedy", version="e01-greedy-v1",
                    settings=dict(piece_values={chess.piece_name(k): v for k, v in PIECE_VALUES.items()},
                        perspective="moving-player: own-minus-opponent after candidate move",
                        terminal_scores="win=+1000, draw=0, loss=-1000 before material",
                        ties="smallest-shared-id", depth=1, rng=None))

    def choose(self, state, *, seed):
        scores = []
        mover = state.board.turn
        for move in state.board.legal_moves:
            move_id = move_to_id(state, move)
            child = state.copy()
            apply_move(child, move_id)
            terminal = terminal_result(child)
            if terminal is not None:
                value = 0 if terminal.winner is None else 1000 if terminal.winner == mover else -1000
            else:
                value = sum(v * (len(child.board.pieces(p, mover)) - len(child.board.pieces(p, not mover)))
                            for p, v in PIECE_VALUES.items())
            scores.append((value, -move_id))
        value, negative_id = max(scores)
        return Decision(-negative_id, dict(score=value, perspective="moving-player"))


@dataclass(frozen=True)
class StockfishPin:
    executable: str
    sha256: str
    uci_name: str
    release: str
    source_commit: str

    def __post_init__(self):
        for name in ("executable", "uci_name", "release", "source_commit"):
            text_id(getattr(self, name), name)
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Engine requires an explicit SHA-256 pin")

    def verify(self):
        if hashlib.sha256(Path(self.executable).read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Evaluation engine checksum mismatch")


@dataclass(frozen=True)
class StockfishPlayer:
    pin: StockfishPin
    nodes: int = 128
    player_id: str = "stockfish-small"
    timeout: float = 10.0

    def __post_init__(self):
        if not isinstance(self.pin, StockfishPin):
            raise ValueError("Explicit StockfishPin required")
        integer(self.nodes, "nodes", 1)
        if self.nodes > 10000:
            raise ValueError("E01 diagnostic node budgets must be <=10000")
        if type(self.timeout) not in (int, float) or not 0 < self.timeout <= 60:
            raise ValueError("timeout must be in (0,60]")

    def metadata(self):
        return dict(player_id=self.player_id, kind="stockfish", version="e01-stockfish-v1",
                    engine=asdict(self.pin), settings=dict(nodes=self.nodes, timeout_seconds=self.timeout,
                        options=dict(Threads=1, Hash=16, MultiPV=1, SkillLevel=20,
                            UCI_LimitStrength=False, UCI_ShowWDL=False, SyzygyPath="",
                            Ponder=False, UCI_Chess960=False),
                        hash_reset="fresh-process-and-Clear-Hash-per-move",
                        tie_handling="pinned-engine-bestmove; no external RNG",
                        configuration_namespace="evaluation-opponent; independent of annotation-teacher"))

    def choose(self, state, *, seed):
        # Lazy import: controlled/random/greedy games do not need an engine.
        import chess.engine
        self.pin.verify()  # Verify before every execution, not just construction.
        options = dict(Threads=1, Hash=16, **{"Skill Level": 20},
                       UCI_LimitStrength=False, UCI_ShowWDL=False, SyzygyPath="")
        trace = dict(requested_nodes=self.nodes, hash_reset_completed=False)
        try:
            with tempfile.TemporaryDirectory(prefix="e01-engine-") as directory, \
                    chess.engine.SimpleEngine.popen_uci(self.pin.executable, timeout=self.timeout, cwd=directory) as engine:
                trace["engine_id"] = dict(engine.id)
                if engine.id.get("name") != self.pin.uci_name:
                    raise ValueError("Evaluation engine UCI name mismatch")
                for k, v in options.items():
                    if engine.options[k].parse(v) != v:
                        raise ValueError(f"Unsupported engine option {k}")
                for k, v in dict(MultiPV=1, Ponder=False, UCI_Chess960=False).items():
                    if engine.options[k].parse(v) != v:
                        raise ValueError(f"Unsupported managed option {k}")
                if engine.options["Clear Hash"].type != "button":
                    raise ValueError("Missing Clear Hash")
                trace["effective_options"] = {k: v.default for k, v in engine.options.items() if v.type != "button"}
                trace["effective_options"].update(options, MultiPV=1, Ponder=False, UCI_Chess960=False)
                engine.configure(options)
                engine.configure({"Clear Hash": None})
                engine.ping()
                trace["hash_reset_completed"] = True
                result = engine.play(state.board.copy(stack=True), chess.engine.Limit(nodes=self.nodes),
                                     game=object(), info=chess.engine.INFO_ALL, ponder=False)
                # Preserve measured work; UCI node limits can overshoot. No WDL target.
                trace["returned_info"] = {k: result.info[k] for k in
                    ("nodes", "depth", "seldepth", "time", "nps", "hashfull", "tbhits") if k in result.info}
                trace["returned_bestmove"] = None if result.move is None else result.move.uci()
                if result.resigned or result.move is None:
                    raise ValueError("No completed legal engine bestmove")
                if type(result.info.get("nodes")) is not int or result.info["nodes"] <= 0:
                    raise ValueError("Missing engine node accounting")
                if result.info.get("tbhits", 0) != 0:
                    raise ValueError("Unexpected evaluation tablebase hits")
                move_id = move_to_id(state, result.move)
            return Decision(move_id, trace)
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            raise DecisionFailure(exc, trace) from exc
