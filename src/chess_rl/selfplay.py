"""R01 sequential model-guided games with durable move-boundary resumption.

Only this orchestration/record layer is new. Rules, history, inference,
exploration, visit policy and checkpoint loading retain their shared owners.
One writer per game directory; run() holds a nonblocking advisory process lock.
"""
from dataclasses import asdict, dataclass
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from chess_rl.adapter import apply_move, id_to_move, legal_mask, terminal_result
from chess_rl.contracts import AdapterConfig, MOVE_COUNT
from chess_rl.evaluation.matches import Opening, digest, state_trace
from chess_rl.search.puct import Node, SearchConfig
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.runner import SearchFailure, TrainingExploration, search


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Phase:
    """Settings take effect at start_ply, counting the complete opening prefix."""
    start_ply: int
    temperature: float
    noise_fraction: float
    dirichlet_alpha: float = 0.3

    def __post_init__(self):
        _integer(self.start_ply, "start_ply")
        TrainingExploration(self.temperature, self.noise_fraction, self.dirichlet_alpha)

    def exploration(self):
        return TrainingExploration(self.temperature, self.noise_fraction, self.dirichlet_alpha)


@dataclass(frozen=True)
class Settings:
    simulation_budget: int = 16
    ply_cap: int = 512
    c_puct: float = 1.5
    unvisited_value: float = 0.0
    phases: tuple[Phase, ...] = (Phase(0, 1.0, 0.25), Phase(30, 0.0, 0.25))

    def __post_init__(self):
        _integer(self.simulation_budget, "simulation_budget", 1)
        _integer(self.ply_cap, "ply_cap")
        SearchConfig(self.c_puct, self.unvisited_value)
        phases = tuple(self.phases)
        if (not phases or any(not isinstance(p, Phase) for p in phases)
                or phases[0].start_ply != 0
                or any(a.start_ply >= b.start_ply for a, b in zip(phases, phases[1:]))):
            raise ValueError("phases must start at zero and strictly increase")
        object.__setattr__(self, "phases", phases)

    def exploration(self, ply):
        return next(p.exploration() for p in reversed(self.phases) if ply >= p.start_ply)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["phases"] = tuple(Phase(**p) for p in value["phases"])
        return cls(**value)


def runtime():
    """Exact continuation is limited to the same recorded deterministic CPU runtime."""
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), numpy=np.__version__, torch=str(torch.__version__),
                chess=__import__("chess").__version__, device="cpu", batch_size=1,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                mkldnn_enabled=torch.backends.mkldnn.enabled,
                float32_matmul_precision=torch.get_float32_matmul_precision())


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _write(path, value, *, replace=False):
    """Flush file + atomic publication + directory fsync; never overwrite a freeze."""
    path = Path(path)
    payload = _json(dict(sha256=digest(value), content=value))
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read(path):
    envelope = json.loads(Path(path).read_bytes())
    if set(envelope) != {"sha256", "content"} or digest(envelope["content"]) != envelope["sha256"]:
        raise ValueError("record checksum mismatch")
    return envelope["content"]


def _checkpoint(path, evaluator):
    checkpoint = evaluator.checkpoint
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                bytes=Path(path).stat().st_size, checkpoint_id=checkpoint.checkpoint_id,
                model_config=asdict(checkpoint.config), contracts=checkpoint.contracts,
                provenance=checkpoint.provenance, role="plumbing; no pretrained P selection")


def _rng(value):
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = copy.deepcopy(value)
    return rng


def _adjudicate(record, state, cap):
    terminal = terminal_result(state)
    record.update(state_trace(state), winner=None, outcome=None)
    if terminal is not None:
        winner = None if terminal.winner is None else "white" if terminal.winner else "black"
        record.update(status="completed", ending_reason=terminal.ending_reason, winner=winner,
                      outcome="1/2-1/2" if winner is None else "1-0" if winner == "white" else "0-1")
    elif len(state.board.move_stack) >= cap:
        record.update(status="truncated", ending_reason="ply_limit")
    else:
        # Every recoverable active boundary is explicitly unfinished. A hard
        # process kill therefore leaves an interrupted record, never a draw.
        record.update(status="interrupted", ending_reason="execution_interrupted")


def audit(configuration, record):
    """Replay durable history/statistics without loading weights or generating targets."""
    if configuration["schema"] != "r01-config-v1" or record["schema"] != "r01-game-v1":
        raise ValueError("unsupported R01 schema")
    AdapterConfig(**{**configuration["contracts"],
                     "wdl_order": tuple(configuration["contracts"]["wdl_order"])})
    if record["configuration_sha256"] != digest(configuration):
        raise ValueError("configuration changed")
    settings = Settings.from_dict(configuration["settings"])
    opening_dict = configuration["opening"]
    opening = Opening(**{k: opening_dict[k] for k in (
        "opening_id", "initial_fen", "moves_uci", "history_complete", "source")})
    if opening.to_dict()["replay_sha256"] != opening_dict["replay_sha256"] or not opening.history_complete:
        raise ValueError("complete opening history required")
    _integer(configuration["seed"], "seed")
    _integer(configuration["opening_selection_seed"], "opening_selection_seed")
    expected_rng = np.random.Generator(np.random.PCG64(configuration["seed"])).bit_generator.state
    if configuration["rng"] != dict(algorithm="numpy-PCG64", seed=configuration["seed"],
        initial_state=expected_rng, python_global="unused", numpy_global="unused",
        torch_cpu="unused by eval/inference; dropout disabled"):
        raise ValueError("unsupported RNG configuration")
    state = opening.state()
    prefix = list(opening.moves_uci)
    if record["moves_uci"][:len(prefix)] != prefix:
        raise ValueError("opening prefix changed")
    if len(record["positions"]) != len(record["moves_uci"]) - len(prefix):
        raise ValueError("one position per committed generated move required")
    for position, uci in zip(record["positions"], record["moves_uci"][len(prefix):], strict=True):
        ply = len(state.board.move_stack)
        if terminal_result(state) is not None or ply >= settings.ply_cap:
            raise ValueError("moves after termination/cap")
        mask = legal_mask(state)
        visits = position["root_visits"]
        if (position["ply"] != ply or position["root_fen"] != state.board.fen(en_passant="fen")
                or position["root_promoted"] != int(state.board.promoted)
                or position["side_to_move"] != ("white" if state.board.turn else "black")
                or position["legal_mask"] != mask.tolist()
                or len(visits) != MOVE_COUNT or any(type(n) is not int or n < 0 for n in visits)
                or any(n for n, legal in zip(visits, mask) if not legal)
                or sum(visits) != settings.simulation_budget
                or position["completed_simulations"] != settings.simulation_budget
                or position["exploration"] != asdict(settings.exploration(ply))
                or position["rng_before"] != expected_rng):
            raise ValueError("invalid position/search/history record")
        # The accepted search produces this distribution. Check storage identity,
        # without implementing a new policy normalizer for execution/training.
        if position["visit_policy"] != (np.array(visits, dtype=np.float64) / sum(visits)).tolist():
            raise ValueError("visit policy inconsistent with raw visits")
        move_id = position["move_id"]
        if id_to_move(state, move_id).uci() != uci or position["move_uci"] != uci or visits[move_id] <= 0:
            raise ValueError("played action must have legal completed visits")
        expected_rng = _rng(position["rng_after"]).bit_generator.state
        apply_move(state, move_id)
    if record["rng_state"] != expected_rng:
        raise ValueError("boundary RNG changed")
    expected = {}
    _adjudicate(expected, state, settings.ply_cap)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("boundary board/status/outcome mismatch")
    return state


class Session:
    """Create/load a single owned CPU evaluator and resume only frozen boundaries.

    create() durably freezes identity/settings/opening/RNG before any search.
    load() verifies checksum, replay, exact runtime and checkpoint identity.
    Only interrupted records can advance; cap/completion records stay final.
    """

    @classmethod
    def create(cls, directory, checkpoint_path, opening: Opening, *, settings: Settings,
               seed: int, opening_selection_seed: int):
        _integer(seed, "seed")
        _integer(opening_selection_seed, "opening_selection_seed")
        if not isinstance(opening, Opening) or not opening.history_complete:
            raise ValueError("self-play requires a complete Opening")
        if not isinstance(settings, Settings):
            raise ValueError("settings must be Settings")
        current_runtime = runtime()
        if not current_runtime["deterministic"] or current_runtime["deterministic_warn_only"]:
            raise ValueError("enable strict deterministic CPU algorithms before creation")
        evaluator = RealEvaluator(checkpoint_path)
        checkpoint = _checkpoint(checkpoint_path, evaluator)
        rng = np.random.Generator(np.random.PCG64(seed))
        configuration = json.loads(_json(dict(schema="r01-config-v1", checkpoint=checkpoint,
            contracts=asdict(AdapterConfig()), runtime=current_runtime, settings=asdict(settings),
            seed=seed, opening_selection_seed=opening_selection_seed, opening=opening.to_dict(),
            rng=dict(algorithm="numpy-PCG64", seed=seed, initial_state=rng.bit_generator.state,
                     python_global="unused", numpy_global="unused",
                     torch_cpu="unused by eval/inference; dropout disabled"),
            execution="sequential CPU; batch-of-one; fresh tree per move",
            boundary="atomic full-history + committed search/move + post-move PCG64 state")))
        record = dict(schema="r01-game-v1", configuration_sha256=digest(configuration),
                      positions=[], interruptions=[], rng_state=rng.bit_generator.state)
        _adjudicate(record, opening.state(), settings.ply_cap)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        _write(directory / "configuration.json", configuration)
        _write(directory / "boundary.json", record)
        return cls(directory, configuration, record, evaluator)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        configuration = _read(directory / "configuration.json")
        record = _read(directory / "boundary.json")
        audit(configuration, record)
        if configuration["runtime"] != runtime():
            raise ValueError("resume runtime differs from recorded CPU runtime")
        path = configuration["checkpoint"]["path"]
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != configuration["checkpoint"]["sha256"]:
            raise ValueError("checkpoint bytes changed")
        evaluator = RealEvaluator(path)
        if json.loads(_json(_checkpoint(path, evaluator))) != configuration["checkpoint"]:
            raise ValueError("checkpoint identity/configuration changed")
        return cls(directory, configuration, record, evaluator)

    def __init__(self, directory, configuration, record, evaluator):
        self.directory = Path(directory)
        self.configuration = configuration
        self.record = record
        self.evaluator = evaluator

    def snapshot(self):
        return copy.deepcopy(dict(configuration=self.configuration, game=self.record))

    def _publish(self, record):
        _write(self.directory / "boundary.json", record, replace=True)
        self.record = record

    def run(self, *, stop_after_moves=None):
        """Stop cooperatively after N new commits, or recover a search interruption.

        SIGKILL/crash loses at most the uncommitted search. Partial searches are
        discarded with the pre-search RNG restored. Soft failures are reported
        in interruptions; no partial statistics become committed positions.
        Publication errors propagate: reload the last complete disk boundary.
        """
        if stop_after_moves is not None:
            _integer(stop_after_moves, "stop_after_moves")
        with open(self.directory / ".writer.lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _read(self.directory / "configuration.json") != self.configuration or _read(
                    self.directory / "boundary.json") != self.record:
                raise ValueError("stale session; reload durable boundary")
            if runtime() != self.configuration["runtime"]:
                raise ValueError("execution runtime changed")
            settings = Settings.from_dict(self.configuration["settings"])
            state = audit(self.configuration, self.record)
            commits = 0
            while self.record["status"] == "interrupted":
                if stop_after_moves is not None and commits >= stop_after_moves:
                    record = copy.deepcopy(self.record)
                    record["interruptions"].append(dict(kind="cooperative", ply=len(state.board.move_stack)))
                    self._publish(record)
                    break
                root = Node(state.copy())
                rng = _rng(self.record["rng_state"])
                ply = len(state.board.move_stack)
                try:
                    result = search(root, settings.simulation_budget,
                        config=SearchConfig(settings.c_puct, settings.unvisited_value),
                        evaluator=self.evaluator, exploration=settings.exploration(ply), rng=rng)
                    candidate = state.copy()
                    apply_move(candidate, result.move_id)
                    position = dict(ply=ply, root_fen=state.board.fen(en_passant="fen"),
                        root_promoted=int(state.board.promoted), side_to_move="white" if state.board.turn else "black",
                        legal_mask=legal_mask(state).tolist(), root_visits=result.root_visits.tolist(),
                        visit_policy=result.training_policy.tolist(), completed_simulations=result.completed_simulations,
                        move_id=result.move_id, move_uci=result.move.uci(), exploration=asdict(settings.exploration(ply)),
                        rng_before=copy.deepcopy(self.record["rng_state"]), rng_after=rng.bit_generator.state)
                except (Exception, KeyboardInterrupt, SystemExit) as exc:
                    cause = exc.__cause__ if isinstance(exc, SearchFailure) else exc
                    record = copy.deepcopy(self.record)
                    record["interruptions"].append(dict(kind="soft-interrupt" if isinstance(
                        cause, (KeyboardInterrupt, SystemExit)) else "execution-error", ply=ply,
                        error_type=type(cause).__name__, error_message=str(cause),
                        stage=exc.stage if isinstance(exc, SearchFailure) else "search-or-apply",
                        discarded_visits=[[i, e.visit_count] for i, e in sorted(root.edges.items()) if e.visit_count]))
                    self._publish(record)
                    break
                record = copy.deepcopy(self.record)
                record["positions"].append(position)
                record["rng_state"] = rng.bit_generator.state
                _adjudicate(record, candidate, settings.ply_cap)
                # Do not catch publication exceptions and roll back an uncertain
                # commit. Disk always holds one entire old or new boundary.
                self._publish(record)
                state = candidate
                commits += 1
            return self.snapshot()


def summarize(snapshots):
    """Counts/rates and duplicate excess using initial board + all actual moves."""
    counts = dict(completed=0, truncated=0, interrupted=0)
    completed_keys = []
    for snapshot in snapshots:
        configuration, game = snapshot["configuration"], snapshot["game"]
        audit(configuration, game)
        counts[game["status"]] += 1
        if game["status"] == "completed":
            completed_keys.append(digest([game["initial_fen"], game["moves_uci"]]))
    total = sum(counts.values())
    duplicates = len(completed_keys) - len(set(completed_keys))
    return dict(games=total, status_counts=counts,
                status_rates={k: n / total if total else 0.0 for k, n in counts.items()},
                duplicate_completed_games=duplicates, unique_completed_games=len(set(completed_keys)),
                duplicate_completed_rate=duplicates / counts["completed"] if counts["completed"] else 0.0)
