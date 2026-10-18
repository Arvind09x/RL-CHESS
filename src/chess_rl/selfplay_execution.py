"""Explicit self-play execution identity, independent of match protocol/selection."""
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class Execution:
    mode: str = "serial"
    workers: int = 5
    games_per_worker: int = 1
    batch_size: int = 1
    max_wait_ms: float = 20.0
    timeout_seconds: float = 120.0
    flush_policy: str = "all-waiting-or-deadline-v1"
    implementation: str = "selfplay-batching-v1"
    padding: str = "standard-start-fixed-shape-v1"

    def __post_init__(self):
        if self.mode not in ("serial", "batched"):
            raise ValueError("unsupported execution mode")
        for value in (self.workers, self.games_per_worker, self.batch_size):
            if type(value) is not int or value < 1:
                raise ValueError("execution counts must be positive integers")
        for value in (self.max_wait_ms, self.timeout_seconds):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("execution waits must be finite and positive")
        if self.mode == "serial" and (self.games_per_worker != 1 or self.batch_size != 1):
            raise ValueError("serial requires one game and batch size one")
        if (self.flush_policy != "all-waiting-or-deadline-v1"
                or self.implementation != "selfplay-batching-v1"
                or self.padding != "standard-start-fixed-shape-v1"):
            raise ValueError("unsupported execution implementation")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("execution requires every frozen field")
        return cls(**value)


def source_identity(root=None):
    """Hash actual source, including untracked modules; never claim a dirty tree is committed."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    paths = sorted(p for folder in ("src", "scripts", "configs", "runtime", "jobs")
                   for p in (root / folder).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths += [root / name for name in ("pyproject.toml", "uv.lock")]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from chess_rl.evaluation.matches import digest
    return dict(schema="selfplay-source-content-v1", sha256=digest(files), files=files)


def binding(plan):
    from chess_rl.evaluation.matches import digest
    return dict(kind="cross-game-selfplay-v1", execution=plan["execution"],
                campaign_sha256=digest(plan), source_sha256=plan["execution_source"]["sha256"],
                checkpoint_sha256=plan["checkpoint"]["sha256"])


def session_runtime(device, execution=None):
    from chess_rl.selfplay import runtime
    result = runtime(device)
    if execution is not None:
        result["batch_size"] = Execution.from_dict(execution["execution"]).batch_size
    return result


def execution_text(device, execution=None):
    return execution if execution is not None else (
        f"sequential {'CPU' if device == 'cpu' else device}; batch-of-one; fresh tree per move")
