"""Frozen match execution identity. No implicit upgrade of historical plans."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Execution:
    mode: str = "serial"
    batch_size: int = 1
    games_per_process: int = 1
    processes: int = 5
    flush_policy: str = "all-waiting-or-deadline-v1"
    max_wait_ms: float = 20.0
    timeout_seconds: float = 120.0
    implementation: str = "match-batching-v2"

    def __post_init__(self):
        if self.mode not in ("serial", "batched") or self.implementation not in ("match-batching-v1", "match-batching-v2"):
            raise ValueError("unsupported execution mode or implementation")
        if any(type(x) is not int or x < 1 for x in (self.batch_size, self.games_per_process, self.processes)):
            raise ValueError("execution sizes must be positive integers")
        if self.flush_policy not in ("oldest-deadline-v1", "all-waiting-or-deadline-v1"):
            raise ValueError("unsupported flush policy")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (self.max_wait_ms, self.timeout_seconds)):
            raise ValueError("execution waits must be finite positive numbers")
        if self.mode == "serial" and (self.batch_size != 1 or self.games_per_process != 1):
            raise ValueError("serial execution requires B=K=1")

    def validation_identity(self):
        # Serial never uses batch collection or its implementation/flush knobs.
        # Source/runtime identity is checked independently and remains mandatory.
        if self.mode == "serial":
            return dict(mode="serial", processes=self.processes, batch_size=1, games_per_process=1)
        return self.to_dict()

    def to_dict(self):
        return asdict(self)


def execution_for(plan):
    if "execution" not in plan:
        # Interpret only; never mutate a historical sealed plan.
        return Execution(processes=plan["workers"])
    raw = plan["execution"]
    if not isinstance(raw, dict) or set(raw) != set(Execution().to_dict()):
        raise ValueError("execution must freeze every configuration field")
    execution = Execution(**raw)
    if execution.processes != plan["workers"]:
        raise ValueError("execution process count differs from workers")
    if execution.mode == "batched" and (plan["protocol"] != "fixture" or plan["split"] == "test"):
        raise ValueError("production/test batching blocked pending runtime acceptance and protocol amendment")
    return execution
