"""Small typed CPU results shared by adapter consumers."""
from dataclasses import dataclass

import chess
import numpy as np

from chess_rl.contracts import FEATURE_SHAPE, MOVE_COUNT


def _array(value, shape, name):
    result = np.array(value, dtype=np.float32, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    return result


def _probabilities(value, shape, name):
    result = _array(value, shape, name)
    if (result < 0).any() or not np.isclose(result.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return result


@dataclass(frozen=True)
class EncodedPosition:
    features: np.ndarray
    history_complete: bool
    available_snapshots: int

    def __post_init__(self):
        if type(self.history_complete) is not bool:
            raise ValueError("history_complete must be a boolean")
        if type(self.available_snapshots) is not int or not 1 <= self.available_snapshots <= 8:
            raise ValueError("available_snapshots must be an integer in 1..8")
        object.__setattr__(self, "features", _array(self.features, FEATURE_SHAPE, "features"))


@dataclass(frozen=True)
class Evaluation:
    priors: np.ndarray
    wdl: np.ndarray
    value: float

    def __post_init__(self):
        object.__setattr__(self, "priors", _probabilities(self.priors, (MOVE_COUNT,), "priors"))
        object.__setattr__(self, "wdl", _probabilities(self.wdl, (3,), "wdl"))
        if not np.isfinite(self.value) or not np.isclose(self.value, self.wdl[0] - self.wdl[2], atol=1e-6, rtol=0):
            raise ValueError("value must equal P(win)-P(loss)")


@dataclass(frozen=True)
class TerminalResult:
    winner: chess.Color | None
    ending_reason: str
    wdl: np.ndarray
    value: float
    history_complete: bool
    priors: None = None

    def __post_init__(self):
        if self.winner is not None and type(self.winner) is not bool:
            raise ValueError("winner must be White, Black or None")
        if self.ending_reason not in chess.Termination.__members__:
            raise ValueError("Unknown ending_reason")
        if type(self.history_complete) is not bool or self.priors is not None:
            raise ValueError("Terminal results require boolean provenance and priors=None")
        object.__setattr__(self, "wdl", _probabilities(self.wdl, (3,), "wdl"))
        expected = (0, 1, 0) if self.winner is None else (0, 0, 1)
        if not np.array_equal(self.wdl, expected) or self.value != (0.0 if self.winner is None else -1.0):
            raise ValueError("Terminal WDL/value must be an exact draw or side-to-move loss")
