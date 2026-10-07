"""Versioned adapter contract; no model/search configuration yet."""
from dataclasses import dataclass

ENCODING_VERSION = "chess128-v2"
MOVE_VOCAB_VERSION = "pair4352-v1"
ENDING_POLICY = "actual-occurrence-v1"
WDL_ORDER = ("win", "draw", "loss")
FEATURE_SHAPE = (64, 128)
MOVE_COUNT = 4352


@dataclass(frozen=True)
class AdapterConfig:
    encoding_version: str = ENCODING_VERSION
    move_vocab_version: str = MOVE_VOCAB_VERSION
    ending_policy: str = ENDING_POLICY
    wdl_order: tuple[str, str, str] = WDL_ORDER

    def __post_init__(self):
        for name, expected in (
            ("encoding_version", ENCODING_VERSION),
            ("move_vocab_version", MOVE_VOCAB_VERSION),
            ("ending_policy", ENDING_POLICY),
            ("wdl_order", WDL_ORDER),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"Unsupported {name}: {getattr(self, name)!r}; expected {expected!r}")
