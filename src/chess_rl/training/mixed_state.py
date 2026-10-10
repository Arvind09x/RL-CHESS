"""R03 frozen source membership and independent cyclic sampling state.

No model execution, game generation or replay mutation. Frozen training views
must be reopened and matched on resume; adding data requires a new cycle.
"""
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from chess_rl.adapter import encode_position, make_state
from chess_rl.canonical_games import canonical_payload, game_id
from chess_rl.evaluation.matches import digest
from chess_rl.pgn_replay import file_identity
from chess_rl.position_sampling import rule_position
from chess_rl.supervised import SupervisedBatch, SupervisedDataset, _freeze_metadata
from .trainer import LossWeights

MIXED_VERSION = "r03-mixed-order-v1"
SCHEDULE = "one-update-per-full-batch-v1"
SAMPLING = "independent-shuffled-cycles-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            "invalid mixed source/membership SHA256")


@dataclass(frozen=True)
class MixedSettings:
    supervised_fraction: float = 0.5
    batch_size: int = 4
    supervised_batch_size: int = 2
    learning_rate: float = 1e-5
    policy_weight: float = 1.0
    wdl_weight: float = 1.0
    schedule: str = SCHEDULE
    sampling: str = SAMPLING

    def __post_init__(self):
        require(type(self.supervised_fraction) in (int, float)
                and math.isfinite(self.supervised_fraction) and 0 < self.supervised_fraction < 1,
                "supervised fraction must be strictly between zero and one")
        require(type(self.batch_size) is int and type(self.supervised_batch_size) is int
                and 0 < self.supervised_batch_size < self.batch_size,
                "mixed batch must contain both sources")
        require(self.supervised_fraction == self.supervised_batch_size / self.batch_size,
                "supervised fraction must match exact batch composition")
        require(type(self.learning_rate) in (int, float) and math.isfinite(self.learning_rate)
                and self.learning_rate > 0, "learning rate must be finite positive")
        LossWeights(self.policy_weight, self.wdl_weight)
        require(self.schedule == SCHEDULE and self.sampling == SAMPLING, "unsupported mixed schedule")

    @property
    def selfplay_batch_size(self):
        return self.batch_size - self.supervised_batch_size

    @property
    def weights(self):
        return LossWeights(self.policy_weight, self.wdl_weight)


@dataclass(frozen=True)
class SourceCursor:
    order: tuple[int, ...]
    cursor: int = 0
    epoch: int = 0
    samples_seen: int = 0

    def __post_init__(self):
        require(type(self.order) is tuple and len(self.order) > 0
                and all(type(v) is int for v in self.order)
                and sorted(self.order) == list(range(len(self.order))), "source order must be a full permutation")
        require(all(type(v) is int and v >= 0 for v in (self.cursor, self.epoch, self.samples_seen))
                and self.cursor <= len(self.order), "invalid source cursor/progress")
        require(self.samples_seen == self.epoch * len(self.order) + self.cursor,
                "source cursor/progress mismatch")


def validate_sampling_rng(state):
    require(type(state) is tuple and len(state) == 3 and type(state[0]) is int and state[0] == 3
            and type(state[1]) is tuple and len(state[1]) == 625
            and all(type(v) is int and 0 <= v < 2**32 for v in state[1][:-1])
            and type(state[1][-1]) is int and 0 <= state[1][-1] <= 624
            and (state[2] is None or type(state[2]) is float and math.isfinite(state[2])),
            "invalid mixed sampling RNG")
    random.Random().setstate(state)


@dataclass(frozen=True)
class MixedState:
    settings: MixedSettings
    supervised_source_id: str
    supervised_membership_sha256: str
    replay_id: str
    replay_membership: tuple[str, ...]
    supervised: SourceCursor
    selfplay: SourceCursor
    sampling_rng: tuple
    updates: int = 0
    version: str = MIXED_VERSION

    def __post_init__(self):
        require(type(self.settings) is MixedSettings and type(self.supervised) is SourceCursor
                and type(self.selfplay) is SourceCursor, "invalid mixed settings/cursors")
        for value in (self.supervised_source_id, self.supervised_membership_sha256, self.replay_id):
            sha(value)
        require(type(self.replay_membership) is tuple and len(self.replay_membership) == len(self.selfplay.order)
                and len(set(self.replay_membership)) == len(self.replay_membership), "invalid replay membership")
        for value in self.replay_membership:
            sha(value)
        require(type(self.updates) is int and self.updates >= 0 and self.version == MIXED_VERSION,
                "invalid mixed update progress/version")
        require(self.supervised.samples_seen == self.updates * self.settings.supervised_batch_size
                and self.selfplay.samples_seen == self.updates * self.settings.selfplay_batch_size,
                "mixed per-source progress/composition mismatch")
        validate_sampling_rng(self.sampling_rng)

    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict and set(value) == set(cls.__dataclass_fields__), "mixed state schema mismatch")
        value = dict(value)
        for key, kind in (("settings", MixedSettings), ("supervised", SourceCursor), ("selfplay", SourceCursor)):
            require(type(value[key]) is dict and set(value[key]) == set(kind.__dataclass_fields__),
                    "mixed nested schema mismatch")
            value[key] = kind(**value[key])
        return cls(**value)

    def validate_sources(self, sources):
        require(type(sources) is MixedSources, "audited mixed sources required")
        sources.validate_dataset()
        require(self.supervised_source_id == sources.supervised_source_id
                and self.supervised_membership_sha256 == sources.supervised_membership_sha256
                and len(self.supervised.order) == len(sources.dataset)
                and self.replay_id == sources.replay_id and self.replay_membership == sources.replay_membership,
                "incompatible mixed source identity/membership")

    @classmethod
    def initial(cls, sources, settings, *, seed):
        require(type(seed) is int and seed >= 0, "explicit sampling seed required")
        rng = random.Random(seed)
        supervised, selfplay = list(range(len(sources.dataset))), list(range(len(sources.replay_membership)))
        rng.shuffle(supervised)
        rng.shuffle(selfplay)
        return cls(settings, sources.supervised_source_id, sources.supervised_membership_sha256,
                   sources.replay_id, sources.replay_membership, SourceCursor(tuple(supervised)),
                   SourceCursor(tuple(selfplay)), rng.getstate())

    def next_selection(self):
        """Pure candidate transition; caller commits it after the optimizer succeeds."""
        rng = random.Random()
        rng.setstate(self.sampling_rng)
        def draw(source, count):
            order, cursor, epoch = source.order, source.cursor, source.epoch
            indices = []
            for _ in range(count):
                if cursor == len(order):
                    order = list(range(len(order)))
                    rng.shuffle(order)
                    order, cursor, epoch = tuple(order), 0, epoch + 1
                indices.append(order[cursor])
                cursor += 1
            return tuple(indices), SourceCursor(order, cursor, epoch, source.samples_seen + count)
        supervised, s = draw(self.supervised, self.settings.supervised_batch_size)
        selfplay, r = draw(self.selfplay, self.settings.selfplay_batch_size)
        state = replace(self, supervised=s, selfplay=r, sampling_rng=rng.getstate(), updates=self.updates + 1)
        return supervised, selfplay, state


@dataclass(frozen=True)
class SelfPlayBatch:
    sample_ids: tuple[str, ...]
    features: np.ndarray
    legal_mask: np.ndarray
    policy_distributions: np.ndarray
    wdl: np.ndarray  # exact one-hot final root-side win/draw/loss


def _supervised_membership(dataset):
    values = []
    for i in range(len(dataset)):
        sample = dataset[i]
        values.append(dict(reference=sample.reference, split=sample.split,
            features=hashlib.sha256(sample.features.tobytes()).hexdigest(),
            mask=hashlib.sha256(sample.legal_mask.tobytes()).hexdigest(),
            policy_class=sample.policy_class, wdl=sample.wdl.tolist()))
    return digest(values)


class MixedSources:
    """D07 train plus an R02 audited, held-out-filtered immutable training view.

    R02 remains unchanged. Its receipts/targets bind replay identity; filtering
    happens before any shuffle. Registries must match D07's verified identities.
    Reopen sources for restoration; in-memory views are owned by the caller.
    """
    def __init__(self, dataset, buffer, *, heldout_games_path, protected_positions_path):
        from chess_rl.replay import ReplayBuffer
        require(type(dataset) is SupervisedDataset and dataset.split == "train" and len(dataset) > 0,
                "mixed sources require nonempty audited D07 training split")
        identity = dataset.source_identity
        require(type(identity) is dict and identity.get("split") == "train", "audited D07 source identity required")
        require(type(buffer) is ReplayBuffer, "audited R02 buffer required")
        paths = {"heldout-games.jsonl": Path(heldout_games_path),
                 "protected-heldout.jsonl": Path(protected_positions_path)}
        for name, path in paths.items():
            require(file_identity(path) == identity["authoritative"][name], "held-out registry identity mismatch")
        export = buffer.export()
        protected = set()
        for raw in paths["protected-heldout.jsonl"].read_text().splitlines():
            protected.add(json.loads(raw)["rule_position"])
        held_games = set()
        for raw in paths["heldout-games.jsonl"].read_text().splitlines():
            held_games.add(json.loads(raw)["game_id"])
        blocked_games = set()
        for source in export["sources"]:
            if source["disposition"] == "admitted":
                game = source["snapshot"]["game"]
                if game_id(canonical_payload(game["initial_fen"], game["moves_uci"])) in held_games:
                    blocked_games.add(source["game_id"])
        samples, excluded = [], []
        for sample in export["samples"]:
            state = make_state(sample["initial_fen"], sample["moves_uci"], history_complete=True)
            if sample["game_id"] in blocked_games or rule_position(state.board) in protected:
                excluded.append(sample["sample_id"])
            else:
                samples.append(sample)
        require(len(samples) > 0, "no completed non-held-out replay samples")
        for name, path in paths.items():
            require(file_identity(path) == identity["authoritative"][name], "held-out registry changed while loading")
        self.dataset = dataset.freeze()
        self._frozen_dataset = self.dataset
        self._samples = _freeze_metadata(samples)
        self.excluded_heldout = tuple(excluded)
        self.supervised_source_id = digest(identity)
        self.supervised_membership_sha256 = _supervised_membership(self.dataset)
        self.replay_id = digest(export)
        self.replay_membership = tuple(s["sample_id"] for s in samples)

    def batches(self, supervised_indices, replay_indices):
        self.validate_dataset()
        s = [self.dataset[i] for i in supervised_indices]
        r = [self._samples[i] for i in replay_indices]
        supervised = SupervisedBatch(tuple(v.reference for v in s), "train",
            np.stack([v.features for v in s]), np.stack([v.legal_mask for v in s]),
            np.array([v.policy_class for v in s], dtype=np.int64), np.stack([v.wdl for v in s]))
        replay = SelfPlayBatch(tuple(v["sample_id"] for v in r),
            np.stack([encode_position(make_state(v["initial_fen"], v["moves_uci"], history_complete=True)).features for v in r]),
            np.array([v["position"]["legal_mask"] for v in r], dtype=np.bool_),
            np.array([v["position"]["visit_policy"] for v in r], dtype=np.float32),
            np.array([v["wdl"] for v in r], dtype=np.float32))
        return supervised, replay

    def validate_dataset(self):
        # Immutable owned metadata was fully fingerprinted at admission. No
        # dataset-wide replay/encoding belongs on the optimizer hot path.
        require(self.dataset is self._frozen_dataset and self.dataset._sealed,
                "supervised dataset changed after membership freeze")
