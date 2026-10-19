"""Versioned per-round sampling; legacy MixedState remains byte-for-byte unchanged.

Select the smallest next deadline (draws_i + 1) / weight_i, ties in input order.
Integer cross-products avoid drift. At every draw prefix, discrepancy from N*w_i
is strictly less than max(1, active_rounds - 1) draws (one active round is exact).
No per-batch rounding or allocation reset. Local cycles sample without replacement.
A shadow legacy sampler preserves Stockfish's original shared-RNG sequence exactly;
its self-play draws are bookkeeping only and never enter the loss.
"""
from dataclasses import dataclass, fields
from fractions import Fraction
import math
import random

from .mixed_state import MixedState, SourceCursor, require, sha, validate_sampling_rng

VERSION = "round-deadline-cycles-v1"
DEFAULT_WEIGHTS = {"mix": [0.20, 0.80], "pure": [0.00, 1.00]}
NOTE = __doc__.strip()


def weight_units(weights, count):
    require(len(weights) == count, "one round weight required per ordered --round-data input")
    require(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights),
            "round weights must be finite nonnegative numbers")
    fractions = [Fraction(str(w)) for w in weights]
    denominator = math.lcm(*(w.denominator for w in fractions))
    units = [int(w * denominator) for w in fractions]
    divisor = math.gcd(*units)
    require(divisor > 0, "round weights cannot all be zero")
    return tuple(u // divisor for u in units)


def configuration(weights, identities, sizes, budget):
    units = weight_units(weights, len(identities))
    require(type(budget) is int and budget > 0, "weighted sampling requires explicit positive --updates")
    require(len(sizes) == len(units) and all(type(n) is int and n > 0 for n in sizes),
            "round training sets must be nonempty")
    return dict(version=VERSION, round_ids=list(identities), training_positions=list(sizes),
                requested_weights=list(weights), normalized_weights=[u / sum(units) for u in units],
                weight_units=list(units), total_updates=budget)


@dataclass(frozen=True)
class RoundSamplingState:
    legacy: MixedState
    config: dict
    rounds: tuple[SourceCursor, ...]
    round_rngs: tuple
    version: str = VERSION

    def __getattr__(self, name):
        # Existing progress/optimizer plumbing sees the unchanged legacy bookkeeping.
        return getattr(self.legacy, name)

    def __post_init__(self):
        require(self.version == VERSION and type(self.legacy) is MixedState,
                "unsupported round sampler version/state")
        c = self.config
        require(c == configuration(c["requested_weights"], c["round_ids"], c["training_positions"],
                                   c["total_updates"]), "round sampler configuration mismatch")
        require(len(set(c["round_ids"])) == len(c["round_ids"]), "repeated round identities")
        for identity in c["round_ids"]:
            sha(identity)
        require(type(self.rounds) is tuple and type(self.round_rngs) is tuple
                and len(self.rounds) == len(self.round_rngs) == len(c["round_ids"]),
                "round sampler state count mismatch")
        require(sum(c["training_positions"]) == len(self.legacy.replay_membership),
                "round sampler membership size mismatch")
        for cursor, rng, size, weight in zip(self.rounds, self.round_rngs, c["training_positions"], c["weight_units"]):
            require(type(cursor) is SourceCursor and len(cursor.order) == size, "round cursor size mismatch")
            require(weight > 0 or cursor.samples_seen == 0, "zero-weight round has draws")
            validate_sampling_rng(rng)
        # The last selected deadline cannot follow an unselected deadline (including ties).
        for i, (r, u) in enumerate(zip(self.rounds, c["weight_units"])):
            if not r.samples_seen:
                continue
            for j, (other, v) in enumerate(zip(self.rounds, c["weight_units"])):
                if v:
                    left, right = r.samples_seen * v, (other.samples_seen + 1) * u
                    require(left < right or (left == right and i < j), "round allocation progress mismatch")
        require(sum(r.samples_seen for r in self.rounds) == self.selfplay.samples_seen
                and self.updates <= c["total_updates"], "round allocation progress/budget mismatch")

    @classmethod
    def initial(cls, legacy, config, *, seed):
        cursors, rngs = [], []
        for i, size in enumerate(config["training_positions"]):
            rng = random.Random(f"{VERSION}:{seed}:{i}")
            order = list(range(size))
            rng.shuffle(order)
            cursors.append(SourceCursor(tuple(order)))
            rngs.append(rng.getstate())
        return cls(legacy, config, tuple(cursors), tuple(rngs))

    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict and set(value) == {f.name for f in fields(cls)},
                "round sampling state schema mismatch")
        return cls(MixedState.from_dict(value["legacy"]), value["config"],
                   tuple(SourceCursor(**r) for r in value["rounds"]), value["round_rngs"], value["version"])

    def next_selection(self):
        require(self.updates < self.config["total_updates"], "round sampling update budget exhausted")
        si, _, legacy = self.legacy.next_selection()
        rounds, rngs = list(self.rounds), list(self.round_rngs)
        units = self.config["weight_units"]
        active = [i for i, u in enumerate(units) if u]
        offsets, offset = [], 0
        for size in self.config["training_positions"]:
            offsets.append(offset)
            offset += size
        selected = []
        for _ in range(self.settings.selfplay_batch_size):
            i = active[0]
            for j in active[1:]:
                if (rounds[j].samples_seen + 1) * units[i] < (rounds[i].samples_seen + 1) * units[j]:
                    i = j
            r = rounds[i]
            order, cursor, epoch = r.order, r.cursor, r.epoch
            if cursor == len(order):
                rng = random.Random()
                rng.setstate(rngs[i])
                order = list(range(len(order)))
                rng.shuffle(order)
                order, cursor, epoch = tuple(order), 0, epoch + 1
                rngs[i] = rng.getstate()
            selected.append(offsets[i] + order[cursor])
            # As in MixedState: permutations are checked at creation/load, not on every draw.
            next_cursor = object.__new__(SourceCursor)
            for key, value in dict(order=order, cursor=cursor + 1, epoch=epoch,
                                   samples_seen=r.samples_seen + 1).items():
                object.__setattr__(next_cursor, key, value)
            rounds[i] = next_cursor
        state = object.__new__(RoundSamplingState)
        for key, value in dict(legacy=legacy, config=self.config, rounds=tuple(rounds),
                               round_rngs=tuple(rngs), version=VERSION).items():
            object.__setattr__(state, key, value)
        return si, tuple(selected), state

    def report(self):
        return dict(configuration=self.config, by_round=[dict(
            round_id=identity, draws=r.samples_seen, unique_positions_seen=min(size, r.samples_seen),
            reused_draws=max(0, r.samples_seen - size), effective_passes=r.samples_seen / size,
            oversampled=r.samples_seen > size, cycles_completed=r.samples_seen // size)
            for identity, size, r in zip(self.config["round_ids"], self.config["training_positions"], self.rounds)])


def sampling_state_from_dict(value):
    if value.get("version") == VERSION:
        return RoundSamplingState.from_dict(value)
    # Unknown versions fail the legacy validator; never reinterpret them.
    return MixedState.from_dict(value)
