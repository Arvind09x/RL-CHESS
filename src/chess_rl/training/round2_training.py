"""Round-2 training from the round-1 selection on the round-1 and round-2 self-play arrays together.

The round-1 arrays (games played by P) and the round-2 arrays (games played by R1-mix) stay
separate verified sets, each with its own manifest; RoundUnion presents them as one train set and
one hold-out set in round order, with the interface round_step, sampler_sources and the evaluator
use. The update step, checkpoint format, sampler, evaluation and candidate rules are
round_training's, unchanged. Round 2 differs from round 1 in four recorded ways: the start weights
(R1-mix) are separate from the forgetting reference (P); new sampler and run seeds, so the
Stockfish rows are not R1-mix's rows in the same order; a hard stop with no eligible checkpoint
falls back to the start checkpoint, not to P; checkpoint ids start with r2-.

Sidecars keep round_training's sidecar format and version (resume_point reads them unchanged);
the round is recorded in the provenance (schema round2-training-v1) and in the checkpoint ids.
"""
from pathlib import Path

import numpy as np

from chess_rl.evaluation.matches import digest
from .mixed_state import SelfPlayBatch, require
from . import round_training as R

UNION_VERSION = "round-union-v1"
RUN_VERSION = "round2-training-v1"
# Unused elsewhere (round-1 sampler 20261017 and run 20261018; round-2 self-play 20261021-20261024,
# its dry run 20261121; matches 2026101801-2026101807). Round 1's sampler seed would give R2-mix
# R1-mix's first 18,424 Stockfish rows again, in the same order.
SAMPLING_SEED = 20261025
RUN_SEED = 20261026
NAMES = dict(mix="R2-mix", pure="R2-pure")
START_TOLERANCE = 0.001
SAMPLER_NOTE = (f"mixed_state is the shared self-play sampler over the union of the round data sets (seed "
                f"{SAMPLING_SEED}, composition 8 Stockfish + 24 self-play). For R2-pure its 8 Stockfish indices "
                "per update are discarded, so its counts and the checkpoint progress samples_seen (32 per update) "
                "are sampler bookkeeping, not trained positions; the recipe is this provenance's recipe field.")
FALLBACK_RULE = ("if the hard stop fires with no eligible checkpoint, the candidate is the start checkpoint "
                 "(the round-1 selection), not P; the original rule's output is kept beside it")


class ConcatRows:
    """Rows of several RoundRows sets in order; batch() keeps the requested order."""

    def __init__(self, parts):
        self.parts = tuple(parts)
        require(self.parts, "at least one row set required")
        self.sample_ids = tuple(s for p in self.parts for s in p.sample_ids)
        self.game_ids = tuple(g for p in self.parts for g in p.game_ids)
        self.plies = tuple(x for p in self.parts for x in p.plies)
        require(len(set(self.sample_ids)) == len(self.sample_ids), "sample ids repeat across rounds")
        self.offsets = tuple(int(x) for x in np.cumsum([0] + [len(p) for p in self.parts]))

    def __len__(self):
        return len(self.sample_ids)

    def part_slices(self):
        return [slice(a, b) for a, b in zip(self.offsets, self.offsets[1:])]

    def batch(self, indices):
        """Dense batch equal row by row to the owning set's batch() for the same samples."""
        indices = tuple(indices)
        require(bool(indices) and all(type(i) is int and 0 <= i < len(self) for i in indices),
                "batch indices must be nonempty valid integers")
        part_of = np.searchsorted(np.asarray(self.offsets), indices, side="right") - 1
        rows = [None] * len(indices)
        for k, part in enumerate(self.parts):
            where = np.flatnonzero(part_of == k)
            if not len(where):
                continue
            local = part.batch([indices[j] - self.offsets[k] for j in where])
            for n, j in enumerate(where):
                rows[j] = (local, n)
        return SelfPlayBatch(tuple(b.sample_ids[n] for b, n in rows),
                             np.stack([b.features[n] for b, n in rows]),
                             np.stack([b.legal_mask[n] for b, n in rows]),
                             np.stack([b.policy_distributions[n] for b, n in rows]),
                             np.stack([b.wdl[n] for b, n in rows]))


def union_rows(train_parts, holdout_parts):
    """(train, holdout) ConcatRows after the cross-round checks: no repeated sample, no game in two rounds
    or in both sets."""
    train, holdout = ConcatRows(train_parts), ConcatRows(holdout_parts)
    require(not set(train.sample_ids) & set(holdout.sample_ids), "a sample is in the training set and the hold-out set")
    train_games, holdout_games = set(train.game_ids), set(holdout.game_ids)
    require(not train_games & holdout_games, "a game is in the training set and the hold-out set")
    require(len(train_games) == sum(len(set(p.game_ids)) for p in train_parts), "a training game appears in two rounds")
    require(len(holdout_games) == sum(len(set(p.game_ids)) for p in holdout_parts), "a hold-out game appears in two rounds")
    return train, holdout


class RoundUnion:
    """Several verified round data sets trained as one; each part is checked against its own player."""

    def __init__(self, parts, manifest, replay_id, train, holdout):
        self.parts, self.manifest, self.replay_id, self.train, self.holdout = parts, manifest, replay_id, train, holdout

    @classmethod
    def open(cls, specs, *, registries=None):
        """specs: [(directory, sha256 of the checkpoint that played its games)] in round order."""
        require(specs, "at least one round data set required")
        parts = []
        for directory, played_by in specs:
            data = R.RoundData.open(directory, registries=registries)
            require(data.manifest["checkpoint_sha256"] == played_by,
                    f"{directory}: its games were played by another checkpoint")
            parts.append(data)
        require(len({p.replay_id for p in parts}) == len(parts), "the same round data is listed twice")
        for key, what in (("supervised_source_id", "Stockfish sources"), ("registries", "held-out registries"),
                          ("holdout", "hold-out rules"), ("contracts", "adapter contracts")):
            require(len({digest(p.manifest[key]) for p in parts}) == 1, f"round data sets have different {what}")
        train, holdout = union_rows([p.train for p in parts], [p.holdout for p in parts])
        members = [dict(round=k + 1, directory=str(p.directory), replay_id=p.replay_id,
                        checkpoint_sha256=p.manifest["checkpoint_sha256"], training_positions=len(p.train),
                        holdout_positions=len(p.holdout), counts=p.manifest["counts"])
                   for k, p in enumerate(parts)]
        manifest = dict(version=UNION_VERSION, parts=members, order="round order, each part in its own row order",
                        supervised_source_id=parts[0].manifest["supervised_source_id"],
                        registries=parts[0].manifest["registries"], holdout=parts[0].manifest["holdout"])
        return cls(tuple(parts), manifest, digest([UNION_VERSION, [p.replay_id for p in parts]]), train, holdout)


def holdout_by_round(values, holdout):
    """Per-position hold-out arrays split into one slice per round, in round order."""
    return [{k: np.asarray(v)[s] for k, v in values.items()} for s in holdout.part_slices()]


def recipe_details(recipe):
    return dict(R.recipe_provenance(recipe), name=NAMES[recipe])


def start_candidate(original, *, start_path, start_sha256, start_checkpoint_id):
    """Map round_training.select_candidate's output to round 2's rule (fallback = the start checkpoint)."""
    if original["candidate"] != "P":
        return dict(original, fallback_to_start=False)
    return dict(original, candidate="start", candidate_update=0, fallback_to_start=True,
                start=dict(path=str(start_path), sha256=start_sha256, checkpoint_id=start_checkpoint_id),
                reason=original["reason"].replace("no earlier checkpoint is eligible, so P",
                                                  "no earlier checkpoint is eligible, so the start checkpoint (R1-mix)"),
                rule=FALLBACK_RULE)


def candidate_record(choice, *, output, sha256):
    """Where the candidate's file is: in the run directory, or the start checkpoint itself."""
    if choice["candidate"] == "start":
        return dict(path=choice["start"]["path"], sha256=choice["start"]["sha256"], update=0,
                    checkpoint_id=choice["start"]["checkpoint_id"], source="start checkpoint (fallback)")
    path = Path(output) / choice["candidate"]
    return dict(path=str(path), sha256=sha256(path), update=choice["candidate_update"], source="this run")
