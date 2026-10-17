"""Round-1 training from P on one frozen self-play replay buffer: R1-mix and R1-pure.

Self-play rows are streamed once from the audited replay buffer into compact,
hash-verified arrays (build_round_data). Rows are filtered before any shuffle
by MixedSources' rule (games whose canonical identity is a held-out Stockfish
game, positions whose rule position is protected) and then split by the
self-play hold-out, which takes whole games by a fixed hash of the replay game
id. Held-out games are kept as a separate evaluation set.

Both recipes draw from one MixedState sampler built with the same seed, so the
self-play order is identical by construction. R1-mix trains on the sampler's 8
cached Stockfish rows plus its 24 self-play rows with the library's
mixed_losses (0.25 / 0.75). R1-pure trains on the 24 self-play rows alone and
discards the 8 Stockfish indices. MixedSources and mixed_step are not used:
they need a whole-buffer export and an uncached dataset.

Checkpoints are the library's mixed format (save_checkpoint with the sampler as
mixed_state), so load_inference, matches and self-play read them unchanged.
The recipe is recorded in the provenance and is read only from there; the
sampler's per-update count of 32 is bookkeeping and is never reported as the
number of positions R1-pure trained on. restore_round resumes a checkpoint with
the safeguards of restore_training. Each checkpoint is followed by a sidecar;
only a checkpoint with a matching sidecar is a resume point.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from chess_rl.adapter import encode_position, make_state
from chess_rl.canonical_games import canonical_payload, game_id as canonical_game_id
from chess_rl.contracts import AdapterConfig, FEATURE_SHAPE, MOVE_COUNT
from chess_rl.evaluation.matches import digest
from chess_rl.model import ChessNetwork
from chess_rl.pgn_replay import file_identity
from chess_rl.position_sampling import rule_position
from chess_rl.tensor_ops import legal_policy_log_probs, validate_soft_wdl
from .checkpoints import (CUDA_MIXED_CHECKPOINT_VERSION, MIXED_CHECKPOINT_VERSION, CheckpointError, ModelConfig,
    _optimizer_payload, _read_payload, _require, _restore_optimizer, _same, _validate_payload,
    cuda_runtime_config, runtime_config)
from .losses import (policy_distribution_cross_entropy, soft_wdl_cross_entropy,
                     validate_policy_classes, validate_policy_distribution)
from .mixed import mixed_losses, validate_optimizer_settings
from .mixed_state import MixedSettings, MixedState, SelfPlayBatch, _supervised_membership, require
from .optimizers import OptimizerSettings, build_optimizer, clip_gradients
from .trainer import finite_tensors

DATA_VERSION = "r1-round-data-v1"
RUN_VERSION = "round1-training-v1"
SIDECAR_VERSION = "round1-checkpoint-sidecar-v1"
RECIPES = ("mix", "pure")
REGISTRIES = ("heldout-games.jsonl", "protected-heldout.jsonl")
HOLDOUT_RULE = "r1-holdout-v1"
HOLDOUT_DESCRIPTION = ('a game is held out iff int(sha256(("r1-holdout-v1" + game_id).encode())'
                       '.hexdigest()[:16], 16) % 10 == 0, game_id being the replay buffer game id')
FILTER_DESCRIPTION = ("before any shuffle: drop every position of a game whose canonical identity "
                      "(canonical_games.game_id of canonical_payload(initial FEN, full moves)) is in "
                      "heldout-games.jsonl, and every position whose rule_position is in protected-heldout.jsonl "
                      "(MixedSources' rule); then move whole hold-out games to the evaluation set")
# Frozen recipe settings (train-mixed-plan, amendments A1 and A4). The sampler always uses the mixed
# composition; R1-pure discards its Stockfish indices.
SETTINGS = MixedSettings(supervised_fraction=0.25, batch_size=32, supervised_batch_size=8, learning_rate=1e-5)
OPTIMIZER_SETTINGS = OptimizerSettings(learning_rate=1e-5, weight_decay=0.01, max_gradient_norm=10.0)
SAMPLING_SEED = 20261017
RUN_SEED = 20261018
CHECKPOINT_EVERY = 250
FLAG_DELTA = 0.05
STOP_DELTA = 0.25
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261017
# Record fields that measure time; every comparison of two runs ignores them.
TIMING_KEYS = frozenset({"elapsed_seconds", "update_seconds", "timings", "started_utc", "written_utc", "created_utc"})
_CONTRACTS = digest(asdict(AdapterConfig()))
_MASK_BYTES = MOVE_COUNT // 8


def held_out(game_id):
    """The fixed self-play hold-out rule over replay game ids."""
    return int(hashlib.sha256((HOLDOUT_RULE + game_id).encode()).hexdigest()[:16], 16) % 10 == 0


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync_dir(path):
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def publish_json(path, value):
    """Write a JSON record atomically; refuse an existing destination."""
    path = Path(path)
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    try:
        with pending.open("x") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        _fsync_dir(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def array_bytes(shape, dtype):
    """Exact .npy size for a C-order array (format 1.0 header padded to 128 bytes for these shapes)."""
    return 128 + int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


# Held-out registries.

def read_registries(prepared, identity):
    """Held-out game ids and protected rule positions, after matching the dataset's verified identities."""
    prepared = Path(prepared)
    identities = {}
    for name in REGISTRIES:
        identities[name] = file_identity(prepared / name)
        require(identities[name] == identity["authoritative"][name], f"held-out registry identity mismatch: {name}")
    held = {json.loads(raw)["game_id"] for raw in (prepared / "heldout-games.jsonl").read_text().splitlines()}
    protected = {json.loads(raw)["rule_position"]
                 for raw in (prepared / "protected-heldout.jsonl").read_text().splitlines()}
    return held, protected, identities


# Streaming the replay buffer into compact rows.

def stream_rows(buffer, *, protected, checkpoint_sha256=None):
    """One pass over buffer.iter_samples(), each sample converted to compact row data at once.

    No sample dict is kept. Returns row arrays in the buffer's sample-id order plus, per game,
    what is needed to rebuild its full move list (initial FEN, plies seen, the highest-ply
    sample's moves plus its own move).
    """
    capacity = buffer.summary()["training_positions"]
    features = np.empty((capacity, *FEATURE_SHAPE), dtype=np.float32)
    masks = np.empty((capacity, _MASK_BYTES), dtype=np.uint8)
    wdl = np.empty((capacity, 3), dtype=np.float32)
    policy_indices, policy_values, lengths = [], [], []
    sample_ids, game_ids, plies, is_protected = [], [], [], []
    games = {}
    n = 0
    for sample in buffer.iter_samples():
        require(n < capacity, "replay buffer yielded more samples than it reports")
        require(sample["schema"] == "r02-sample-v1" and sample["history_complete"] is True
                and digest(sample["contracts"]) == _CONTRACTS, "unexpected replay sample schema/contracts")
        if checkpoint_sha256 is not None:
            require(sample["checkpoint"]["sha256"] == checkpoint_sha256, "replay sample played by another checkpoint")
        position = sample["position"]
        ply = position["ply"]
        require(type(ply) is int and ply == len(sample["moves_uci"]), "sample ply does not match its moves")
        state = make_state(sample["initial_fen"], sample["moves_uci"], history_complete=True)
        encoded = encode_position(state).features
        require(encoded.dtype == np.float32 and encoded.shape == FEATURE_SHAPE, "unexpected encoded features")
        features[n] = encoded
        mask = np.array(position["legal_mask"], dtype=np.bool_)
        policy = np.array(position["visit_policy"], dtype=np.float32)
        require(mask.shape == (MOVE_COUNT,) and policy.shape == (MOVE_COUNT,) and mask.any()
                and not policy[~mask].any(), "visit policy must have legal support only")
        legal = np.flatnonzero(mask)
        policy_indices.append(legal.astype(np.int16))
        policy_values.append(policy[legal])
        lengths.append(len(legal))
        masks[n] = np.packbits(mask)
        target = np.array(sample["wdl"], dtype=np.float32)
        require(target.shape == (3,) and bool(((target == 0) | (target == 1)).all()) and target.sum() == 1,
                "self-play WDL must be an exact one-hot final result")
        wdl[n] = target
        sample_ids.append(sample["sample_id"])
        game_ids.append(sample["game_id"])
        plies.append(ply)
        is_protected.append(rule_position(state.board) in protected)
        game = games.setdefault(sample["game_id"], dict(initial_fen=sample["initial_fen"], plies=[], last=None))
        require(game["initial_fen"] == sample["initial_fen"], "one game with two initial positions")
        game["plies"].append(ply)
        if game["last"] is None or ply > game["last"][0]:
            game["last"] = (ply, list(sample["moves_uci"]) + [position["move_uci"]])
        n += 1
    require(n == capacity, "replay buffer yielded fewer samples than it reports")
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    rows = dict(features=features, legal_mask=masks, wdl=wdl, policy_offsets=offsets,
                policy_indices=np.concatenate(policy_indices) if n else np.zeros(0, np.int16),
                policy_values=np.concatenate(policy_values) if n else np.zeros(0, np.float32),
                sample_ids=sample_ids, game_ids=game_ids, plies=np.array(plies, dtype=np.int32),
                protected=np.array(is_protected, dtype=np.bool_))
    return rows, games


def classify_games(games, held_games):
    """Replay game id -> blocked (canonical identity is a held-out Stockfish game), after full streaming.

    Each game's full move list is the highest-ply sample's moves plus its own move; it must
    reproduce the replay game id, and the sampled plies must be contiguous up to the last move.
    """
    blocked = {}
    for gid, game in games.items():
        plies = sorted(game["plies"])
        last, moves = game["last"]
        require(plies == list(range(plies[0], last + 1)) and len(moves) == last + 1,
                f"game {gid}: sampled plies are not contiguous to the final move")
        require(digest([game["initial_fen"], moves]) == gid, f"game {gid}: rebuilt moves do not match its id")
        blocked[gid] = canonical_game_id(canonical_payload(game["initial_fen"], moves)) in held_games
    return blocked


def split_rows(rows, blocked, *, holdout=held_out):
    """Row selections (train, holdout) and every exclusion count."""
    game_ids = rows["game_ids"]
    blocked_rows = np.array([blocked[g] for g in game_ids], dtype=np.bool_)
    keep = ~blocked_rows & ~rows["protected"]
    hold_games = {g for g in blocked if not blocked[g] and holdout is not None and holdout(g)}
    hold_rows = np.array([g in hold_games for g in game_ids], dtype=np.bool_) & keep
    train_rows = keep & ~hold_rows
    train = np.flatnonzero(train_rows)
    hold = np.flatnonzero(hold_rows)
    counts = dict(
        games=len(blocked), positions=len(game_ids),
        blocked_games=sum(blocked.values()), blocked_game_positions=int(blocked_rows.sum()),
        protected_positions=int(rows["protected"].sum()),
        protected_positions_outside_blocked_games=int((rows["protected"] & ~blocked_rows).sum()),
        holdout_games=len(hold_games), holdout_positions=len(hold),
        training_games=len({game_ids[i] for i in train}), training_positions=len(train))
    return train, hold, counts


def _set_arrays(rows, selection):
    lengths = np.diff(rows["policy_offsets"])[selection]
    offsets = np.zeros(len(selection) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    parts = [np.arange(rows["policy_offsets"][i], rows["policy_offsets"][i + 1]) for i in selection]
    take = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return dict(features=rows["features"][selection], legal_mask=rows["legal_mask"][selection],
                wdl=rows["wdl"][selection], policy_offsets=offsets,
                policy_indices=rows["policy_indices"][take], policy_values=rows["policy_values"][take])


SET_ARRAYS = {  # name: (dtype, trailing shape or None for one-dimensional)
    "features": ("float32", FEATURE_SHAPE), "legal_mask": ("uint8", (_MASK_BYTES,)), "wdl": ("float32", (3,)),
    "policy_offsets": ("int64", ()), "policy_indices": ("int16", ()), "policy_values": ("float32", ()),
}


def planned_bytes(counts, nnz):
    """Exact bytes of the arrays a set of the given counts writes (rows.json excluded)."""
    total = 0
    for name, n in counts.items():
        total += array_bytes((n, *FEATURE_SHAPE), np.float32) + array_bytes((n, _MASK_BYTES), np.uint8)
        total += array_bytes((n, 3), np.float32) + array_bytes((n + 1,), np.int64)
        total += array_bytes((nnz[name],), np.int16) + array_bytes((nnz[name],), np.float32)
    return total


def _write_array(path, array):
    with open(path, "xb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path, text):
    with open(path, "x") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def build_round_data(output, *, buffer, prepared, identity, checkpoint_sha256, source, require_free=None,
                     extra_bytes=0, holdout=held_out, timings=None):
    """Stream, filter and publish the round's self-play arrays once; return the opened data.

    buffer is an audited ReplayBuffer; identity the supervised dataset's verified source identity
    (its authoritative registries decide the held-out filtering). source is recorded verbatim
    (pointer and buffer verification). require_free(bytes, what) raises if the bytes do not fit;
    extra_bytes is what else the jobs will write (checkpoints plus reserve). holdout=None disables
    the self-play hold-out and exists only for equivalence tests against MixedSources.
    """
    import time
    output = Path(output)
    require(not output.exists(), f"{output} exists; round data is never rewritten")
    held_games, protected, registries = read_registries(prepared, identity)
    started = time.perf_counter()
    rows, games = stream_rows(buffer, protected=protected, checkpoint_sha256=checkpoint_sha256)
    stream_seconds = time.perf_counter() - started
    blocked = classify_games(games, held_games)
    train, hold, counts = split_rows(rows, blocked, holdout=holdout)
    require(len(train) > 0, "no self-play training rows remain after filtering")
    for name in REGISTRIES:
        require(file_identity(Path(prepared) / name) == registries[name], f"held-out registry changed while loading: {name}")
    selections = dict(train=train, holdout=hold)
    lengths = np.diff(rows["policy_offsets"])
    nnz = {name: int(lengths[sel].sum()) for name, sel in selections.items()}
    need = planned_bytes({name: len(sel) for name, sel in selections.items()}, nnz)
    if require_free is not None:
        require_free(need + int(extra_bytes), "round data arrays plus planned checkpoints and reserve")
    started = time.perf_counter()
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    partial.mkdir()
    try:
        sets = {}
        for name, selection in selections.items():
            directory = partial / name
            directory.mkdir()
            arrays = _set_arrays(rows, selection)
            files = {}
            for array_name, array in arrays.items():
                path = directory / f"{array_name}.npy"
                _write_array(path, array)
                files[array_name] = dict(file=path.name, dtype=str(array.dtype), shape=list(array.shape),
                                         bytes=path.stat().st_size, sha256=_sha256(path))
            index = dict(sample_ids=[rows["sample_ids"][i] for i in selection],
                         game_ids=[rows["game_ids"][i] for i in selection],
                         plies=[int(rows["plies"][i]) for i in selection])
            path = directory / "rows.json"
            _write_text(path, json.dumps(index, separators=(",", ":")) + "\n")
            files["rows"] = dict(file=path.name, bytes=path.stat().st_size, sha256=_sha256(path))
            _fsync_dir(directory)
            sets[name] = dict(count=len(selection), nonzero_policy_entries=nnz[name], games=len(set(index["game_ids"])),
                              files=files)
        write_seconds = time.perf_counter() - started
        manifest = dict(version=DATA_VERSION, created_utc=_now(), source=source, checkpoint_sha256=checkpoint_sha256,
            contracts=asdict(AdapterConfig()),
            encoding="encode_position(make_state(initial_fen, moves_uci, history_complete=True)).features",
            policy_storage="CSR over every legal move: policy_offsets, policy_indices (move ids), policy_values",
            legal_mask_storage="numpy packbits over 4352 move ids, big-endian bit order",
            row_order="replay buffer sample-id order",
            registries=registries, supervised_source_id=digest(identity),
            filtering=FILTER_DESCRIPTION, holdout=dict(rule=HOLDOUT_RULE if holdout is held_out else None,
                                                     description=HOLDOUT_DESCRIPTION if holdout is held_out else "disabled"),
            counts=counts, sets=sets,
            timings=dict(timings or {}, stream_seconds=stream_seconds, write_seconds=write_seconds))
        _write_text(partial / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _fsync_dir(partial)
        require(not output.exists(), f"{output} appeared while building")
        os.rename(partial, output)
        _fsync_dir(output.parent)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return RoundData.open(output, registries=registries)


class RoundRows:
    """One verified, memory-mapped set of compact self-play rows."""

    def __init__(self, arrays, index):
        self._arrays = arrays
        self.sample_ids = tuple(index["sample_ids"])
        self.game_ids = tuple(index["game_ids"])
        self.plies = tuple(index["plies"])

    def __len__(self):
        return len(self.sample_ids)

    def batch(self, indices):
        """Dense batch identical to MixedSources.batches' self-play half for the same samples."""
        indices = tuple(indices)
        require(bool(indices) and all(type(i) is int and 0 <= i < len(self) for i in indices),
                "batch indices must be nonempty valid integers")
        order = np.asarray(indices, dtype=np.int64)
        a = self._arrays
        features = np.array(a["features"][order], dtype=np.float32, copy=True)
        mask = np.unpackbits(np.asarray(a["legal_mask"][order]), axis=1).astype(np.bool_)
        policy = np.zeros((len(indices), MOVE_COUNT), dtype=np.float32)
        for row, i in enumerate(indices):
            start, stop = int(a["policy_offsets"][i]), int(a["policy_offsets"][i + 1])
            moves = np.asarray(a["policy_indices"][start:stop], dtype=np.int64)
            require(len(moves) == int(mask[row].sum()) and bool(mask[row, moves].all()),
                    "stored policy support differs from the legal mask")
            policy[row, moves] = a["policy_values"][start:stop]
        wdl = np.array(a["wdl"][order], dtype=np.float32, copy=True)
        return SelfPlayBatch(tuple(self.sample_ids[i] for i in indices), features, mask, policy, wdl)


@dataclass(frozen=True)
class RoundData:
    directory: Path
    manifest: dict
    replay_id: str
    train: RoundRows
    holdout: RoundRows

    @classmethod
    def open(cls, directory, *, registries=None):
        """Open after checking every file's size and hash, every shape, and (if given) the registries."""
        directory = Path(directory)
        manifest = read_json(directory / "manifest.json")
        require(manifest.get("version") == DATA_VERSION, "round data version mismatch")
        require(digest(manifest["contracts"]) == _CONTRACTS, "round data adapter contracts mismatch")
        if registries is not None:
            require(digest(manifest["registries"]) == digest(registries), "round data held-out registries mismatch")
        sets = {}
        for name in ("train", "holdout"):
            record = manifest["sets"][name]
            files = record["files"]
            require(set(files) == set(SET_ARRAYS) | {"rows"}, f"round data {name}: unexpected files")
            arrays = {}
            for array_name, record_file in files.items():
                path = directory / name / record_file["file"]
                require(path.is_file() and path.stat().st_size == record_file["bytes"]
                        and _sha256(path) == record_file["sha256"], f"round data file {path} is missing or altered")
                if array_name == "rows":
                    continue
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                dtype, trailing = SET_ARRAYS[array_name]
                require(str(array.dtype) == dtype == record_file["dtype"] and list(array.shape) == record_file["shape"]
                        and array.shape[1:] == tuple(trailing), f"round data {path}: unexpected dtype/shape")
                arrays[array_name] = array
            index = read_json(directory / name / "rows.json")
            n = record["count"]
            require(len(index["sample_ids"]) == len(index["game_ids"]) == len(index["plies"]) == n
                    and len(arrays["features"]) == len(arrays["legal_mask"]) == len(arrays["wdl"]) == n
                    and len(arrays["policy_offsets"]) == n + 1, f"round data {name}: inconsistent counts")
            offsets = arrays["policy_offsets"]
            require(offsets[0] == 0 and bool((np.diff(offsets) > 0).all())
                    and offsets[-1] == len(arrays["policy_indices"]) == len(arrays["policy_values"]),
                    f"round data {name}: invalid policy offsets")
            require(len(set(index["sample_ids"])) == n, f"round data {name}: repeated sample ids")
            sets[name] = RoundRows(arrays, index)
        require(not set(sets["train"].game_ids) & set(sets["holdout"].game_ids), "a game is in both sets")
        return cls(directory, manifest, digest(manifest), sets["train"], sets["holdout"])


# Sampler.

class SamplerSources:
    """The identities MixedState.initial reads; no data access."""

    def __init__(self, *, supervised_count, supervised_source_id, supervised_membership_sha256, replay_id,
                 replay_membership):
        self.dataset = range(supervised_count)
        self.supervised_source_id = supervised_source_id
        self.supervised_membership_sha256 = supervised_membership_sha256
        self.replay_id = replay_id
        self.replay_membership = tuple(replay_membership)


def supervised_identity(dataset):
    """(source id, membership digest): MixedSources' source id; the cache manifest's arrays when cached."""
    manifest = getattr(dataset, "manifest", None)
    membership = digest(manifest["arrays"]) if isinstance(manifest, dict) else _supervised_membership(dataset)
    return digest(dataset.source_identity), membership


def sampler_sources(dataset, data):
    source_id, membership = supervised_identity(dataset)
    return SamplerSources(supervised_count=len(dataset), supervised_source_id=source_id,
                          supervised_membership_sha256=membership, replay_id=data.replay_id,
                          replay_membership=data.train.sample_ids)


def sampler_identity(state):
    return dict(supervised_source_id=state.supervised_source_id,
                supervised_membership_sha256=state.supervised_membership_sha256,
                supervised_count=len(state.supervised.order), replay_id=state.replay_id,
                replay_membership_sha256=digest(list(state.replay_membership)))


# One update.

def round_step(model, optimizer, *, recipe, supervised, rows, state, max_gradient_norm):
    """One optimizer update of either recipe; returns (metrics record, next sampler state).

    Both recipes take the sampler's next selection, so their self-play order is the same. Input,
    loss and gradient failures happen before optimizer.step() and do not advance the sampler; a
    failure in or after the step requires restoring the last checkpoint.
    """
    require(recipe in RECIPES, f"unknown recipe {recipe!r}")
    require(type(state) is MixedState, "MixedState sampler required")
    require(math.isfinite(max_gradient_norm) and max_gradient_norm > 0, "invalid max_gradient_norm")
    require(type(model) is ChessNetwork and all(layer.self_attn.relative is not None
            for layer in model.backbone.layers), "round training requires the combined M06 network")
    parameters = list(model.parameters())
    device, dtype = parameters[0].device, parameters[0].dtype
    require(device.type in ("cpu", "cuda") and all(p.device == device and p.dtype == dtype for p in parameters)
            and finite_tensors(parameters), "finite uniform supported parameters required")
    validate_optimizer_settings(_optimizer_payload(model, optimizer), state.settings)
    require(len(optimizer.state) == (len(parameters) if state.updates else 0)
            and all(optimizer.state[p]["step"].item() == state.updates for p in optimizer.state),
            "optimizer/sampler update progress mismatch")
    si, ri, next_state = state.next_selection()
    replay = rows.batch(ri)
    nr = len(ri)
    batches = [(replay, nr, "policy_distributions")]
    sup = None
    if recipe == "mix":
        sup = supervised.batch(si)
        require(sup.split == "train", "Stockfish rows must come from the training split")
        batches.insert(0, (sup, len(si), "policy_classes"))
    for batch, size, target in batches:
        for name, shape, kind in (("features", (size, *FEATURE_SHAPE), np.float32),
                ("legal_mask", (size, MOVE_COUNT), np.bool_), ("wdl", (size, 3), np.float32),
                (target, (size,) if target == "policy_classes" else (size, MOVE_COUNT),
                 np.int64 if target == "policy_classes" else np.float32)):
            value = getattr(batch, name)
            require(type(value) is np.ndarray and value.shape == shape and value.dtype == kind,
                    f"invalid {name} shape/dtype")
        require(bool(np.isfinite(batch.features).all()), "features must be finite")
    r = {name: torch.tensor(getattr(replay, name), device=device) for name in ("legal_mask", "policy_distributions", "wdl")}
    if sup is not None:
        ns = len(si)
        tensors = sup.to_torch(device=device)
        validate_policy_classes(tensors["legal_mask"], tensors["policy_classes"])
    validate_policy_distribution(r["legal_mask"], r["policy_distributions"])
    if sup is not None:
        validate_soft_wdl(tensors["wdl"], torch.zeros((ns, 3), dtype=dtype, device=device))
    validate_soft_wdl(r["wdl"], torch.zeros((nr, 3), dtype=dtype, device=device))
    require(bool(((r["wdl"] == 0) | (r["wdl"] == 1)).all()), "exact final WDL required")
    if sup is not None:
        features = torch.tensor(np.concatenate((sup.features, replay.features)), dtype=dtype, device=device)
    else:
        features = torch.tensor(replay.features, dtype=dtype, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        policy, wdl = model(features)
        if sup is not None:
            loss, metrics = mixed_losses(policy, wdl, supervised_mask=tensors["legal_mask"],
                classes=tensors["policy_classes"], teacher_wdl=tensors["wdl"], selfplay_mask=r["legal_mask"],
                visits=r["policy_distributions"], final_wdl=r["wdl"], settings=state.settings)
            losses = dict(stockfish=dict(policy_loss=metrics.supervised.policy_loss, wdl_loss=metrics.supervised.wdl_loss,
                                         total_loss=metrics.supervised.total_loss),
                          selfplay=dict(policy_loss=metrics.selfplay.policy_loss, wdl_loss=metrics.selfplay.wdl_loss,
                                        total_loss=metrics.selfplay.total_loss),
                          stockfish_contribution=metrics.supervised_contribution,
                          selfplay_contribution=metrics.selfplay_contribution, total_loss=metrics.total_loss)
        else:
            require(policy.shape == (nr, MOVE_COUNT) and wdl.shape == (nr, 3), "logit shape must match the batch")
            validate_soft_wdl(r["wdl"], wdl)
            rp = policy_distribution_cross_entropy(policy, r["legal_mask"], r["policy_distributions"])
            rw = soft_wdl_cross_entropy(wdl, r["wdl"])
            loss = state.settings.policy_weight * rp + state.settings.wdl_weight * rw
            require(bool(torch.isfinite(loss)), "self-play loss must be finite")
            value = float(loss.detach())
            losses = dict(selfplay=dict(policy_loss=float(rp.detach()), wdl_loss=float(rw.detach()), total_loss=value),
                          total_loss=value)
        loss.backward()
        require(all(p.grad is not None for p in parameters) and finite_tensors(p.grad for p in parameters),
                "every combined-model parameter requires a finite gradient")
        norm, clipped = clip_gradients(parameters, max_gradient_norm)
        optimizer.step()
        require(finite_tensors(parameters), "nonfinite optimizer result")
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    record = dict(event="update", update=next_state.updates, recipe=recipe, losses=losses,
                  gradient_norm=float(norm), clipped=bool(clipped),
                  trained_positions=dict(selfplay=nr, stockfish=len(si) if sup is not None else 0),
                  discarded_sampler_indices=0 if sup is not None else len(si),
                  selfplay_indices=[int(i) for i in ri], selfplay_epoch=next_state.selfplay.epoch)
    record["stockfish_indices" if sup is not None else "discarded_stockfish_indices"] = [int(i) for i in si]
    return record, next_state


# Model, optimizer and checkpoints.

def build_round_optimizer(model):
    """AdamW grouped exactly as P's (build_optimizer): decay on matrices only, foreach/fused off."""
    return build_optimizer(model, OPTIMIZER_SETTINGS)


def optimizer_configuration(config):
    """Optimizer kind, defaults and groups a fresh round optimizer records (no moments)."""
    model = replace(config, device="cpu").build()
    payload = _optimizer_payload(model, build_round_optimizer(model))
    return {k: v for k, v in payload.items() if k != "state"}


def recipe_provenance(recipe):
    """What the recipe trains on per update; the only place reporting reads the recipe from."""
    require(recipe in RECIPES, f"unknown recipe {recipe!r}")
    ns, nr = SETTINGS.supervised_batch_size, SETTINGS.selfplay_batch_size
    if recipe == "mix":
        return dict(recipe="mix", name="R1-mix", description=f"{nr} self-play + {ns} Stockfish rows per update; "
                    "loss 0.75 x self-play + 0.25 x Stockfish (mixed_losses)",
                    trained_positions_per_update=dict(selfplay=nr, stockfish=ns), discarded_sampler_indices_per_update=0,
                    loss=dict(selfplay_weight=1 - SETTINGS.supervised_fraction, stockfish_weight=SETTINGS.supervised_fraction))
    return dict(recipe="pure", name="R1-pure", description=f"{nr} self-play rows per update; loss = self-play loss",
                trained_positions_per_update=dict(selfplay=nr, stockfish=0), discarded_sampler_indices_per_update=ns,
                loss=dict(selfplay_weight=1.0, stockfish_weight=0.0))


SAMPLER_NOTE = ("mixed_state is the shared self-play sampler (seed 20261017, composition 8 Stockfish + 24 self-play). "
                "For R1-pure its 8 Stockfish indices per update are discarded, so its counts and the checkpoint "
                "progress samples_seen (32 per update) are sampler bookkeeping, not trained positions; the recipe "
                "is this provenance's recipe field.")


@dataclass(frozen=True)
class RoundCheckpoint:
    model: ChessNetwork
    optimizer: torch.optim.Optimizer
    config: ModelConfig
    checkpoint_id: str
    provenance: dict
    updates: int
    state: MixedState


def restore_round(path, *, expected_config, expected_provenance, expected_settings, expected_identity):
    """Resume a round checkpoint with restore_training's safeguards; mixed sources are not required.

    Checks provenance, model configuration, loss weights, optimizer configuration, sampler
    settings and identities, and the exact CPU (and CUDA) runtime before any state is
    installed. The model and Adam moments follow the recorded device; Adam step counters must
    equal the update count. RNG (CUDA, Python, NumPy, Torch) is installed last, so a failed
    restore leaves the caller's RNG unchanged.
    """
    try:
        _require(type(expected_config) is ModelConfig, "restore requires an explicit expected ModelConfig")
        _require(type(expected_settings) is MixedSettings, "restore requires explicit MixedSettings")
        payload = _read_payload(path)
        _require(payload.get("version") in (MIXED_CHECKPOINT_VERSION, CUDA_MIXED_CHECKPOINT_VERSION),
                 "round checkpoints use the mixed checkpoint format")
        _same(payload.get("provenance"), expected_provenance, "round provenance")
        model, _, config, weights, progress, state, numpy_state = _validate_payload(payload, expected_config)
        _require(type(state) is MixedState, "round checkpoint requires mixed sampler state")
        _same(asdict(state.settings), asdict(expected_settings), "sampler settings")
        _same(asdict(weights), asdict(expected_settings.weights), "loss weights")
        _same({k: v for k, v in payload["training"]["optimizer"].items() if k != "state"},
              optimizer_configuration(config), "optimizer configuration")
        _same(sampler_identity(state), expected_identity, "sampler sources")
        _same(payload["training"]["runtime"], runtime_config(), "resumption runtime")
        cuda = config.device == "cuda:0"
        if cuda:
            _same(payload["training"]["cuda_runtime"], cuda_runtime_config(), "CUDA resumption runtime")
            cuda_state = payload["training"]["rng"]["cuda"]["state"]
            torch.Generator(device="cuda:0").set_state(cuda_state)
            model.to("cuda:0")
        optimizer = _restore_optimizer(model, payload["training"]["optimizer"], progress)
        for parameter, values in optimizer.state.items():
            _require(values["step"].device.type == "cpu" and values["step"].item() == progress.step,
                     "Adam step must stay on CPU and equal the update count")
            _require(all(t.device == parameter.device for n, t in values.items() if n != "step"),
                     "Adam moments must follow the model device")
        _require(len(optimizer.state) == (len(list(model.parameters())) if progress.step else 0),
                 "Adam state must cover every parameter after the first update")
        model.train(payload["model_training"])
        model._resume_runtime_history = []
        if cuda:
            torch.cuda.set_rng_state(cuda_state, 0)
        random.setstate(payload["training"]["rng"]["python"])
        np.random.set_state(numpy_state)
        torch.set_rng_state(payload["training"]["rng"]["torch"])
        return RoundCheckpoint(model, optimizer, config, payload["checkpoint_id"],
                               json.loads(json.dumps(payload["provenance"])), progress.step, state)
    except Exception as error:
        raise CheckpointError(f"cannot restore round checkpoint {path}: {error}") from error


# Evaluation.

def validation_rows(model, dataset, batch_size):
    """Per-position policy CE, WDL CE and top-1 on a supervised split, in source order, eval mode."""
    mode = model.training
    model.eval()
    device = next(model.parameters()).device
    policy, wdl, correct = [], [], []
    try:
        with torch.inference_mode():
            for batch in dataset.batches(batch_size):
                t = batch.to_torch(device=device)
                p, w = model(t["features"])
                logp = legal_policy_log_probs(p, t["legal_mask"])
                policy.append((-logp.gather(1, t["policy_classes"][:, None])[:, 0]).double().cpu().numpy())
                lw = torch.log_softmax(w, dim=1)
                supported = torch.where(t["wdl"] > 0, lw, torch.zeros_like(lw))
                wdl.append((-(t["wdl"] * supported).sum(dim=1)).double().cpu().numpy())
                best = p.masked_fill(~t["legal_mask"], -torch.inf).argmax(1)
                correct.append((best == t["policy_classes"]).double().cpu().numpy())
    finally:
        model.train(mode)
    policy, wdl, correct = (np.concatenate(v) for v in (policy, wdl, correct))
    return dict(policy=policy, wdl=wdl, total=policy + wdl, correct=correct)


def holdout_rows(model, rows, batch_size):
    """Per-position visit-distribution CE and WDL CE on self-play rows, eval mode."""
    mode = model.training
    model.eval()
    parameter = next(model.parameters())
    policy, wdl = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                b = rows.batch(range(start, min(start + batch_size, len(rows))))
                features = torch.tensor(b.features, dtype=parameter.dtype, device=parameter.device)
                mask = torch.tensor(b.legal_mask, device=parameter.device)
                visits = torch.tensor(b.policy_distributions, device=parameter.device)
                target = torch.tensor(b.wdl, device=parameter.device)
                p, w = model(features)
                logp = legal_policy_log_probs(p, mask)
                supported = torch.where(visits > 0, logp, torch.zeros_like(logp))
                policy.append((-(visits * supported).sum(dim=1)).double().cpu().numpy())
                lw = torch.log_softmax(w, dim=1)
                supported = torch.where(target > 0, lw, torch.zeros_like(lw))
                wdl.append((-(target * supported).sum(dim=1)).double().cpu().numpy())
    finally:
        model.train(mode)
    if not policy:
        return dict(policy=np.zeros(0), wdl=np.zeros(0), total=np.zeros(0))
    policy, wdl = np.concatenate(policy), np.concatenate(wdl)
    return dict(policy=policy, wdl=wdl, total=policy + wdl)


def clustered_bootstrap(values, groups, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, confidence=0.95):
    """Mean of per-position values with a percentile interval from resampling whole games."""
    values = np.asarray(values, dtype=np.float64)
    require(len(values) == len(groups), "one group label per value required")
    if len(values) == 0:
        return dict(mean=None, low=None, high=None, games=0, positions=0)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    stats = np.empty(resamples)
    for start in range(0, resamples, 250):
        picks = rng.integers(0, len(sums), size=(min(250, resamples - start), len(sums)))
        stats[start:start + len(picks)] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(stats, [(1 - confidence) / 2, (1 + confidence) / 2])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), games=int(len(sums)),
                positions=int(len(values)), resamples=resamples, seed=seed, confidence=confidence,
                method="game-clustered percentile bootstrap")


UNCERTAINTY_NOTE = ("intervals are uncertainty across validation games (game-clustered percentile bootstrap); "
                    "evaluation on the fixed set is deterministic")


def paired_labels(interval, *, flag=FLAG_DELTA):
    """Labels on a paired (checkpoint minus P) validation-loss interval; they never change flag or stop.

    increase_supported: the interval's lower end is above 0, evidence of increased loss on the
    validation-game population. increase_above_flag_supported: its lower end is above the flag
    threshold, evidence that the increase exceeds it. [0.02, 0.07] supports the first, not the second.
    """
    low = interval["low"]
    require(type(low) is float and math.isfinite(low), "paired interval needs a finite lower end")
    return dict(paired_ci_low=low, paired_ci_high=interval["high"], confidence=interval["confidence"],
                increase_supported=bool(low > 0), increase_above_flag_supported=bool(low > flag), flag=flag,
                meaning=dict(increase_supported="paired_ci_low > 0: evidence of increased loss on the validation-game population",
                             increase_above_flag_supported=f"paired_ci_low > {flag}: evidence that the increase exceeds the flag threshold"),
                rule="labels only; the flag (measured delta > +0.05) and the stop (measured delta > +0.25) apply to "
                     "the measured delta regardless of either label",
                interval=UNCERTAINTY_NOTE)


def describe_values(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), max=float(values.max()),
                quantiles={str(q): float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))})


def rows_digest(rows):
    """Exact identity of per-position evaluation arrays (float64 bytes)."""
    h = hashlib.sha256()
    for name in sorted(rows):
        h.update(name.encode())
        h.update(np.ascontiguousarray(rows[name], dtype=np.float64).tobytes())
    return h.hexdigest()


def select_candidate(evaluations, *, completed, flag=FLAG_DELTA, stop=STOP_DELTA):
    """Candidate of a run from its checkpoint evaluations, in update order.

    evaluations: dicts with update, checkpoint and delta_total (checkpoint total validation loss
    minus P's, P measured in the same job). Hard stop at the first delta above `stop`; the
    candidate is then the last earlier checkpoint with delta at most `stop`, or P if none. A
    completed run's candidate is its final checkpoint, flagged if its delta exceeds `flag`.
    """
    previous = None
    for evaluation in evaluations:
        if evaluation["delta_total"] > stop:
            chosen = previous
            return dict(stopped=True, stopped_at=evaluation["update"], stop_delta=evaluation["delta_total"],
                        candidate="P" if chosen is None else chosen["checkpoint"],
                        candidate_update=0 if chosen is None else chosen["update"],
                        reason=(f"hard stop: delta {evaluation['delta_total']:+.4f} > +{stop} at update "
                                f"{evaluation['update']}; " + ("no earlier checkpoint is eligible, so P" if chosen is None
                                else f"last checkpoint with delta <= +{stop}")),
                        final_flag=None)
        previous = evaluation
    require(completed and previous is not None, "a run that neither completed nor stopped has no candidate")
    return dict(stopped=False, stopped_at=None, stop_delta=None, candidate=previous["checkpoint"],
                candidate_update=previous["update"], reason="training completed; final checkpoint",
                final_flag=bool(previous["delta_total"] > flag))


def exploratory_candidate(evaluations, *, completed, flag=FLAG_DELTA, stop=STOP_DELTA):
    """Candidate of a run trained without the hard stop: its final checkpoint, with every breach listed.

    The original rule's candidate (select_candidate on the same evaluations) is reported alongside it;
    finishing the run tests a new hypothesis and does not change that result.
    """
    require(completed and evaluations, "an exploratory run has a candidate only once it completes")
    final = evaluations[-1]
    breaches = [e["update"] for e in evaluations if e["delta_total"] > stop]
    return dict(stopped=False, stopped_at=None, stop_delta=None, candidate=final["checkpoint"],
                candidate_update=final["update"], exploratory=True, threshold_breaches=breaches,
                reason=("exploratory run without the hard stop: final checkpoint; "
                        f"{len(breaches)} checkpoint(s) exceeded +{stop}"),
                final_flag=bool(final["delta_total"] > flag))


def stopped_training(sidecar):
    """Whether training ended at this checkpoint; older sidecars only record the breach."""
    return sidecar.get("training_stopped", sidecar["hard_stop"])


# Run directory: checkpoint/sidecar pairs and metrics segments.

def checkpoint_name(update):
    return f"update-{update:06d}.pt"


def sidecar_name(update):
    return f"update-{update:06d}.json"


def _update_of(path):
    stem = Path(path).name.split(".")[0]
    require(stem.startswith("update-") and stem[7:].isdigit() and len(stem) == 13, f"unexpected run file {path}")
    return int(stem[7:])


def resume_point(directory):
    """Latest complete checkpoint+sidecar pair and the checkpoints after it that have no sidecar.

    A sidecar names its checkpoint's sha256; a mismatch or a sidecar without its checkpoint is
    corruption and raises. Checkpoints without a sidecar are allowed only after the latest pair
    (a crash between the two writes).
    """
    directory = Path(directory)
    checkpoints = {_update_of(p): p for p in directory.glob("update-*.pt")}
    sidecars = {_update_of(p): p for p in directory.glob("update-*.json")}
    require(set(sidecars) <= set(checkpoints), "a sidecar has no checkpoint")
    latest = None
    for update in sorted(sidecars):
        record = read_json(sidecars[update])
        require(record.get("version") == SIDECAR_VERSION and record["update"] == update
                and record["checkpoint"] == checkpoint_name(update), f"{sidecars[update]}: malformed sidecar")
        require(_sha256(checkpoints[update]) == record["checkpoint_sha256"],
                f"{checkpoints[update]} does not match its sidecar")
        latest = (update, record)
    orphans = sorted(u for u in checkpoints if u not in sidecars)
    require(latest is None or all(u > latest[0] for u in orphans), "a checkpoint before the latest pair has no sidecar")
    return latest, [checkpoints[u] for u in orphans]


def set_aside(directory, paths):
    """Move incomplete checkpoints into orphans/ (renamed, never deleted or overwritten)."""
    moved = []
    if not paths:
        return moved
    target = Path(directory) / "orphans"
    target.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for path in paths:
        destination = target / f"{Path(path).name}.orphan-{stamp}"
        require(not destination.exists(), f"{destination} exists")
        os.rename(path, destination)
        moved.append(dict(checkpoint=Path(path).name, moved_to=str(destination.relative_to(directory))))
    _fsync_dir(directory)
    _fsync_dir(target)
    return moved


def segments(directory):
    paths = sorted(Path(directory).glob("metrics-*.jsonl"))
    for number, path in enumerate(paths, 1):
        require(path.name.startswith(f"metrics-{number:03d}-from-"), f"unexpected metrics segment {path.name}")
    return paths


def read_metrics(directory):
    """Update records reconstructed across segments.

    Each segment opens with a header naming the update it resumed from; records after that
    update in earlier segments are superseded (their updates were redone). A last line without
    a newline is a crashed partial write and is ignored. Returns (records, superseded, partial).
    """
    records, superseded, partial = [], 0, 0
    for path in segments(directory):
        text = path.read_text()
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        elif lines:
            partial += 1
            lines.pop()
        for number, line in enumerate(lines):
            record = json.loads(line)
            if record["event"] == "segment":
                require(number == 0, f"{path.name}: header must be the first line")
                start = record["from_update"]
                require(len(records) >= start, f"{path.name}: resumes after missing updates")
                superseded += len(records) - start
                records = records[:start]
            else:
                require(record["event"] == "update" and record["update"] == len(records) + 1,
                        f"{path.name}: update records out of order")
                records.append(record)
    return records, superseded, partial


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value
