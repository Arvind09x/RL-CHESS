"""D07: completed Stockfish bundles -> replayed adapter samples -> NumPy batches.

Opening validates every label before split filtering. Files are read-only; no
engine, model, cache, losses or training is involved. Torch is imported only by
SupervisedBatch.to_torch(). The dataset retains compact records and joined games,
not encoded tensors; each access returns newly replayed state and arrays.
"""
from dataclasses import asdict, dataclass
from types import MappingProxyType
from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
from datetime import datetime
import importlib.metadata
from pathlib import Path
from typing import Iterator
from weakref import WeakKeyDictionary

import chess
import chess.engine
import numpy as np

from chess_rl.adapter import ChessState, encode_position, legal_mask, make_state, terminal_result
from chess_rl.canonical_games import SPLITS, canonical_json, load_json
from chess_rl.contracts import FEATURE_SHAPE, MOVE_COUNT
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity
from chess_rl.position_sampling import RECORD_VERSION, phase, rule_position, validate_game
from chess_rl.stockfish_worker import replay_identity, validate_response, verify_artifacts

LABEL_VERSION = "stockfish-label-v1"
REFERENCE_FIELDS = {"record_version", "game_id", "split", "sample_ply", "phase", "rule_position"}
LABEL_FIELDS = (REFERENCE_FIELDS | {
    "root_fen", "history_complete", "available_snapshots", "replay_sha256", "dataset_id",
    "encoding_version", "move_vocab_version", "ending_policy", "wdl_order",
    "teacher_config_id", "requested_settings", "successful_attempt", "target_provenance",
    "best_move_uci", "best_move_id", "teacher_wdl_counts", "teacher_wdl", "returned_search",
    "returned_bestmove_uci"})
BUNDLE_FILES = ("selected.jsonl", "labels.jsonl", "attempts.jsonl", "exclusions.jsonl", "settings.json")


def _equal(actual, expected, name):
    # JSON's bool/int equality must not silently accept true as count/ID/version.
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{name} mismatch")


def _read(path):
    try:
        value = load_json(path.read_bytes())
        canonical_json(value)  # rejects NaN/Infinity throughout, including extras
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error


def _rows(path):
    try:
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, 1):
                try:
                    row = load_json(raw)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a JSON object")
                    yield row
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    except OSError as error:
        raise ValueError(f"{path}: {error}") from error


def _key(ref):
    if (not isinstance(ref, Mapping) or not isinstance(ref.get("game_id"), str)
            or type(ref.get("sample_ply")) is not int or ref["sample_ply"] < 0):
        raise ValueError("Malformed game/ply reference")
    return ref["game_id"], ref["sample_ply"]


def _references(rows):
    result = {}
    for ref in rows:
        key = _key(ref)
        if set(ref) != REFERENCE_FIELDS or ref["record_version"] != RECORD_VERSION or ref["split"] not in SPLITS:
            raise ValueError("Reference schema/version/split mismatch")
        if key in result:
            raise ValueError("Duplicate selected reference")
        result[key] = ref
    return result


def _provenance(manifest, identities, settings, teacher_evidence):
    evidence = _read(teacher_evidence)
    _equal(settings["artifact_identities"], identities, "Bundle artifact provenance")
    _equal(settings["requested_settings"], asdict(manifest.teacher), "Bundle teacher settings")
    teacher = settings["teacher"]
    expected_binary = dict(bytes=evidence["executable_bytes"], sha256=manifest.engine.executable_sha256)
    _equal(evidence["executable_sha256"], manifest.engine.executable_sha256, "Teacher evidence executable")
    _equal(teacher["executable_identity"], expected_binary, "Teacher executable identity")
    _equal(teacher["build_evidence"], file_identity(teacher_evidence), "Teacher build evidence")
    _equal(teacher["embedded_nnue"], evidence["nnue"], "Teacher embedded assets")
    for saved, asset in zip(evidence["nnue"], manifest.engine.embedded_nnue, strict=True):
        _equal({k: saved[k] for k in ("name", "sha256", "source_url")},
               {k: getattr(asset, k) for k in ("name", "sha256", "source_url")}, "Manifest embedded asset")
    if (teacher["python_chess"] != "1.999" or teacher["chess"] != "1.11.2"
            or importlib.metadata.version("python-chess") != "1.999" or chess.__version__ != "1.11.2"):
        raise ValueError("Pinned replay/teacher dependency mismatch")
    result = dict(kind="stockfish-root-wdl-and-pv",
        **{field: identities[name]["sha256"] for field, name in (
            ("manifest_sha256", "manifest"), ("games_sha256", "games.jsonl"),
            ("samples_sha256", "samples.jsonl"), ("overlap_audit_sha256", "overlap-audit.json"))},
        executable_sha256=manifest.engine.executable_sha256, embedded_nnue=evidence["nnue"])
    if "process_policy" in settings:
        if settings["process_policy"] not in ("fresh-process-v1", "persistent-clear-hash-v1"):
            raise ValueError("unsupported teacher process policy")
        result["process_policy"] = settings["process_policy"]
    return result


def _ledger(refs, labels, attempts, exclusions, summary, manifest, max_attempts):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("Invalid bundle retry limit")
    if summary.get("status") != "complete" or summary.get("pending_references") != []:
        raise ValueError("Unfinished annotation bundle")
    by_key = {key: [] for key in refs}
    for attempt in attempts:
        key = _key(attempt["reference"])
        if key not in refs:
            raise ValueError("Attempt is outside selected membership")
        _equal(attempt["reference"], refs[key], "Attempt reference")
        _equal(attempt["requested_settings"], asdict(manifest.teacher), "Attempt settings")
        rows = by_key[key]
        if type(attempt["attempt"]) is not int or attempt["attempt"] != len(rows) + 1 or len(rows) >= max_attempts:
            raise ValueError("Attempt sequence/retry limit mismatch")
        if attempt["status"] not in ("failed", "success") or rows and rows[-1]["status"] == "success":
            raise ValueError("Interrupted or post-success attempt")
        start, end = (datetime.fromisoformat(attempt[k]) for k in ("started_at_utc", "finished_at_utc"))
        if start.utcoffset() is None or end.utcoffset() is None or end < start:
            raise ValueError("Unfinished/invalid attempt timestamps")
        rows.append(attempt)
    labeled = {}
    for label in labels:
        key = _key(label)
        if key in labeled or key not in refs:
            raise ValueError("Duplicate/unselected label")
        _equal({k: label[k] for k in REFERENCE_FIELDS - {"record_version"}},
               {k: v for k, v in refs[key].items() if k != "record_version"}, "Label reference")
        rows = by_key[key]
        if (not rows or rows[-1]["status"] != "success"
                or type(label["successful_attempt"]) is not int
                or label["successful_attempt"] != rows[-1]["attempt"]):
            raise ValueError("Missing successful-attempt linkage")
        labeled[key] = rows[-1]
    excluded = set()
    for row in exclusions:
        key = _key(row["reference"])
        if key not in refs or key in excluded or key in labeled:
            raise ValueError("Invalid/duplicate exclusion membership")
        _equal(row["reference"], refs[key], "Exclusion reference")
        if (row.get("reason") != "attempts_exhausted" or type(row.get("attempts")) is not int
                or row["attempts"] != max_attempts or len(by_key[key]) != max_attempts
                or any(a["status"] != "failed" for a in by_key[key])):
            raise ValueError("Unfinished exclusion linkage")
        excluded.add(key)
    if set(refs) != set(labeled) | excluded:
        raise ValueError("Unfinished selected membership")
    for field, count in (("selected", len(refs)), ("successful_labels", len(labels)),
                         ("attempts", len(attempts)), ("excluded", len(exclusions)),
                         ("failed_attempts", sum(a["status"] == "failed" for a in attempts))):
        _equal(summary.get(field), count, f"Summary {field}")
    return labeled


def _response(state, saved, best_uci):
    """Restore D05 serialized INFO types and reuse its completed-target validator."""
    if (not isinstance(saved, dict) or not isinstance(best_uci, str)
            or not isinstance(saved.get("pv"), list)
            or any(not isinstance(uci, str) for uci in saved["pv"])):
        raise ValueError("Malformed saved search/move/PV")
    info = dict(saved)
    for field in ("wdl", "score"):
        if saved[field]["turn"] not in ("white", "black"):
            raise ValueError(f"Invalid search {field} perspective")
    raw = saved["wdl"]["relative"]
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("Invalid search WDL shape")
    info["wdl"] = chess.engine.PovWdl(chess.engine.Wdl(*raw), saved["wdl"]["turn"] == "white")
    score = saved["score"]
    cp, mate = score["relative_cp"], score["relative_mate"]
    if (cp is None) == (mate is None) or any(v is not None and type(v) is not int for v in (cp, mate)):
        raise ValueError("Invalid saved score")
    info["score"] = chess.engine.PovScore(chess.engine.Cp(cp) if cp is not None else chess.engine.Mate(mate),
                                          score["turn"] == "white")
    info["pv"] = [chess.Move.from_uci(uci) for uci in saved["pv"]]
    return validate_response(state, info, chess.Move.from_uci(best_uci))


def _load_record(label, ref, game, manifest, provenance, attempt):
    if set(label) != LABEL_FIELDS or label["record_version"] != LABEL_VERSION:
        raise ValueError("Label schema/version mismatch")
    for field in ("dataset_id", "encoding_version", "move_vocab_version", "ending_policy", "wdl_order"):
        expected = getattr(manifest, field)
        _equal(label[field], list(expected) if field == "wdl_order" else expected, f"Label {field}")
    _equal(label["teacher_config_id"], manifest.teacher.config_id, "Teacher configuration ID")
    _equal(label["requested_settings"], asdict(manifest.teacher), "Label teacher settings")
    _equal(label["target_provenance"], provenance, "Label target provenance")
    validate_game(game, manifest)
    if ref["split"] != game["split"] or not 0 <= ref["sample_ply"] <= len(game["moves_uci"]):
        raise ValueError("Canonical game split/ply mismatch")
    state = make_state(game["start_fen"], game["moves_uci"][:ref["sample_ply"]], history_complete=True)
    if terminal_result(state) is not None:
        raise ValueError("Terminal supervised root")
    for field, expected in (("root_fen", state.board.fen(en_passant="fen")),
                            ("history_complete", True), ("available_snapshots", min(8, ref["sample_ply"] + 1)),
                            ("replay_sha256", replay_identity(state)), ("rule_position", rule_position(state.board)),
                            ("phase", phase(state.board, ref["sample_ply"]))):
        _equal(label[field], expected, f"Root/history {field}")
    target = _response(state, label["returned_search"], label["best_move_uci"])
    for field, expected in target.items():
        _equal(label[field], expected, f"Teacher target {field}")
    returned = attempt["returned"]
    if (not isinstance(returned, dict)
            or returned.get("completed") is not True or returned.get("hash_reset_completed") is not True
            or returned.get("transport_errors") or "error" in attempt):
        raise ValueError("Unfinished/malformed successful attempt")
    _equal(returned["principal_response"], label["returned_search"], "Attempt principal response")
    _equal(returned["bestmove"], label["best_move_uci"], "Attempt best move")
    _equal(returned["returned_bestmove"], label["best_move_uci"], "UCI best move")
    rows = returned["returned_rows"]
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Malformed returned principal rows")
    principal = [r for r in rows if r.get("multipv") == 1]
    if not principal:
        raise ValueError("Missing completed principal row")
    _equal(principal[-1], label["returned_search"], "Final principal row")
    _equal(returned["engine_id"], dict(name=manifest.engine.uci_name, author=manifest.engine.uci_author), "UCI teacher ID")
    for name, expected in dict(Threads=manifest.teacher.threads, Hash=manifest.teacher.hash_mb,
            MultiPV=1, UCI_ShowWDL=True, UCI_LimitStrength=False, Skill_Level=20,
            SyzygyPath="", Ponder=False, UCI_Chess960=False,
            EvalFile=manifest.engine.embedded_nnue[0].name,
            EvalFileSmall=manifest.engine.embedded_nnue[1].name).items():
        name = "Skill Level" if name == "Skill_Level" else name
        _equal(returned["effective_options"].get(name), expected, f"Effective teacher {name}")
    policy = provenance.get("process_policy", "fresh-process-v1")
    _equal(returned.get("process_policy", "fresh-process-v1"), policy, "Teacher process policy")
    if policy == "persistent-clear-hash-v1" and "uci_sent" not in returned:
        raise ValueError("Persistent teacher requires raw reset/readiness evidence")
    # Early accepted D05 bundles predate raw transport capture. When capture
    # exists it must attest this exact full prefix, reset and requested budget.
    if "uci_sent" in returned:
        sent = returned["uci_sent"]
        initial = state.board.root().fen(en_passant="fen")
        position = "position startpos" if initial == chess.STARTING_FEN else "position fen " + initial
        if state.board.move_stack:
            position += " moves " + " ".join(m.uci() for m in state.board.move_stack)
        go = f"go nodes {manifest.teacher.nodes}"
        persistent = policy == "persistent-clear-hash-v1"
        ending = "isready" if persistent else "quit"
        if (not isinstance(sent, list) or not sent or sent[-1] != ending
                or any(not isinstance(command, str) for command in sent)
                or sent.count(go) != 1 or sent.count(position) != 1
                or "setoption name Clear Hash" not in sent
                or not sent.index("setoption name Clear Hash") < sent.index(position) < sent.index(go)
                or any(command.startswith("go ") and command != go for command in sent)):
            raise ValueError("Attempt raw history/reset/budget mismatch")
        if persistent:
            if (returned.get("process_healthy_after_search") is not True or "ucinewgame" not in sent
                    or not sent.index("ucinewgame") < sent.index("setoption name Clear Hash")
                    or "isready" not in sent[sent.index("setoption name Clear Hash") + 1:sent.index(position)]):
                raise ValueError("Persistent teacher reset/readiness mismatch")
        else:
            _equal(returned.get("engine_exit_code"), 0, "Attempt engine exit")
    return state


@dataclass(frozen=True)
class SupervisedSample:
    reference: tuple[str, int]
    split: str
    state: ChessState
    features: np.ndarray
    legal_mask: np.ndarray
    policy_class: int
    wdl: np.ndarray


@dataclass(frozen=True)
class SupervisedBatch:
    references: tuple[tuple[str, int], ...]
    split: str
    features: np.ndarray  # float32 [B,64,128]
    legal_mask: np.ndarray  # bool [B,4352]
    policy_classes: np.ndarray  # int64 [B]
    wdl: np.ndarray  # float32 [B,3], soft win/draw/loss

    def to_torch(self, *, device="cpu") -> dict:
        """Explicit optional conversion. Returned tensors own independent storage."""
        import torch
        return {name: torch.tensor(getattr(self, name), device=device) for name in
                ("features", "legal_mask", "policy_classes", "wdl")}


def _freeze_metadata(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(v) for v in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("metadata must contain immutable JSON values")
    return value


def _thaw_metadata(value):
    if isinstance(value, Mapping):
        return {k: _thaw_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_metadata(v) for v in value]
    return value


# Decoded boards/arrays are private implementation storage, never dataset attributes.
_decoded_caches = WeakKeyDictionary()


class SupervisedDataset:
    """Use open_dataset(); labels stay in source order within the chosen split."""
    def __init__(self, split, labels, refs, games, *, source_identity=None):
        self._split, self._labels, self._refs, self._games = split, labels, refs, games
        self._source_identity = copy.deepcopy(source_identity)
        self._admission_fingerprint = None

    def _metadata_fingerprint(self):
        checksum = hashlib.sha256()
        values = [self.split, self.source_identity]
        games = set()
        for label in self._labels:
            ref = self._refs[_key(label)]
            values.extend((label, ref))
            games.add(ref["game_id"])
        values.extend(self._games[key] for key in sorted(games))
        for value in values:
            checksum.update(canonical_json(_thaw_metadata(value)).encode())
            checksum.update(b"\n")
        return checksum.hexdigest()

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise ValueError("frozen training dataset cannot be modified; reopen sources")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise ValueError("frozen training dataset cannot be modified; reopen sources")
        object.__delattr__(self, name)

    def freeze(self, *, cache_bytes=64 * 1024 * 1024):
        """Own an immutable training view; bounded decoded-sample LRU, no rescans.

        Only referenced metadata is retained. Each returned sample still owns
        its board and arrays. External artifacts are audited on source opening.
        """
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        if (self._admission_fingerprint is not None
                and self._metadata_fingerprint() != self._admission_fingerprint):
            raise ValueError("dataset metadata changed after source validation; reopen sources")
        refs = {key: self._refs[key] for key in map(_key, self._labels)}
        games = {r["game_id"]: self._games[r["game_id"]] for r in refs.values()}
        result = SupervisedDataset(self.split, _freeze_metadata(self._labels),
            _freeze_metadata(refs), _freeze_metadata(games), source_identity=self.source_identity)
        result._source_identity = _freeze_metadata(result._source_identity)
        # Admission/freezing performs the complete decoded-target check once.
        for i in range(len(result)):
            sample = result[i]
            if (sample.split != result.split or type(sample.policy_class) is not int
                    or not 0 <= sample.policy_class < MOVE_COUNT
                    or not sample.legal_mask[sample.policy_class]
                    or sample.wdl.shape != (3,) or not np.isfinite(sample.wdl).all()
                    or np.any(sample.wdl < 0) or not np.isclose(sample.wdl.sum(), 1)):
                raise ValueError("invalid frozen sample target")
        _decoded_caches[result] = [OrderedDict(), cache_bytes, 0]
        result._admission_fingerprint = result._metadata_fingerprint()
        result._sealed = True
        return result

    @property
    def cache_info(self):
        """Read-only (entries, payload estimate, budget); no storage exposure."""
        info = _decoded_caches.get(self)
        return (len(info[0]), info[2], info[1]) if info is not None else (0, 0, 0)

    @property
    def source_identity(self):
        """Verified loader provenance, independent of paths; owned JSON copy.

        Manually constructed datasets have no audited source identity.
        """
        return _thaw_metadata(self._source_identity)

    @property
    def split(self):
        return self._split

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index: int) -> SupervisedSample:
        if type(index) is not int:
            raise ValueError("Sample index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(index)
        info = _decoded_caches.get(self)
        cache = info[0] if info is not None else None
        if cache is not None and index in cache:
            sample = cache[index]
            cache.move_to_end(index)
            return SupervisedSample(sample.reference, sample.split, sample.state.copy(),
                sample.features.copy(), sample.legal_mask.copy(), sample.policy_class, sample.wdl.copy())
        label = self._labels[index]
        ref = self._refs[_key(label)]
        game = self._games[ref["game_id"]]
        state = make_state(game["start_fen"], game["moves_uci"][:ref["sample_ply"]], history_complete=True)
        encoded, mask = encode_position(state), legal_mask(state)
        if encoded.features.shape != FEATURE_SHAPE or mask.shape != (MOVE_COUNT,):
            raise ValueError("Adapter output shape mismatch")
        sample = SupervisedSample(_key(label), self.split, state, encoded.features, mask,
                                 label["best_move_id"], np.array(label["teacher_wdl"], dtype=np.float32))
        if cache is not None:
            # Include retained board history in the bounded cache estimate.
            size = sample.features.nbytes + sample.legal_mask.nbytes + sample.wdl.nbytes + 256 * (len(state.board.move_stack) + 1)
            if size <= info[1]:
                while cache and info[2] + size > info[1]:
                    old = cache.popitem(last=False)[1]
                    info[2] -= old.features.nbytes + old.legal_mask.nbytes + old.wdl.nbytes + 256 * (len(old.state.board.move_stack) + 1)
                cache[index] = sample
                info[2] += size
                return SupervisedSample(sample.reference, sample.split, sample.state.copy(),
                    sample.features.copy(), sample.legal_mask.copy(), sample.policy_class, sample.wdl.copy())
        return sample

    def reference(self, index):
        """Source-order identity without replay or tensor construction."""
        return _key(self._labels[index])

    def batch(self, indices):
        """Decode only a chosen nonempty batch, including resume cursor slices."""
        indices = tuple(indices)
        if not indices or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError("batch indices must be nonempty valid integers")
        samples = [self[i] for i in indices]
        return SupervisedBatch(tuple(s.reference for s in samples), self.split,
            np.stack([s.features for s in samples]), np.stack([s.legal_mask for s in samples]),
            np.array([s.policy_class for s in samples], dtype=np.int64), np.stack([s.wdl for s in samples]))

    def batches(self, batch_size: int, *, order=None, drop_last=False) -> Iterator[SupervisedBatch]:
        """Explicit permutation only; default source order, retain partial batch."""
        if type(batch_size) is not int or batch_size <= 0 or type(drop_last) is not bool:
            raise ValueError("batch_size must be positive integer; drop_last must be boolean")
        try:
            indices = list(range(len(self))) if order is None else list(order)
        except TypeError as error:
            raise ValueError("order must be a permutation of this split's indices") from error
        if (any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(self)))):
            raise ValueError("order must be a permutation of this split's indices")
        for start in range(0, len(indices), batch_size):
            group = indices[start:start + batch_size]
            if drop_last and len(group) < batch_size:
                break
            yield self.batch(group)


def open_dataset(*, bundle: Path, manifest_path: Path, games_path: Path,
                 split_audit_path: Path, prepared: Path, teacher_evidence: Path,
                 split: str) -> SupervisedDataset:
    """Verify a complete D05 bundle and audits, then select train/validation/test.

    teacher_evidence is the accepted build attestation named by the manifest;
    no executable/assets are opened or engine processes started. No split default
    exists. All labels (including other splits) must pass before any are exposed.
    """
    if split not in SPLITS:
        raise ValueError("Explicit split must be train, validation or test")
    try:
        manifest = DatasetManifest.load(manifest_path)
        identities = verify_artifacts(manifest, manifest_path, games_path, split_audit_path, prepared)
        summary = _read(bundle / "summary.json")
        if set(summary["output_identities"]) != set(BUNDLE_FILES):
            raise ValueError("Bundle output schema mismatch")
        bundle_identities = {name: file_identity(bundle / name) for name in BUNDLE_FILES}
        _equal(summary["output_identities"], bundle_identities, "Completed bundle bytes")
        settings = _read(bundle / "settings.json")
        evidence_identity = file_identity(teacher_evidence)
        provenance = _provenance(manifest, identities, settings, teacher_evidence)
        refs = _references(_rows(bundle / "selected.jsonl"))
        labels = list(_rows(bundle / "labels.jsonl"))
        attempts, exclusions = list(_rows(bundle / "attempts.jsonl")), list(_rows(bundle / "exclusions.jsonl"))
        successes = _ledger(refs, labels, attempts, exclusions, summary, manifest, settings["max_attempts"])
        # Join only requested game/ply keys, but scan entire verified tables so
        # duplicates and missing membership cannot hide behind an early break.
        audited = {}
        for ref in _rows(prepared / "samples.jsonl"):
            key = _key(ref)
            if key in refs:
                if key in audited:
                    raise ValueError("Duplicate audited sample")
                _equal(ref, refs[key], "Audited sample membership")
                audited[key] = ref
        if set(audited) != set(refs):
            raise ValueError("Missing audited sample membership")
        wanted, positions = {key[0] for key in refs}, {r["rule_position"] for r in refs.values()}
        held_games, protected = {}, {}
        for row in _rows(prepared / "heldout-games.jsonl"):
            if row["game_id"] in wanted:
                held_games[row["game_id"]] = row["split"]
        for row in _rows(prepared / "protected-heldout.jsonl"):
            if row["rule_position"] in positions:
                protected[row["rule_position"]] = row["split"]
        for ref in refs.values():
            expected = ref["split"] if ref["split"] != "train" else None
            if (held_games.get(ref["game_id"]) != expected
                    or protected.get(ref["rule_position"]) != expected):
                raise ValueError("Protected held-out game/position membership mismatch")
        games = {}
        for game in _rows(games_path):
            if game["game_id"] in wanted:
                if game["game_id"] in games:
                    raise ValueError("Duplicate canonical game")
                games[game["game_id"]] = game
        if set(games) != wanted:
            raise ValueError("Missing canonical game")
        for number, label in enumerate(labels, 1):
            try:
                key = _key(label)
                _load_record(label, refs[key], games[key[0]], manifest, provenance, successes[key])
            except (ValueError, TypeError, KeyError, IndexError) as error:
                raise ValueError(f"labels.jsonl:{number} ({_key(label)}): {error}") from error
        # Check that no file changed between its identity check and consumption.
        _equal(verify_artifacts(manifest, manifest_path, games_path, split_audit_path, prepared), identities,
               "Authoritative inputs changed while loading")
        _equal({name: file_identity(bundle / name) for name in BUNDLE_FILES}, bundle_identities,
               "Bundle changed while loading")
        _equal(_read(bundle / "summary.json"), summary, "Summary changed while loading")
        _equal(file_identity(teacher_evidence), evidence_identity, "Teacher evidence changed while loading")
        dataset = SupervisedDataset(split, [r for r in labels if r["split"] == split], refs, games,
            source_identity=dict(schema="d07-source-identity-v1", dataset_id=manifest.dataset_id,
                split=split, authoritative=identities, bundle=bundle_identities,
                summary=summary, teacher_evidence=evidence_identity))
        dataset._admission_fingerprint = dataset._metadata_fingerprint()
        return dataset
    except (OSError, KeyError, TypeError, IndexError, ValueError) as error:
        raise ValueError(f"Cannot load supervised bundle {bundle}: {error}") from error
