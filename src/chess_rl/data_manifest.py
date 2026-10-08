"""D01 manifest configuration only; no replay, sampling or annotation algorithms."""
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from chess_rl.contracts import (
    ENCODING_VERSION, MOVE_VOCAB_VERSION, ENDING_POLICY, WDL_ORDER,
)

MANIFEST_VERSION = "dataset-manifest-v1"
STORAGE_VERSION = "replay-jsonl-v1"
IDENTITY_VERSION = "start-moves-sha256-v1"
AUDIT_POLICY = "rule-position-disjoint-v1"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if re.match(r"^(pending|todo|tbd|unknown|unresolved)\b", value.strip(), re.IGNORECASE):
        raise ValueError(f"{name} must be resolved text; use null for pending provenance")


def _digest(value, name, size=64):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value):
        raise ValueError(f"{name} must be a lowercase {size}-digit hex digest")


def _utc(value, name):
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError()
    except ValueError as exc:
        raise ValueError(f"{name} must be a UTC ISO timestamp") from exc


def _https(value, name):
    _text(value, name)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.fragment:
        raise ValueError(f"{name} must be an absolute HTTPS URL")


def _locked(value, expected, name):
    # bool is an int subclass; equality alone would silently accept it as 1.
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"Unsupported {name}: {value!r}; expected {expected!r}")


@dataclass(frozen=True)
class ArchiveProvenance:
    month: str
    url: str
    index_url: str
    verified_at_utc: str
    verification_evidence: str
    downloaded_sha256: str | None
    retrieved_at_utc: str | None
    downloaded_bytes: int | None

    def __post_init__(self):
        if not isinstance(self.month, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", self.month):
            raise ValueError("month must be YYYY-MM")
        _locked(self.index_url, "https://database.nikonoel.fr/", "archive index")
        _locked(self.url, f"{self.index_url}lichess_elite_{self.month}.zip", "archive URL")
        _utc(self.verified_at_utc, "archive verification time")
        _text(self.verification_evidence, "archive verification evidence")
        if self.downloaded_sha256 is not None:
            _digest(self.downloaded_sha256, "archive checksum")
        if self.retrieved_at_utc is not None:
            _utc(self.retrieved_at_utc, "archive retrieval time")
        if self.downloaded_bytes is not None and (
            type(self.downloaded_bytes) is not int or self.downloaded_bytes <= 0
        ):
            raise ValueError("downloaded_bytes must be a positive integer or null")
        resolved = (self.downloaded_sha256, self.retrieved_at_utc, self.downloaded_bytes)
        if any(v is not None for v in resolved) and any(v is None for v in resolved):
            raise ValueError("archive byte provenance must be resolved together")


@dataclass(frozen=True)
class EngineAsset:
    name: str
    source_url: str
    sha256: str
    embedded: bool

    def __post_init__(self):
        _digest(self.sha256, "NNUE checksum")
        _locked(self.name, f"nn-{self.sha256[:12]}.nnue", "NNUE filename")
        _locked(self.source_url, f"https://tests.stockfishchess.org/api/nn/{self.name}", "NNUE source")
        _locked(self.embedded, True, "embedded NNUE")


@dataclass(frozen=True)
class EngineProvenance:
    release: str
    release_tag: str
    source_commit: str
    platform: str
    build: str
    download_url: str
    package_sha256: str
    executable_member: str
    executable_sha256: str
    uci_name: str
    uci_author: str
    verified_at_utc: str
    verification_evidence: str
    assets: str
    embedded_nnue: tuple[EngineAsset, EngineAsset]

    def __post_init__(self):
        for field in fields(self):
            if field.name != "embedded_nnue":
                _text(getattr(self, field.name), field.name)
        _digest(self.source_commit, "Stockfish source commit", 40)
        _digest(self.package_sha256, "Stockfish package checksum")
        _digest(self.executable_sha256, "Stockfish executable checksum")
        _utc(self.verified_at_utc, "engine verification time")
        _https(self.download_url, "engine download URL")
        _locked(self.uci_name, f"Stockfish {self.release}", "UCI name")
        _locked(self.release_tag, f"sf_{self.release}", "Stockfish release tag")
        _locked(self.assets, "embedded NNUE; no separate runtime assets; Syzygy disabled", "engine assets")
        if (type(self.embedded_nnue) is not tuple or len(self.embedded_nnue) != 2
            or any(type(asset) is not EngineAsset for asset in self.embedded_nnue)
            or len({asset.name for asset in self.embedded_nnue}) != 2):
            raise ValueError("engine must identify two distinct embedded NNUE assets")
        binary_prefix = f"https://github.com/official-stockfish/Stockfish/releases/download/{self.release_tag}/"
        source_release = f"https://codeload.github.com/official-stockfish/Stockfish/tar.gz/refs/tags/{self.release_tag}"
        if not (self.download_url == source_release or (
            self.download_url.startswith(binary_prefix) and self.download_url[len(binary_prefix):]
        )):
            raise ValueError("teacher must reference a pinned official Stockfish release")


@dataclass(frozen=True)
class TeacherConfig:
    config_id: str = "stockfish-17.1-100kn-v1"
    nodes: int = 100_000
    threads: int = 1
    hash_mb: int = 128
    multipv: int = 1
    show_wdl: bool = True
    full_history_roots: bool = True
    hash_reset: str = "Clear Hash before each independent annotation"
    wdl_perspective: str = "root side to move"
    failure_policy: str = "retry or explicitly exclude; never count as a successful label"
    missing_wdl_policy: str = "failure; no centipawn conversion or PGN-result substitution"
    tablebases: str = "disabled"

    def __post_init__(self):
        for field in fields(self):
            _locked(getattr(self, field.name), field.default, f"teacher.{field.name}")


@dataclass(frozen=True)
class PreparationEvidence:
    """Descriptions and artifact checksums filled by D03/D04, not D01 claims."""
    split_algorithm: str | None
    sampling_algorithm: str | None
    phase_boundaries: str | None
    split_audit_sha256: str | None
    overlap_audit_sha256: str | None
    protected_heldout_registry_sha256: str | None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if field.name.endswith("sha256"):
                    _digest(value, field.name)
                else:
                    _text(value, field.name)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    archive: ArchiveProvenance
    engine: EngineProvenance
    teacher: TeacherConfig
    preparation: PreparationEvidence
    split_seed: int
    sampling_seed: int
    schema_version: str = MANIFEST_VERSION
    encoding_version: str = ENCODING_VERSION
    move_vocab_version: str = MOVE_VOCAB_VERSION
    ending_policy: str = ENDING_POLICY
    wdl_order: tuple[str, str, str] = WDL_ORDER
    storage_version: str = STORAGE_VERSION
    game_identity: str = IDENTITY_VERSION
    duplicate_identity: str = IDENTITY_VERSION
    overlap_policy: str = AUDIT_POLICY
    split_percent: tuple[int, int, int] = (90, 5, 5)
    split_before_sampling: bool = True
    max_distinct_nonterminal_positions_per_game: int = 8
    initial_successful_labels: int = 10_000
    separate_benchmark_positions: int = 1_000
    decision_record: str = "docs/d01-data-manifest.md"
    game_id_format: str = "g:SHA256(canonical UTF-8 JSON of identity_version, normalized six-field start_fen and full actual moves_uci; sorted keys, compact separators, no newline)"
    duplicate_policy: str = "same start and full moves share game_id regardless of PGN headers/results; retain all archive/member/ordinal/LichessURL provenance; reject identity conflicts"
    storage_layout: str = "UTF-8 JSONL: games once with authoritative replay/history/source/result; samples reference game_id and sample_ply with root/provenance/class/soft WDL/search info; no repeated moves or dense one-hot policies"
    overlap_requirements: str = "audit pieces/side/castling/legal en-passant independent of counters/network history; distinct plies and rule positions per game; remove cross-split overlap before annotation; report phase/split counts before/after, removals/refill/expansion"
    heldout_policy: str = "freeze validation/test game assignments and sample registries; never train or replay"
    expansion_policy: str = "measure coverage/costs/learning curves before 100k, several hundred thousand or optional 1m; preserve assignments and heldouts; repeat overlap audits"

    def __post_init__(self):
        _text(self.dataset_id, "dataset_id")
        for name, expected in (("archive", ArchiveProvenance), ("engine", EngineProvenance),
                               ("teacher", TeacherConfig), ("preparation", PreparationEvidence)):
            if type(getattr(self, name)) is not expected:
                raise ValueError(f"{name} must be {expected.__name__}")
        for name in ("split_seed", "sampling_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be a uint32 integer")
        for field in fields(self):
            if field.name not in {"dataset_id", "archive", "engine", "teacher", "preparation", "split_seed", "sampling_seed"}:
                _locked(getattr(self, field.name), field.default, field.name)
        # JSON normalizes tuples, so additionally reject bool/float percentage entries.
        if any(type(v) is not int for v in self.split_percent):
            raise ValueError("split_percent must contain integers")
        _locked(self.teacher.config_id, f"stockfish-{self.engine.release}-100kn-v1", "teacher config identity")

    def annotation_blockers(self) -> tuple[str, ...]:
        missing = [f"archive.{name}" for name in
                   ("downloaded_sha256", "retrieved_at_utc", "downloaded_bytes")
                   if getattr(self.archive, name) is None]
        missing.extend(f"preparation.{field.name}" for field in fields(self.preparation)
                       if getattr(self.preparation, field.name) is None)
        return tuple(missing)

    def require_annotation_ready(self) -> None:
        """Fail closed on pending provenance. Workers must also verify artifact bytes."""
        missing = self.annotation_blockers()
        if missing:
            raise ValueError("Manifest is preparation-ready only; annotation blocked by: " + ", ".join(missing))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "DatasetManifest":
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON field: {key}")
                result[key] = value
            return result

        try:
            data = json.loads(text, object_pairs_hook=unique_pairs)
            if not isinstance(data, dict):
                raise ValueError("manifest must be an object")
            # Require explicit serialized fields, including null pending provenance.
            if set(data) != {field.name for field in fields(cls)}:
                raise ValueError("manifest fields differ from supported schema")
            for name, kind in (("archive", ArchiveProvenance), ("engine", EngineProvenance),
                               ("teacher", TeacherConfig), ("preparation", PreparationEvidence)):
                if not isinstance(data[name], dict) or set(data[name]) != {f.name for f in fields(kind)}:
                    raise ValueError(f"{name} fields differ from supported schema")
                if name == "engine":
                    assets = data[name]["embedded_nnue"]
                    if not isinstance(assets, list) or any(
                        not isinstance(a, dict) or set(a) != {f.name for f in fields(EngineAsset)}
                        for a in assets
                    ):
                        raise ValueError("embedded_nnue fields differ from supported schema")
                    data[name]["embedded_nnue"] = tuple(EngineAsset(**a) for a in assets)
                data[name] = kind(**data[name])
            for name in ("wdl_order", "split_percent"):
                if not isinstance(data[name], list):
                    raise ValueError(f"{name} must be a JSON array")
                data[name] = tuple(data[name])
            return cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid manifest: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
