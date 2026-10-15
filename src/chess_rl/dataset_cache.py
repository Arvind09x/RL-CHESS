"""Verified on-disk cache of decoded supervised samples.

A labelled bundle stores references and targets, not tensors: every use of a sample replays its
game to the sampled ply and encodes the board. This module writes the decoded tensors of a split
once, during the dataset's own admission decode pass, and reads them back through the same
SupervisedDataset interface so training, validation and resume are unchanged.

Layout, inside the bundle so the cache is tied to its source: ``<bundle>/cache/<split>/`` with
``features.npy`` float32 [N,64,128], ``legal_mask.npy`` uint8 [N,544] (bit-packed 4352 bools),
``policy.npy`` int32 [N], ``wdl.npy`` float32 [N,3], ``references.json`` and ``manifest.json``.

Opening checks, without replaying anything: the format, encoding and move-vocabulary versions
equal this code's constants; the recorded source identity equals one recomputed now from the
bundle, summary, teacher evidence and authoritative inputs; every array file's size and sha256
match the manifest; shapes and counts agree. Any mismatch raises; nothing falls back silently.

Cached samples carry ``state=None``: the board is not reconstructed from the cache.
"""
from dataclasses import asdict
import importlib.metadata
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from types import MappingProxyType

import chess
import numpy as np

from chess_rl.canonical_games import SPLITS, canonical_json
from chess_rl.contracts import ENCODING_VERSION, FEATURE_SHAPE, MOVE_COUNT, MOVE_VOCAB_VERSION
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity
from chess_rl.stockfish_worker import verify_artifacts
from chess_rl.supervised import (BUNDLE_FILES, SupervisedBatch, SupervisedDataset, SupervisedSample,
                                 VerifiedBundle, _freeze_metadata, _read, _thaw_metadata)

CACHE_FORMAT_VERSION = "d07-sample-cache-v1"
MASK_BYTES = MOVE_COUNT // 8
# The same pinned replay/encoding dependencies the fresh loader enforces (supervised._provenance).
PINNED_RUNTIME = {"python-chess": "1.999", "chess": "1.11.2"}
EXPECTED_ENTRIES = ("features", "legal_mask", "policy", "wdl", "references")
ARRAYS = {  # name: (dtype, trailing shape)
    "features": ("float32", FEATURE_SHAPE),
    "legal_mask": ("uint8", (MASK_BYTES,)),
    "policy": ("int32", ()),
    "wdl": ("float32", (3,)),
}


def cache_dir(bundle, split):
    return Path(bundle) / "cache" / split


def bytes_per_sample():
    return sum(np.dtype(dtype).itemsize * int(np.prod(shape, dtype=np.int64)) for dtype, shape in ARRAYS.values())


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_digest(reference, features, legal_mask, policy_class, wdl):
    """Digest of one decoded sample as the reader must reproduce it."""
    digest = hashlib.sha256()
    digest.update(json.dumps([reference[0], int(reference[1])]).encode())
    digest.update(np.ascontiguousarray(features, dtype=np.float32).tobytes())
    digest.update(np.packbits(np.asarray(legal_mask, dtype=np.bool_)).tobytes())
    digest.update(int(policy_class).to_bytes(4, "little", signed=True))
    digest.update(np.ascontiguousarray(wdl, dtype=np.float32).tobytes())
    return digest.digest()


def _installed_runtime():
    return {"python-chess": importlib.metadata.version("python-chess"), "chess": chess.__version__}


def _fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _git_revision():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def current_source_identity(*, bundle, manifest_path, games_path, split_audit_path, prepared, teacher_evidence, split):
    """The source_identity verify_bundle would record now, from file bytes alone (no replay)."""
    manifest = DatasetManifest.load(manifest_path)
    identities = verify_artifacts(manifest, manifest_path, games_path, split_audit_path, prepared)
    bundle = Path(bundle)
    return dict(schema="d07-source-identity-v1", dataset_id=manifest.dataset_id, split=split,
                authoritative=identities, bundle={name: file_identity(bundle / name) for name in BUNDLE_FILES},
                summary=_read(bundle / "summary.json"), teacher_evidence=file_identity(Path(teacher_evidence)))


class _Writer:
    """Sequential .npy writers for one split; rows must arrive in index order."""
    def __init__(self, directory, count):
        self._count, self._next, self._handles, self._digests = count, 0, {}, {}
        for name, (dtype, shape) in ARRAYS.items():
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, dict(descr=np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                                             fortran_order=False, shape=(count,) + tuple(shape)))
            handle = open(directory / f"{name}.npy", "wb")
            handle.write(header.getvalue())
            self._handles[name], self._digests[name] = handle, hashlib.sha256(header.getvalue())
        self._references = []

    def write(self, index, sample):
        if index != self._next:
            raise ValueError(f"cache rows must be written in order: expected {self._next}, got {index}")
        if sample.features.shape != FEATURE_SHAPE or sample.features.dtype != np.float32:
            raise ValueError("unexpected feature array")
        if sample.legal_mask.shape != (MOVE_COUNT,) or sample.legal_mask.dtype != np.bool_:
            raise ValueError("unexpected legal mask")
        rows = dict(features=np.ascontiguousarray(sample.features), legal_mask=np.packbits(sample.legal_mask),
                    policy=np.array(sample.policy_class, dtype=np.int32), wdl=np.ascontiguousarray(sample.wdl, dtype=np.float32))
        for name, row in rows.items():
            data = row.tobytes(order="C")
            self._handles[name].write(data)
            self._digests[name].update(data)
        self._references.append(list(sample.reference))
        self._next += 1

    def close(self):
        if self._next != self._count:
            raise ValueError(f"cache incomplete: {self._next} of {self._count} rows written")
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        return self._references


def build_split_cache(verified: VerifiedBundle, split, *, bundle, data_paths, sample_check=1000, seed=20261015):
    """Write one split's cache from a verified snapshot; returns the manifest dict.

    Rows are written during freeze()'s own decode pass; a digest of every decoded sample is taken
    before the writer receives it. Afterwards every array file is re-read and hashed (the bytes equal what was
    written), every row is read back through the cache reader and its digest must equal the one
    recorded at decode time (reader equivalence for all rows, without a second replay), and
    ``sample_check`` randomly chosen rows plus the first and last are additionally decoded afresh
    and compared field by field. The directory is published atomically by rename.
    """
    bundle = Path(bundle)
    final = cache_dir(bundle, split)
    if final.exists():
        raise FileExistsError(f"cache already exists: {final}; remove it to rebuild")
    summary = _read(bundle / "summary.json")
    if summary.get("loader_acceptance") is None:
        raise ValueError("summary.json has no loader_acceptance; finalize acceptance before building a cache")
    dataset = verified.select(split)
    count = len(dataset)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{split}.tmp-", dir=final.parent))
    try:
        writer = _Writer(tmp, count)
        expected_digests = []

        def observe(index, sample):
            # Expected digest is taken from the decoded sample before the writer sees it, so a
            # writer fault cannot influence what the read-back must reproduce.
            expected_digests.append(_row_digest(sample.reference, sample.features, sample.legal_mask,
                                                sample.policy_class, sample.wdl))
            writer.write(index, sample)

        frozen = dataset.freeze(on_sample=observe)
        references = writer.close()
        (tmp / "references.json").write_text(json.dumps(references, separators=(",", ":")) + "\n")
        _fsync_path(tmp / "references.json")
        files = {}
        for name in ARRAYS:
            path = tmp / f"{name}.npy"
            digest = _sha256(path)
            if digest != writer._digests[name].hexdigest():
                raise ValueError(f"{name}.npy bytes differ from what was written")
            files[name] = dict(file=f"{name}.npy", dtype=ARRAYS[name][0], shape=[count, *ARRAYS[name][1]],
                               bytes=path.stat().st_size, sha256=digest)
        files["references"] = dict(file="references.json", bytes=(tmp / "references.json").stat().st_size,
                                   sha256=_sha256(tmp / "references.json"))
        manifest = dict(cache_format_version=CACHE_FORMAT_VERSION, split=split, count=count,
            encoding_version=ENCODING_VERSION, move_vocab_version=MOVE_VOCAB_VERSION,
            source_identity=frozen.source_identity, admission_fingerprint=frozen._admission_fingerprint,
            arrays=files, built_at_utc=datetime.now(timezone.utc).isoformat(), code_revision=_git_revision(),
            pinned_runtime=_installed_runtime(),
            builder_runtime=dict(python=platform.python_version(), numpy=np.__version__, host=platform.node()))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Every row read back through the real reader must reproduce the digest taken at decode time.
        reader = CachedSupervisedDataset._open_directory(tmp, manifest)
        for i in range(count):
            row = reader[i]
            if _row_digest(row.reference, row.features, row.legal_mask, row.policy_class, row.wdl) != expected_digests[i]:
                raise ValueError(f"cache read-back digest mismatch at index {i}")
        rng = np.random.default_rng(seed)
        picks = sorted({0, count - 1, *map(int, rng.choice(count, size=min(sample_check, count), replace=False))})
        for i in picks:
            fresh, cached = frozen[i], reader[i]
            if (fresh.reference != cached.reference or fresh.policy_class != cached.policy_class
                    or not np.array_equal(fresh.features, cached.features) or fresh.features.dtype != cached.features.dtype
                    or not np.array_equal(fresh.legal_mask, cached.legal_mask) or not np.array_equal(fresh.wdl, cached.wdl)):
                raise ValueError(f"cache read-back mismatch at index {i}")
        manifest["verification"] = dict(all_rows_digest_checked=count,
                                        fresh_decode_compared=picks if len(picks) <= 16 else dict(count=len(picks), seed=seed))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _fsync_path(tmp / "manifest.json")
        _fsync_path(tmp)
        os.rename(tmp, final)
        _fsync_path(final.parent)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


class CachedSupervisedDataset(SupervisedDataset):
    """Memory-mapped, verified, read-only view of a cached split; same interface as a frozen dataset."""
    def __init__(self, directory, manifest, arrays, references):
        object.__setattr__(self, "_sealed", False)
        self._directory = Path(directory)
        self._manifest = _freeze_metadata(manifest)
        self._arrays = MappingProxyType(dict(arrays))
        self._references = tuple(references)
        self._split = manifest["split"]
        self._source_identity = _freeze_metadata(manifest["source_identity"])
        self._admission_fingerprint = manifest["admission_fingerprint"]
        self._labels = self._refs = self._games = None
        self._sealed = True

    @classmethod
    def _open_directory(cls, directory, manifest):
        directory = Path(directory)
        arrays = {}
        for name, (dtype, shape) in ARRAYS.items():
            array = np.load(directory / f"{name}.npy", mmap_mode="r")
            expected = (manifest["count"],) + tuple(shape)
            if array.dtype != np.dtype(dtype) or array.shape != expected:
                raise ValueError(f"{name}.npy has dtype {array.dtype} shape {array.shape}; expected {dtype} {expected}")
            arrays[name] = array
        references = json.loads((directory / "references.json").read_text())
        if len(references) != manifest["count"] or any(len(r) != 2 for r in references):
            raise ValueError("references.json does not match the cached count")
        return cls(directory, manifest, arrays, tuple((r[0], int(r[1])) for r in references))

    @classmethod
    def open(cls, *, split, **data_paths):
        """Open a split's cache after every integrity and identity check; raises on any mismatch."""
        if split not in SPLITS:
            raise ValueError("Explicit split must be train, validation or test")
        directory = cache_dir(data_paths["bundle"], split)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no sample cache for split {split!r} at {directory}")
        manifest = json.loads(manifest_path.read_text())
        for field, expected in (("cache_format_version", CACHE_FORMAT_VERSION), ("encoding_version", ENCODING_VERSION),
                                ("move_vocab_version", MOVE_VOCAB_VERSION), ("split", split)):
            if manifest.get(field) != expected:
                raise ValueError(f"cache {field} {manifest.get(field)!r} is not {expected!r}")
        if manifest.get("pinned_runtime") != PINNED_RUNTIME or _installed_runtime() != PINNED_RUNTIME:
            raise ValueError(f"cache or environment replay dependencies differ from the pinned {PINNED_RUNTIME}")
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(EXPECTED_ENTRIES):
            raise ValueError(f"cache manifest must describe exactly {EXPECTED_ENTRIES}")
        count = manifest.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("cache manifest count must be a nonnegative integer")
        for name, record in arrays.items():
            expected_file = f"{name}.npy" if name in ARRAYS else "references.json"
            if record.get("file") != expected_file:
                raise ValueError(f"cache manifest entry {name} must name {expected_file}")
            if name in ARRAYS and (record.get("dtype") != ARRAYS[name][0]
                                   or list(record.get("shape", [])) != [count, *ARRAYS[name][1]]):
                raise ValueError(f"cache manifest entry {name} declares unexpected dtype/shape")
        current = current_source_identity(split=split, **data_paths)
        if canonical_json(current) != canonical_json(manifest["source_identity"]):
            raise ValueError("cache source identity does not match the bundle and inputs on disk; rebuild the cache")
        for name in EXPECTED_ENTRIES:
            record = arrays[name]
            path = directory / record["file"]
            if not path.exists() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
                raise ValueError(f"cache file {record['file']} is missing, truncated or altered")
        return cls._open_directory(directory, manifest)

    @property
    def manifest(self):
        return _thaw_metadata(self._manifest)

    def freeze(self, *, cache_bytes=64 * 1024 * 1024, on_sample=None):
        """Already immutable and fully decoded; returns itself."""
        return self

    @property
    def cache_info(self):
        return (0, 0, 0)

    def _metadata_fingerprint(self):
        return self._admission_fingerprint

    def __len__(self):
        return self._manifest["count"]

    def reference(self, index):
        return self._references[index]

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError("Sample index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(index)
        features = np.array(self._arrays["features"][index], dtype=np.float32, copy=True)
        mask = np.unpackbits(np.asarray(self._arrays["legal_mask"][index])).astype(bool)
        return SupervisedSample(self._references[index], self.split, None, features, mask,
                                int(self._arrays["policy"][index]), np.array(self._arrays["wdl"][index], dtype=np.float32, copy=True))

    def batch(self, indices):
        indices = tuple(indices)
        if not indices or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError("batch indices must be nonempty valid integers")
        order = np.asarray(indices, dtype=np.int64)
        features = np.array(self._arrays["features"][order], dtype=np.float32, copy=True)
        masks = np.unpackbits(np.asarray(self._arrays["legal_mask"][order]), axis=1).astype(bool)
        return SupervisedBatch(tuple(self._references[i] for i in indices), self.split, features, masks,
                               np.asarray(self._arrays["policy"][order], dtype=np.int64),
                               np.array(self._arrays["wdl"][order], dtype=np.float32, copy=True))
