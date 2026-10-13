"""Attested alternate Stockfish builds: worker acceptance, loader provenance, attestation helper."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from chess_rl.pgn_replay import file_identity
from chess_rl.stockfish_worker import alternate_build_record, teacher_provenance, verify_teacher_bytes
from chess_rl.supervised import open_dataset
from .test_stockfish_worker import manifest as manifest_fixture  # noqa: F401
from .test_supervised import fixture_bundle, EVIDENCE, seal_bundle

ROOT = Path(__file__).resolve().parents[1]


def fake_engine(manifest, tmp_path, *, alternate):
    blobs = [b"first embedded asset", b"second embedded asset"]
    assets = tuple(replace(asset, sha256=hashlib.sha256(blob).hexdigest(),
        name=f"nn-{hashlib.sha256(blob).hexdigest()[:12]}.nnue",
        source_url=f"https://tests.stockfishchess.org/api/nn/nn-{hashlib.sha256(blob).hexdigest()[:12]}.nnue")
        for asset, blob in zip(manifest.engine.embedded_nnue, blobs))
    reference = b"reference header" + b"".join(blobs)
    manifest = replace(manifest, engine=replace(manifest.engine,
        executable_sha256=hashlib.sha256(reference).hexdigest(), embedded_nnue=assets))
    binary = (b"linux build header, longer" if alternate else b"reference header") + b"".join(blobs)
    executable = tmp_path / "engine"
    executable.write_bytes(binary)
    offset = len(binary) - sum(map(len, blobs))
    evidence = dict(executable_sha256=hashlib.sha256(binary).hexdigest(), executable_bytes=len(binary),
        nnue=[dict(name=a.name, sha256=a.sha256, bytes=len(b), embedded_offset=offset + sum(map(len, blobs[:i])))
              for i, (a, b) in enumerate(zip(assets, blobs))])
    if alternate:
        evidence["alternate_build"] = dict(release_tag=manifest.engine.release_tag,
            source_commit=manifest.engine.source_commit, source_archive_sha256=manifest.engine.package_sha256,
            platform="Linux x86_64", build="make -j build ARCH=x86-64-avx2")
    return manifest, executable, evidence


def test_reference_build_record_is_unchanged(manifest_fixture, tmp_path):
    manifest, executable, evidence = fake_engine(manifest_fixture, tmp_path, alternate=False)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    teacher = verify_teacher_bytes(manifest, executable, path)
    assert "alternate_build" not in teacher
    assert teacher["executable_identity"]["sha256"] == manifest.engine.executable_sha256
    assert "alternate_build" not in teacher_provenance({k: dict(sha256="0" * 64) for k in
        ("manifest", "games.jsonl", "samples.jsonl", "overlap-audit.json")}, teacher)


def test_alternate_build_requires_matching_source_attestation(manifest_fixture, tmp_path):
    manifest, executable, evidence = fake_engine(manifest_fixture, tmp_path, alternate=True)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    teacher = verify_teacher_bytes(manifest, executable, path)
    assert teacher["alternate_build"]["reference_executable_sha256"] == manifest.engine.executable_sha256
    assert teacher["alternate_build"]["source_commit"] == manifest.engine.source_commit
    assert teacher["executable_identity"]["sha256"] != manifest.engine.executable_sha256
    identities = {k: dict(sha256=hashlib.sha256(k.encode()).hexdigest()) for k in
                  ("manifest", "games.jsonl", "samples.jsonl", "overlap-audit.json")}
    provenance = teacher_provenance(identities, teacher)
    assert provenance["executable_sha256"] == teacher["executable_identity"]["sha256"]
    assert provenance["alternate_build"] == teacher["alternate_build"]
    # No attestation, or a different commit/archive/tag, is rejected.
    for tamper in ("drop", "source_commit", "source_archive_sha256", "release_tag"):
        bad = json.loads(json.dumps(evidence))
        if tamper == "drop":
            del bad["alternate_build"]
        else:
            bad["alternate_build"][tamper] = "x" * len(str(bad["alternate_build"][tamper]))
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="attestation"):
            verify_teacher_bytes(manifest, executable, path)
    # Attestation never substitutes for matching embedded networks.
    path.write_text(json.dumps(evidence))
    executable.write_bytes(executable.read_bytes()[:-1] + b"?")
    evidence["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="NNUE"):
        verify_teacher_bytes(manifest, executable, path)
    assert alternate_build_record(manifest, evidence, manifest.engine.executable_sha256) is None


def test_loader_accepts_alternate_build_bundle_and_rejects_mismatch(tmp_path):
    """A bundle labelled by an attested alternate build loads; provenance drift is rejected."""
    paths = fixture_bundle(tmp_path)
    bundle = paths["bundle"]
    reference = json.loads(EVIDENCE.read_text())
    from chess_rl.data_manifest import DatasetManifest
    manifest = DatasetManifest.load(paths["manifest_path"])
    alternate_sha = hashlib.sha256(b"linux-build").hexdigest()
    evidence = dict(reference, executable_sha256=alternate_sha, executable_bytes=12345,
        alternate_build=dict(release_tag=manifest.engine.release_tag, source_commit=manifest.engine.source_commit,
            source_archive_sha256=manifest.engine.package_sha256, platform="Linux x86_64", build="make"))
    evidence_path = tmp_path / "linux-evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    settings = json.loads((bundle / "settings.json").read_text())
    expected = alternate_build_record(manifest, evidence, alternate_sha)
    settings["teacher"] = dict(settings["teacher"], executable_identity=dict(bytes=12345, sha256=alternate_sha),
        build_evidence=file_identity(evidence_path), alternate_build=expected)
    rows = [json.loads(line) for line in (bundle / "labels.jsonl").read_text().splitlines()]
    for row in rows:
        row["target_provenance"]["executable_sha256"] = alternate_sha
        row["target_provenance"]["alternate_build"] = expected
    for name in ("summary.json",):
        (bundle / name).unlink()
    (bundle / "settings.json").write_text(json.dumps(settings))
    (bundle / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal_bundle(bundle)
    data = open_dataset(**dict(paths, teacher_evidence=evidence_path), split="train")
    assert len(data) == 3
    # Labels claiming the reference binary while the bundle attests an alternate build are rejected.
    rows[0]["target_provenance"]["executable_sha256"] = manifest.engine.executable_sha256
    (bundle / "summary.json").unlink()
    (bundle / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal_bundle(bundle)
    with pytest.raises(ValueError, match="provenance"):
        open_dataset(**dict(paths, teacher_evidence=evidence_path), split="train")
    # Reference evidence with an alternate-build bundle is a mismatch too.
    with pytest.raises(ValueError):
        open_dataset(**paths, split="train")


def test_attestation_locates_networks_exactly_once(tmp_path):
    from scripts.attest_stockfish import locate_networks
    from chess_rl.data_manifest import DatasetManifest
    manifest = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-d04-v1.json")
    blobs = [b"net-one-bytes-" * 3, b"net-two-bytes-" * 2]
    assets = tuple(replace(a, sha256=hashlib.sha256(b).hexdigest(), name=f"nn-{hashlib.sha256(b).hexdigest()[:12]}.nnue",
                           source_url=f"https://tests.stockfishchess.org/api/nn/nn-{hashlib.sha256(b).hexdigest()[:12]}.nnue")
                   for a, b in zip(manifest.engine.embedded_nnue, blobs))
    manifest = replace(manifest, engine=replace(manifest.engine, embedded_nnue=assets))
    for asset, blob in zip(assets, blobs):
        (tmp_path / asset.name).write_bytes(blob)
    binary = b"ELF" + blobs[0] + b"gap" + blobs[1] + b"tail"
    records = locate_networks(binary, manifest, tmp_path)
    assert [r["embedded_offset"] for r in records] == [3, 3 + len(blobs[0]) + 3]
    assert all(binary[r["embedded_offset"]:r["embedded_offset"] + r["bytes"]] == b for r, b in zip(records, blobs))
    with pytest.raises(SystemExit, match="exactly once"):
        locate_networks(binary + blobs[0], manifest, tmp_path)
    (tmp_path / assets[0].name).write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="manifest hash"):
        locate_networks(binary, manifest, tmp_path)
