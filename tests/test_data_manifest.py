"""Manifest acceptance independent of network, engine installation and Torch."""
from dataclasses import fields, replace
import json
from pathlib import Path

import pytest

from chess_rl.contracts import ENCODING_VERSION, MOVE_VOCAB_VERSION, ENDING_POLICY, WDL_ORDER
from chess_rl.data_manifest import DatasetManifest, PreparationEvidence, TeacherConfig

SELECTED = Path(__file__).resolve().parents[1] / "data/manifests/elite-2025-01-v1.json"


@pytest.fixture
def manifest():
    return DatasetManifest.load(SELECTED)


def test_selected_round_trip_and_shared_contract(manifest, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json())
    restored = DatasetManifest.load(path)
    assert restored == manifest
    assert restored.to_json() == SELECTED.read_text()
    assert (restored.encoding_version, restored.move_vocab_version, restored.ending_policy,
            restored.wdl_order) == (ENCODING_VERSION, MOVE_VOCAB_VERSION, ENDING_POLICY, WDL_ORDER)
    assert restored.split_percent == (90, 5, 5)
    assert restored.split_before_sampling
    assert restored.max_distinct_nonterminal_positions_per_game == 8
    assert restored.initial_successful_labels == 10_000
    assert restored.separate_benchmark_positions == 1_000
    assert restored.game_identity == restored.duplicate_identity


@pytest.mark.parametrize("name,value", [
    ("schema_version", "dataset-manifest-v2"), ("encoding_version", "chess128-v1"),
    ("move_vocab_version", "pair88-v1"), ("ending_policy", "intended-move-v1"),
    ("wdl_order", ("loss", "draw", "win")), ("storage_version", "dense-policy-v1"),
    ("game_identity", "pgn-header-v1"), ("duplicate_identity", "site-only-v1"),
    ("overlap_policy", "history-tensor-only"), ("split_percent", (80, 10, 10)),
    ("split_percent", (90.0, 5, 5)), ("split_before_sampling", False),
    ("max_distinct_nonterminal_positions_per_game", 9), ("initial_successful_labels", 100_000),
    ("separate_benchmark_positions", 0), ("split_seed", True), ("sampling_seed", -1),
    ("split_seed", 2**32), ("sampling_seed", "20261009"), ("dataset_id", ""),
    ("heldout_policy", "may train"), ("expansion_policy", "mandatory million"),
])
def test_incompatible_decisions_rejected(manifest, name, value):
    with pytest.raises(ValueError):
        replace(manifest, **{name: value})


@pytest.mark.parametrize("name,value", [
    ("nodes", 1000), ("threads", 2), ("threads", True), ("hash_mb", 256),
    ("multipv", 2), ("show_wdl", False), ("show_wdl", 1),
    ("full_history_roots", False), ("hash_reset", "once per worker"),
    ("wdl_perspective", "white"), ("failure_policy", "accept partial"),
    ("missing_wdl_policy", "convert centipawns"), ("tablebases", "auto"),
    ("config_id", "other"),
])
def test_teacher_contract_rejected(name, value):
    with pytest.raises(ValueError):
        TeacherConfig(**{name: value})


@pytest.mark.parametrize("group,name,value", [
    ("archive", "month", "2025-13"), ("archive", "url", "https://example.com/month.zip"),
    ("archive", "downloaded_sha256", "pending D02"),
    ("archive", "downloaded_bytes", None),  # partial retrieval record
    ("archive", "verified_at_utc", "2026-10-08"),
    ("archive", "verification_evidence", ""), ("engine", "executable_sha256", None),
    ("engine", "package_sha256", "A" * 64), ("engine", "source_commit", "main"),
    ("engine", "uci_name", "Stockfish 16"), ("engine", "download_url", "https://example.com/sf.tar"),
    ("engine", "assets", ""), ("preparation", "overlap_audit_sha256", "passed"),
    ("preparation", "phase_boundaries", ""), ("preparation", "sampling_algorithm", "pending D04"),
    ("engine", "build", "unknown"),
])
def test_invalid_provenance_rejected(manifest, group, name, value):
    with pytest.raises(ValueError):
        replace(getattr(manifest, group), **{name: value})


def complete_metadata(manifest):
    # Synthetic provenance exercises the metadata gate, never attests a real audit.
    archive = replace(manifest.archive, downloaded_sha256="a" * 64,
                      retrieved_at_utc="2026-10-08T08:00:00Z", downloaded_bytes=100)
    prep = replace(manifest.preparation, split_algorithm="synthetic split description",
                   sampling_algorithm="synthetic sampler", phase_boundaries="synthetic phases",
                   split_audit_sha256="b" * 64, overlap_audit_sha256="c" * 64,
                   protected_heldout_registry_sha256="d" * 64)
    return replace(manifest, archive=archive, preparation=prep)


def test_selected_is_preparation_only(manifest):
    assert manifest.archive.downloaded_sha256 is not None
    assert "archive.downloaded_sha256" not in manifest.annotation_blockers()
    assert len(manifest.annotation_blockers()) == 6
    with pytest.raises(ValueError, match="preparation-ready only.*preparation.split_algorithm"):
        manifest.require_annotation_ready()
    complete_metadata(manifest).require_annotation_ready()


@pytest.mark.parametrize("name", [field.name for field in fields(PreparationEvidence)])
def test_every_required_preparation_field_blocks_annotation(manifest, name):
    resolved = complete_metadata(manifest)
    pending = replace(resolved, preparation=replace(resolved.preparation, **{name: None}))
    with pytest.raises(ValueError, match=name):
        pending.require_annotation_ready()


def test_archive_checksum_remains_required_after_algorithms_resolved(manifest):
    pending = replace(complete_metadata(manifest), archive=replace(manifest.archive,
                      downloaded_sha256=None, downloaded_bytes=None, retrieved_at_utc=None))
    with pytest.raises(ValueError, match="archive.downloaded_sha256"):
        pending.require_annotation_ready()


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nested_unknown", "nested_missing", "claim_ready"])
def test_serialized_shape_fail_closed(manifest, mutation):
    data = json.loads(manifest.to_json())
    if mutation == "unknown":
        data["arbitrary"] = 1
    elif mutation == "missing":
        del data["encoding_version"]
    elif mutation == "nested_unknown":
        data["teacher"]["depth"] = 20
    elif mutation == "nested_missing":
        del data["archive"]["downloaded_sha256"]
    else:
        data["annotation_ready"] = True
    with pytest.raises(ValueError):
        DatasetManifest.from_json(json.dumps(data))


@pytest.mark.parametrize("text", ['[]', 'null', '{"schema_version":1,"schema_version":2}',
                                  '{"archive":{"url":"a","url":"b"}}'])
def test_json_rejects_nonobjects_and_duplicate_fields(text):
    with pytest.raises(ValueError):
        DatasetManifest.from_json(text)


def test_serialized_invalid_setting_is_not_coerced(manifest):
    data = json.loads(manifest.to_json())
    data["teacher"]["nodes"] = 100000.0
    with pytest.raises(ValueError, match="nodes"):
        DatasetManifest.from_json(json.dumps(data))


def test_embedded_assets_require_checksums_and_no_external_dependencies(manifest):
    asset = manifest.engine.embedded_nnue[0]
    for change in ({"sha256": "pending"}, {"embedded": False}, {"name": "nn-other.nnue"}):
        with pytest.raises(ValueError):
            replace(asset, **change)
    with pytest.raises(ValueError):
        replace(manifest.engine, embedded_nnue=(asset, asset))


@pytest.mark.parametrize("corruption", [None, "name", "author", "uciok", "readyok", "Threads",
                                      "Hash", "MultiPV", "UCI_ShowWDL", "Clear Hash", "EvalFile"])
def test_idle_uci_verification_and_capability_rejection(manifest, tmp_path, monkeypatch, corruption):
    import hashlib
    import subprocess
    from scripts.check_d01 import verify_teacher

    # Saved real D01 transcript is the external protocol fixture; no executable
    # installation is needed. Mutations test the verifier against failed engines.
    report = json.loads((SELECTED.parents[2] / "runs/d01/manifest-check.json").read_text())
    transcript = report["teacher"]["stdout"]
    executable = tmp_path / "teacher"
    executable.write_bytes(b"synthetic executable")
    synthetic = replace(manifest, engine=replace(manifest.engine, executable_sha256=
                        hashlib.sha256(executable.read_bytes()).hexdigest()))
    if corruption in ("name", "author"):
        transcript = "\n".join(line for line in transcript.splitlines()
                               if not line.startswith(f"id {corruption} "))
    elif corruption in ("uciok", "readyok"):
        transcript = transcript.replace(corruption, "")
    elif corruption:
        transcript = "\n".join(line for line in transcript.splitlines()
                               if not line.startswith(f"option name {corruption} type "))
    def fake_run(argv, *, input, **kwargs):
        sent = input.splitlines()
        assert not any(command.startswith(("go", "bench", "speedtest", "eval")) for command in sent)
        assert "setoption name Hash value 128" in sent
        assert "setoption name Clear Hash" in sent
        assert "setoption name UCI_ShowWDL value true" in sent
        assert "setoption name MultiPV value 1" in sent
        assert "setoption name Threads value 1" in sent
        return subprocess.CompletedProcess(argv, 0, transcript, "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    if corruption:
        with pytest.raises(ValueError):
            verify_teacher(synthetic, executable)
    else:
        checked = verify_teacher(synthetic, executable)
        assert checked["search_commands_sent"] is False
        # A binary mismatch must be rejected before the process is launched.
        def forbid_process(*args, **kwargs):
            pytest.fail("executed a teacher with a mismatched checksum")
        monkeypatch.setattr(subprocess, "run", forbid_process)
        with pytest.raises(ValueError, match="checksum"):
            verify_teacher(manifest, executable)
