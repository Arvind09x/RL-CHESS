"""D09 labeller: nested membership, compact ledger, resume, prefix finalize, equivalence compare."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import chess
import pytest

from chess_rl.data_manifest import DatasetManifest
from chess_rl.initial_labels import key
from chess_rl.label_expansion import (compact_attempt, compare_labels, completing_records,
    label_differences, select_membership)
from chess_rl.pgn_replay import file_identity
from chess_rl.supervised import open_dataset
from .test_supervised import fixture_bundle, read_rows, write_rows

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/expand_labels.py"
ENV = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")


def fake_teacher(tmp_path):
    """A shell wrapper around the fake UCI engine with two fake nets embedded as bytes."""
    manifest = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-d04-v1.json")
    blobs = [b"FAKE-NET-ONE-" * 4, b"FAKE-NET-TWO-" * 3]
    assets = tuple(replace(a, sha256=hashlib.sha256(b).hexdigest(), name=f"nn-{hashlib.sha256(b).hexdigest()[:12]}.nnue",
                           source_url=f"https://tests.stockfishchess.org/api/nn/nn-{hashlib.sha256(b).hexdigest()[:12]}.nnue")
                   for a, b in zip(manifest.engine.embedded_nnue, blobs))
    script = ROOT / "tests/fake_uci_engine.py"
    body = (f"#!/bin/sh\nexec {sys.executable} {script} {assets[0].name} {assets[1].name}\n").encode()
    binary = body + b"#" + blobs[0] + b"\n#" + blobs[1] + b"\n"
    executable = tmp_path / "fake-stockfish"
    executable.write_bytes(binary)
    executable.chmod(0o755)
    engine = replace(manifest.engine, executable_sha256=hashlib.sha256(binary).hexdigest(), embedded_nnue=assets)
    offsets = [binary.index(blobs[0]), binary.index(blobs[1])]
    evidence = dict(executable_sha256=engine.executable_sha256, executable_bytes=len(binary),
        nnue=[dict(name=a.name, sha256=a.sha256, bytes=len(b), embedded_offset=o, source_url=a.source_url)
              for a, b, o in zip(assets, blobs, offsets)])
    evidence_path = tmp_path / "fake-evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    return engine, executable, evidence_path


def specifications():
    # Three train games with several plies each, one validation and one test game.
    return [
        ("train", chess.STARTING_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"], [1, 2, 3, 4, 5], None),
        ("train", chess.STARTING_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3"], [1, 2, 3, 4], None),
        ("train", chess.STARTING_FEN, ["c2c4", "c7c5", "g1f3", "g8f6"], [1, 2, 3], None),
        ("validation", chess.STARTING_FEN, ["g1f3", "d7d5", "g2g3", "g8f6"], [1, 2, 3], None),
        ("test", chess.STARTING_FEN, ["b1c3", "e7e5", "e2e4", "f8b4"], [1, 2], None)]


@pytest.fixture
def workspace(tmp_path):
    engine, executable, evidence = fake_teacher(tmp_path)
    paths = fixture_bundle(tmp_path, specifications(), engine=engine)
    config = tmp_path / "data-config.json"
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1",
        paths={k: str(v) for k, v in dict(paths, teacher_evidence=evidence).items()})))
    # A "prior" bundle of two train references labelled by the fake engine itself.
    return dict(paths=paths, config=config, executable=executable, evidence=evidence, tmp=tmp_path)


def cli(*args, env=None, check=True):
    result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT, env=env or ENV,
                            capture_output=True, text=True, timeout=300)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_membership_is_nested_capped_and_capacity_checked(workspace):
    samples = workspace["paths"]["prepared"] / "samples.jsonl"
    prior = [r for r in read_rows(samples) if r["split"] == "train"][:2]
    refs, capacity, eligible = select_membership(samples, prior, dict(validation=2, test=1, train=6), seed=7, per_game_cap=3)
    assert [key(r) for r in refs[:2]] == [key(r) for r in prior]
    assert [r["split"] for r in refs] == ["train", "train", "validation", "validation", "test"] + ["train"] * 4
    from collections import Counter
    assert max(Counter(r["game_id"] for r in refs).values()) <= 3
    assert capacity["train"] == 9 and capacity["validation"] == 3 and capacity["test"] == 2  # 3 games x cap 3
    assert eligible["train"] == 10
    smaller, _, _ = select_membership(samples, prior, dict(validation=2, test=1, train=4), seed=7, per_game_cap=3)
    assert [key(r) for r in smaller] == [key(r) for r in refs[:5]] + [key(r) for r in refs[5:7]]
    other, _, _ = select_membership(samples, prior, dict(validation=2, test=1, train=6), seed=8, per_game_cap=3)
    assert [key(r) for r in other[5:]] != [key(r) for r in refs[5:]]
    with pytest.raises(ValueError, match="exceeds capacity"):
        select_membership(samples, prior, dict(validation=2, test=1, train=10), seed=7, per_game_cap=3)
    with pytest.raises(ValueError, match="quota below"):
        select_membership(samples, prior, dict(validation=0, test=0, train=1), seed=7, per_game_cap=3)


def test_compact_attempt_keeps_what_the_loader_verifies():
    record = dict(reference={"a": 1}, attempt=1, status="success", returned=dict(
        returned_rows=[dict(multipv=1, depth=1), dict(multipv=2, depth=1), dict(multipv=1, depth=2, pv=["e2e4"])],
        uci_received=["a", "b"], option_capabilities={"x": 1}, uci_sent=["uci"], engine_id={}, effective_options={}))
    compact = compact_attempt(record)
    assert compact["returned"]["returned_rows"] == [dict(multipv=1, depth=2, pv=["e2e4"])]
    assert compact["returned"]["returned_rows_total"] == 3 and "uci_received" not in compact["returned"]
    assert compact["returned"]["uci_sent"] == ["uci"] and len(compact["returned"]["uci_received_sha256"]) == 64
    assert compact["ledger"] == "compact-attempt-ledger-v1"
    assert record["returned"]["uci_received"] == ["a", "b"]  # input untouched


def test_equivalence_criterion_ignores_timing_and_provenance_only():
    base = dict(game_id="g", sample_ply=1, best_move_uci="e2e4", successful_attempt=1,
                target_provenance=dict(executable_sha256="a"), returned_search=dict(time=5, nps=100, nodes=100000, pv=["e2e4"]))
    other = json.loads(json.dumps(base))
    other.update(successful_attempt=2, target_provenance=dict(executable_sha256="b"))
    other["returned_search"].update(time=9, nps=250)
    assert label_differences(base, other) == []
    other["returned_search"]["nodes"] = 99999
    other["best_move_uci"] = "d2d4"
    assert label_differences(base, other) == ["best_move_uci", "returned_search.nodes"]
    summary = compare_labels([base], [other])
    assert summary["compared"] == 1 and summary["mismatched"] == 1
    assert summary["mismatched_fields"] == {"best_move_uci": 1, "returned_search.nodes": 1}


def test_completing_records_uses_the_segment_that_finished_each_reference():
    ref = dict(game_id="g", sample_ply=1)
    seg1 = dict(attempts=[dict(reference=ref, attempt=1, status="failed")], labels=[], exclusions=[])
    seg2 = dict(attempts=[dict(reference=ref, attempt=1, status="failed"), dict(reference=ref, attempt=2, status="success")],
                labels=[dict(game_id="g", sample_ply=1)], exclusions=[])
    labels, exclusions, attempts, discarded = completing_records([seg1, seg2])
    assert set(labels) == {("g", 1)} and not exclusions and discarded == 1
    assert [a["attempt"] for a in attempts[("g", 1)]] == [1, 2]
    with pytest.raises(ValueError, match="twice"):
        completing_records([seg2, seg2])


@pytest.mark.parametrize("policy", ["fresh-process-v1", "persistent-clear-hash-v1"])
def test_end_to_end_run_resume_finalize_prefix_and_loader(workspace, policy):
    out = workspace["tmp"] / f"d09-{policy}"
    cli("freeze", "--data-config", workspace["config"], "--output", out, "--train", 6, "--validation", 3,
        "--test", 2, "--per-game-cap", 3, "--without-d08")
    plan = json.loads((out / "plan.json").read_text())
    assert plan["selected"] == 11 and plan["capacity_after_cap"]["train"] == 9
    refs = list(read_rows(out / "selected.jsonl"))
    assert [r["split"] for r in refs] == ["validation"] * 3 + ["test"] * 2 + ["train"] * 6
    # First segment: interrupt the driver once some references are done, leaving work pending.
    import signal, time
    driver = subprocess.Popen([sys.executable, str(SCRIPT), "run", "--output", str(out), "--executable",
        str(workspace["executable"]), "--teacher-evidence", str(workspace["evidence"]), "--workers", "2",
        "--engine-policy", policy, "--poll-seconds", "0.5"], cwd=ROOT, env=dict(ENV, FAKE_ENGINE_SLEEP="0.4"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        done = sum(1 for p in (out / "segments").glob("done-*.jsonl") for _ in p.read_bytes().splitlines()) if (out / "segments").exists() else 0
        if done >= 2:
            break
        time.sleep(0.2)
    driver.send_signal(signal.SIGINT)
    output, _ = driver.communicate(timeout=60)
    assert driver.returncode != 0, output
    state = json.loads((out / "run.json").read_text())
    done_after_first = state["segments"][0]["completed_after"]
    assert 2 <= done_after_first < 11, output
    # Prefix finalize while work is pending is refused only when the prefix itself is incomplete.
    result = cli("finalize", "--output", out, "--bundle", workspace["tmp"] / "never", check=False)
    assert result.returncode != 0 and "not yet labelled" in result.stdout + result.stderr
    # Second segment completes everything.
    cli("run", "--output", out, "--executable", workspace["executable"], "--teacher-evidence",
        workspace["evidence"], "--workers", 3, "--engine-policy", policy)
    state = json.loads((out / "run.json").read_text())
    assert len(state["segments"]) == 2 and state["segments"][1]["completed_after"] == 11
    bundle = workspace["tmp"] / f"bundle-{policy}"
    cli("finalize", "--output", out, "--bundle", bundle, "--train-prefix", 4)
    summary = json.loads((bundle / "summary.json").read_text())
    assert summary["selected"] == 9 and summary["by_split"] == dict(validation=3, test=2, train=4)
    assert summary["successful_labels"] == 9 and summary["excluded"] == 0
    assert summary["loader_acceptance"] == dict(validation=3, test=2, train=4)
    assert summary["equivalence_with_prior"] is None
    settings = json.loads((bundle / "settings.json").read_text())
    assert settings["process_policy"] == policy and settings["attempt_ledger"] == "compact-attempt-ledger-v1"
    # Attempts in the bundle are compact and every label links to its attempt.
    attempts = list(read_rows(bundle / "attempts.jsonl"))
    assert all(a["ledger"] == "compact-attempt-ledger-v1" and len(a["returned"]["returned_rows"]) == 1 for a in attempts)
    assert all("uci_received" not in a["returned"] for a in attempts)
    labels = list(read_rows(bundle / "labels.jsonl"))
    assert [key(r) for r in labels] == [key(r) for r in read_rows(bundle / "selected.jsonl")]
    assert all(l["target_provenance"]["process_policy"] == policy for l in labels)
    # The production loader accepts the bundle through the data config paths.
    paths = workspace["paths"]
    data = open_dataset(**dict(paths, bundle=bundle, teacher_evidence=workspace["evidence"]), split="train")
    assert len(data) == 4
    # Full finalize into a second bundle; the prefix bundle is a prefix of it.
    full = workspace["tmp"] / f"full-{policy}"
    cli("finalize", "--output", out, "--bundle", full)
    full_train = [key(r) for r in read_rows(full / "selected.jsonl") if r["split"] == "train"]
    prefix_train = [key(r) for r in read_rows(bundle / "selected.jsonl") if r["split"] == "train"]
    assert full_train[:4] == prefix_train
    # Compare the prefix bundle to the full bundle: identical labels on the overlap.
    cli("compare", "--left", bundle, "--right", full)


def test_prior_overlap_equivalence_gate(workspace):
    """A D08-style prior bundle is re-labelled; a tampered prior label fails finalize."""
    out = workspace["tmp"] / "with-prior"
    # Build the prior by labelling two train references with the same fake engine.
    prior_out = workspace["tmp"] / "prior-run"
    cli("freeze", "--data-config", workspace["config"], "--output", prior_out, "--train", 2, "--validation", 0,
        "--test", 0, "--per-game-cap", 3, "--without-d08")
    cli("run", "--output", prior_out, "--executable", workspace["executable"], "--teacher-evidence",
        workspace["evidence"], "--workers", 1, "--engine-policy", "fresh-process-v1")
    prior = workspace["tmp"] / "prior-bundle"
    cli("finalize", "--output", prior_out, "--bundle", prior)
    cli("freeze", "--data-config", workspace["config"], "--output", out, "--d08-bundle", prior, "--train", 5,
        "--validation", 1, "--test", 1, "--per-game-cap", 3)
    refs = list(read_rows(out / "selected.jsonl"))
    assert [key(r) for r in refs[:2]] == [key(r) for r in read_rows(prior / "selected.jsonl")]
    cli("run", "--output", out, "--executable", workspace["executable"], "--teacher-evidence",
        workspace["evidence"], "--workers", 2, "--engine-policy", "persistent-clear-hash-v1")
    bundle = workspace["tmp"] / "with-prior-bundle"
    cli("finalize", "--output", out, "--bundle", bundle)
    summary = json.loads((bundle / "summary.json").read_text())
    assert summary["equivalence_with_prior"]["compared"] == 2 and summary["equivalence_with_prior"]["mismatched"] == 0
    # A D09 label that disagrees with the accepted prior fails the equivalence gate; the
    # bundle is still written for inspection.
    segment = next((out / "segments").glob("labels-w*-s001.jsonl"))
    rows = list(read_rows(segment))
    target = next(r for r in rows if key(r) in {key(p) for p in read_rows(prior / "selected.jsonl")})
    target["best_move_uci"] = "a2a3" if target["best_move_uci"] != "a2a3" else "a2a4"
    write_rows(segment, rows)
    result = cli("finalize", "--output", out, "--bundle", workspace["tmp"] / "tampered", check=False)
    assert result.returncode != 0 and "equivalence failed" in result.stdout + result.stderr
    assert (workspace["tmp"] / "tampered" / "summary.json").exists()
    # The prior bundle's bytes are frozen in the plan; an altered prior is refused outright.
    rows = list(read_rows(prior / "labels.jsonl"))
    rows[0]["best_move_uci"] = "h2h3"
    write_rows(prior / "labels.jsonl", rows)
    result = cli("finalize", "--output", out, "--bundle", workspace["tmp"] / "tampered-prior", check=False)
    assert result.returncode != 0 and "frozen in the plan" in result.stdout + result.stderr


def test_run_data_config_relocates_inputs_and_refuses_different_bytes(workspace):
    """The pod re-resolves inputs by a data config; identical bytes pass, altered bytes do not."""
    import shutil
    out = workspace["tmp"] / "relocated"
    cli("freeze", "--data-config", workspace["config"], "--output", out, "--train", 2, "--validation", 1,
        "--test", 1, "--per-game-cap", 3, "--without-d08")
    paths = workspace["paths"]
    moved = workspace["tmp"] / "elsewhere"
    moved.mkdir()
    relocated = {}
    for name in ("manifest_path", "games_path", "split_audit_path", "prepared"):
        src = Path(paths[name])
        dst = moved / src.name
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        relocated[name] = str(dst)
    relocated["bundle"] = str(paths["bundle"])
    relocated["teacher_evidence"] = str(workspace["evidence"])
    config = moved / "data-config.json"
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1", paths=relocated)))
    cli("run", "--output", out, "--data-config", config, "--executable", workspace["executable"],
        "--teacher-evidence", workspace["evidence"], "--workers", 1, "--engine-policy", "fresh-process-v1")
    state = json.loads((out / "run.json").read_text())
    assert state["paths"]["games_path"] == relocated["games_path"]
    bundle = workspace["tmp"] / "relocated-bundle"
    cli("finalize", "--output", out, "--bundle", bundle)
    assert json.loads((bundle / "settings.json").read_text())["games_path"] == relocated["games_path"]
    assert (bundle / "teacher-evidence.json").read_bytes() == workspace["evidence"].read_bytes()
    # Altered input bytes under the same config are refused before any engine work.
    with (moved / "games.jsonl").open("a") as handle:
        handle.write("\n")
    result = cli("run", "--output", out, "--data-config", config, "--executable", workspace["executable"],
                 "--teacher-evidence", workspace["evidence"], "--workers", 1, "--engine-policy", "fresh-process-v1",
                 check=False)
    assert result.returncode != 0 and ("different input bytes" in result.stderr or "identity" in result.stderr.lower())
