"""Operational regressions: bounded work, ownership, crash recovery, portability."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from chess_rl.adapter import make_state
from chess_rl.search.puct import Node, expand, traverse, _copy_search_state
from chess_rl.runtime import data_paths
from chess_rl.supervised import open_dataset
from .test_supervised import fixture_bundle


def test_frozen_dataset_owns_metadata_and_cached_arrays(tmp_path):
    data = open_dataset(**fixture_bundle(tmp_path), split="train")
    frozen = data.freeze(cache_bytes=100_000)
    expected = frozen[0]
    data._labels[0]["best_move_id"] += 1
    assert frozen[0].policy_class == expected.policy_class
    with pytest.raises(TypeError):
        frozen._labels[0]["best_move_id"] = 1
    with pytest.raises(ValueError, match="frozen"):
        frozen._labels = ()
    expected.features[:] = -100
    assert not np.any(frozen[0].features == -100)
    for i in range(len(frozen)):
        frozen[i]
    _, usage, budget = frozen.cache_info
    assert usage <= budget
    assert not hasattr(frozen, "_sample_cache")
    with pytest.raises(ValueError, match="frozen"):
        del frozen._sealed
    with pytest.raises(TypeError):
        frozen._source_identity["split"] = "test"
    with pytest.raises(TypeError):
        frozen._games[expected.reference[0]]["moves_uci"][0] = "a1a8"
    expected_identity = frozen.source_identity
    data._source_identity["split"] = "test"
    data._games[expected.reference[0]]["moves_uci"] = ["a1a8"]
    data._refs[expected.reference]["sample_ply"] = 999
    expected.state.board.clear()
    expected.legal_mask[:] = False
    expected.wdl[:] = 0
    assert frozen.source_identity == expected_identity
    assert frozen[0].state.board.is_valid()
    assert frozen[0].legal_mask.any() and frozen[0].wdl.sum() == pytest.approx(1)


def test_direct_batch_and_reference_do_not_decode_previous_samples(tmp_path, monkeypatch):
    data = open_dataset(**fixture_bundle(tmp_path), split="train")
    expected = data[2].reference
    calls = []
    old = type(data).__getitem__
    def read(self, index):
        calls.append(index)
        return old(self, index)
    monkeypatch.setattr(type(data), "__getitem__", read)
    assert data.reference(2) == expected
    assert data.batch([2]).references == (expected,)
    assert calls == [2]


def test_owned_copy_retains_long_history_without_revalidating(monkeypatch):
    state = make_state(moves_uci=["g1f3", "g8f6", "f3g1", "f6g8"] * 75)
    node = Node(state)
    state.board._stack[0].pawns = 0
    monkeypatch.setattr("chess_rl.adapter.state.require_legal_move", lambda *a: pytest.fail("history replay"))
    copied = node.state
    assert copied.board.move_stack == state.board.move_stack
    assert copied.board.is_repetition(3)
    copied.board.pop()
    assert len(state.board.move_stack) == 300


def test_public_mutation_cannot_enter_trusted_search_copy():
    state = make_state(moves_uci=["e2e4", "e7e5"])
    node = Node(state)
    state.board.clear()
    with pytest.raises(ValueError):
        Node(state)
    assert not hasattr(state, "_copy_owned")
    with pytest.raises(ValueError, match="search-owned"):
        _copy_search_state(state)
    exposed = node.state
    exposed.board._stack[0].pawns = 0
    exposed.board.clear()
    exposed.initial_fen = "invalid"
    assert node.state.board.is_valid() and len(node.state.board.move_stack) == 2
    def mutating_evaluator(states):
        from chess_rl.search.evaluator import evaluate
        result = evaluate(states)
        states[0].board.clear()
        return result
    expand(node, mutating_evaluator)
    child = traverse(node).leaf
    assert child.state.board.is_valid() and len(child.state.board.move_stack) == 3


def test_child_adoption_has_one_history_copy(monkeypatch):
    import chess
    import chess_rl.search.puct as impl
    node = Node(make_state(moves_uci=["e2e4"]))
    copies = []
    original = chess.Board.copy
    def counted(board, *args, **kwargs):
        copies.append(board)
        return original(board, *args, **kwargs)
    # Isolate adoption from the adapter's independent rule-query copies.
    monkeypatch.setattr(chess.Board, "copy", counted)
    monkeypatch.setattr(impl, "apply_move", lambda state, _: state.board.push_uci("e7e5"))
    child = impl._child_node(node, 0)
    assert len(copies) == 1
    assert child._state.board.move_stack == node._state.board.move_stack + [chess.Move.from_uci("e7e5")]


def test_freezing_rejects_invalid_mutated_target(tmp_path):
    data = open_dataset(**fixture_bundle(tmp_path), split="train")
    data._labels[0]["best_move_id"] = 0  # a1a1 is illegal
    with pytest.raises(ValueError, match="changed after source validation"):
        data.freeze()


def test_portable_paths_follow_config_location_after_move(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    fields = {"bundle", "manifest_path", "games_path", "split_audit_path", "prepared", "teacher_evidence"}
    for name in fields:
        (root / name).touch()
    config = root / "data.json"
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1", paths={n:n for n in fields})))
    moved = tmp_path / "moved"
    root.rename(moved)
    assert all(p.parent == moved for p in data_paths(moved / "data.json").values())


torch = pytest.importorskip("torch")
from .test_mixed_update import sources, optimizer
from .test_selfplay import cpu, fixture_checkpoint, opening
from .test_replay import played
from .test_checkpoints import assert_tree_equal
from .test_checkpoints import saved
from chess_rl.selfplay import Session, Settings, read_game_record
from chess_rl.replay import ReplayBuffer, read_source
from chess_rl.training.checkpoints import ModelConfig
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedState


def test_mixed_step_never_rescans_admitted_dataset(sources, monkeypatch):
    data, _ = sources
    settings = MixedSettings()
    state = MixedState.initial(data, settings, seed=5)
    monkeypatch.setattr("chess_rl.training.mixed_state._supervised_membership",
                        lambda *a: pytest.fail("whole dataset scanned during update"))
    monkeypatch.setattr("chess_rl.training.mixed_state.SourceCursor.__post_init__",
                        lambda *a: pytest.fail("whole permutation validated during update"))
    monkeypatch.setattr("chess_rl.training.mixed_state.MixedState.__post_init__",
                        lambda *a: pytest.fail("whole membership validated during update"))
    model = ModelConfig(d_policy=7, d_value=5).build()
    _, after = mixed_step(model, optimizer(model, settings), data, state)
    assert after.updates == 1


def test_mixed_sources_reject_metadata_assignment_and_deletion(sources):
    data, _ = sources
    with pytest.raises(ValueError, match="frozen mixed"):
        data.replay_id = "0" * 64
    with pytest.raises(ValueError, match="frozen mixed"):
        data._samples = ()
    with pytest.raises(ValueError, match="frozen mixed"):
        del data._sealed
    with pytest.raises(TypeError):
        data._samples[0]["position"]["legal_mask"][0] = True


def test_selfplay_does_not_replay_owned_root_history_per_move(tmp_path, fixture_checkpoint, monkeypatch):
    session = Session.create(tmp_path / "owned-roots", fixture_checkpoint, opening(),
        settings=Settings(2, 5), seed=8, opening_selection_seed=9, journal=True)
    monkeypatch.setattr("chess_rl.adapter.state.require_legal_move",
                        lambda *a: pytest.fail("owned root history replayed"))
    snapshot = session.run()
    assert len(snapshot["game"]["positions"]) == 5


def test_resume_reopens_and_rejects_changed_source_bytes(tmp_path, sources):
    from .test_mixed_update import save_mixed, open_sources
    from chess_rl.training.checkpoints import restore_training, CheckpointError
    data, config = sources
    path = tmp_path / "resume.pt"
    args = save_mixed(path, data)
    # A changed reopened source (valid completed replay) has a new identity.
    from .test_selfplay import create
    from .test_replay import WHITE_WIN
    checkpoint = data._samples[0]["checkpoint"]["path"]
    buffer = ReplayBuffer(config["replay"])
    buffer.ingest(played(tmp_path, checkpoint, WHITE_WIN, name="new-source").snapshot())
    args["mixed_sources"] = open_sources(config)
    with pytest.raises(CheckpointError, match="source identity/membership"):
        restore_training(path, **args)
    # Corrupt authoritative bytes are rejected by the loader before restoration.
    labels = Path(config["dataset"]["bundle"]) / "labels.jsonl"
    labels.write_bytes(labels.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Completed bundle bytes"):
        open_sources(config)


def test_replay_reuses_validation_but_detects_other_writer(tmp_path, fixture_checkpoint, monkeypatch):
    import chess_rl.replay as impl
    snapshot = played(tmp_path, fixture_checkpoint).snapshot()
    buffer = ReplayBuffer.create(tmp_path / "replay.sqlite")
    buffer.ingest(snapshot)
    calls = []
    original = impl.finalize
    def count(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(impl, "finalize", count)
    buffer.export()
    buffer.export()
    assert calls == []
    with sqlite3.connect(buffer.path) as connection:
        connection.execute("UPDATE samples SET payload='{}'")
    with pytest.raises(ValueError):
        buffer.export()
    assert calls


def test_move_journal_crash_before_pointer_is_recoverable(tmp_path, fixture_checkpoint, monkeypatch):
    import chess_rl.selfplay as impl
    session = Session.create(tmp_path / "journal", fixture_checkpoint, opening(), settings=Settings(2, 3),
                             seed=8, opening_selection_seed=9, journal=True)
    original = impl._write
    def crash(path, value, **kwargs):
        if Path(path).name == "boundary.json":
            raise OSError("crash after move publication")
        return original(path, value, **kwargs)
    monkeypatch.setattr(impl, "_write", crash)
    with pytest.raises(OSError):
        session.run()
    assert read_game_record(session.directory)["positions"] == []
    assert (session.directory / "moves/000000.json").exists()
    monkeypatch.setattr(impl, "_write", original)
    recovered = Session.load(session.directory)
    result = recovered.run()
    assert result["game"]["status"] == "truncated"
    assert len(result["game"]["positions"]) == 3
    assert read_source(session.directory) == result
    header = impl._read(session.directory / "boundary.json")
    assert header["positions"] == [] and header["journal_positions"] == 3
    assert (session.directory / "boundary.json").stat().st_size < 4096
    (session.directory / "moves/000001.json").unlink()
    with pytest.raises(FileNotFoundError):
        Session.load(session.directory)


def test_sigkill_after_durable_move_before_pointer_matches_control(tmp_path, saved):
    import select
    import signal
    root = Path(__file__).resolve().parents[1]
    def command(mode, name):
        return [sys.executable, "-m", "tests.selfplay_crash_worker", mode,
                str(tmp_path / name), str(saved["path"]), str(tmp_path / (name + ".json"))]
    control = subprocess.run(command("control", "control"), cwd=root,
                             capture_output=True, text=True, timeout=60)
    assert control.returncode == 0, control.stderr
    worker = subprocess.Popen(command("crash", "recovered"), cwd=root,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([worker.stdout], [], [], 60)
        assert ready and worker.stdout.readline().strip() == "DURABLE_MOVE_READY"
        worker.kill()
        _, stderr = worker.communicate(timeout=10)
        assert worker.returncode == -signal.SIGKILL, stderr
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
    prefix = read_game_record(tmp_path / "recovered")
    assert len(prefix["positions"]) == len(prefix["moves_uci"]) == 2
    assert (tmp_path / "recovered/moves/000002.json").is_file()
    recovered = subprocess.run(command("recover", "recovered"), cwd=root,
                               capture_output=True, text=True, timeout=60)
    assert recovered.returncode == 0, recovered.stderr
    a, b = [json.loads((tmp_path / (n + ".json")).read_text()) for n in ("control", "recovered")]
    assert a == b  # records, PCG64, outcome, configuration/checkpoint/source/sample IDs
    assert b["snapshot"]["game"]["outcome"] == "0-1"
    assert len(b["sample_ids"]) == len(set(b["sample_ids"])) == 4
    files = sorted((tmp_path / "recovered/moves").glob("*.json"))
    assert [p.name for p in files] == [f"{i:06d}.json" for i in range(4)]
    assert read_source(tmp_path / "recovered") == b["snapshot"]


def test_supervised_cli_fresh_process_resume_matches_full(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "data"
    fixture.mkdir()
    config = tmp_path / "data-config.json"
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1",
        paths={k:str(v) for k,v in fixture_bundle(fixture).items()})))
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")
    def run(name, steps, resume=None):
        out = tmp_path / name
        cmd = [sys.executable, "scripts/train_supervised.py", "--device", "cpu", "--data-config", str(config),
               "--output", str(out), "--steps", str(steps), "--batch-size", "2", "--validation-every", "1"]
        if resume:
            cmd += ["--resume", str(resume)]
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return out
    full = run("full", 3)
    prefix = run("prefix", 1)
    resumed = run("resume", 3, prefix / "step-000001.pt")
    a, b = [torch.load(p / "step-000003.pt", weights_only=True) for p in (full, resumed)]
    assert_tree_equal(a, b)
    assert json.loads((resumed / "summary.json").read_text())["test_split_used"] is False
