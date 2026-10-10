"""Operational regressions: bounded work, ownership, crash recovery, portability."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from chess_rl.adapter import make_state
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
    assert frozen._cache_usage[0] <= frozen._cache_budget


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
    monkeypatch.setattr("chess_rl.adapter.state.require_legal_move", lambda *a: pytest.fail("history replay"))
    copied = state._copy_owned()
    assert copied.board.move_stack == state.board.move_stack
    assert copied.board.is_repetition(3)
    copied.board.pop()
    assert len(state.board.move_stack) == 300


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
    model = ModelConfig(d_policy=7, d_value=5).build()
    _, after = mixed_step(model, optimizer(model, settings), data, state)
    assert after.updates == 1


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
