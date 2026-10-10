"""Bounded R02 fixtures: real R01 records with controlled shared search."""
import copy
import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from chess_rl.adapter import encode_position, make_state
from chess_rl.evaluation.matches import digest
from chess_rl.replay import ReplayBuffer, finalize, read_source
from chess_rl.selfplay import Phase, _write
from .test_selfplay import cpu, fixture_checkpoint, create, controlled
from .helpers import CYCLE


BLACK_WIN = ["f2f3", "e7e5", "g2g4", "d8h4"]
WHITE_WIN = ["e2e4", "f7f6", "d2d4", "g7g5", "d1h5"]


def played(tmp_path, checkpoint, moves=BLACK_WIN, *, name="game", state=None):
    session = create(tmp_path, checkpoint, state, cap=len(moves) + (len(state.board.move_stack) if state else 0),
                     budget=1, name=name, phases=(Phase(0, 0., 0.),))
    for uci in moves:
        session.evaluator = controlled(uci)
        session.run(stop_after_moves=1)
    return session


@pytest.mark.parametrize("moves,winner", [(BLACK_WIN, "black"), (WHITE_WIN, "white")])
def test_exact_both_color_win_loss_and_preservation(tmp_path, fixture_checkpoint, moves, winner):
    session = played(tmp_path, fixture_checkpoint, moves)
    snapshot = session.snapshot()
    before = copy.deepcopy(snapshot)
    result = finalize(snapshot)
    assert result["status"] == "completed" and snapshot["game"]["winner"] == winner
    assert len(result["samples"]) == len(moves)
    for ply, sample in enumerate(result["samples"]):
        position = snapshot["game"]["positions"][ply]
        assert sample["wdl"] == ([1, 0, 0] if position["side_to_move"] == winner else [0, 0, 1])
        assert sample["position"] == position
        assert sample["moves_uci"] == moves[:ply]
        assert sample["checkpoint"] == snapshot["configuration"]["checkpoint"]
        assert sample["opening"] == snapshot["configuration"]["opening"]
        assert sample["search_settings"] == snapshot["configuration"]["settings"]
        assert sample["contracts"]["wdl_order"] == ["win", "draw", "loss"]
        visits = np.array(position["root_visits"])
        np.testing.assert_array_equal(position["visit_policy"], visits / visits.sum())
    assert snapshot == before
    assert finalize(read_source(session.directory)) == result


def test_full_actual_history_prefix_and_draw_both_colors(tmp_path, fixture_checkpoint):
    state = make_state(moves_uci=CYCLE)
    session = played(tmp_path, fixture_checkpoint, CYCLE, state=state)
    result = finalize(session.snapshot())
    assert result["status"] == "completed" and len(result["samples"]) == 4
    for sample in result["samples"]:
        assert sample["wdl"] == [0, 1, 0]
        root = make_state(sample["initial_fen"], sample["moves_uci"], history_complete=True)
        assert len(root.board.move_stack) == sample["position"]["ply"]
        assert sample["moves_uci"][:4] == CYCLE
        encoded = encode_position(root)
        assert sample["available_snapshots"] == min(8, len(root.board.move_stack) + 1)
        # A current FEN cannot recover history snapshots or prefix repetition.
        isolated = encode_position(make_state(root.board.fen()))
        assert not np.array_equal(encoded.features, isolated.features)
    assert {s["position"]["side_to_move"] for s in result["samples"]} == {"white", "black"}


def test_soft_policy_preserved_and_unit_sum_corruption_rejected(tmp_path, fixture_checkpoint):
    state = make_state("4k3/8/8/8/8/8/8/R3K3 w - - 99 1", history_complete=True)
    session = create(tmp_path, fixture_checkpoint, state, cap=1, budget=3,
                     phases=(Phase(0, 0., 0.),))
    snapshot = session.run()
    sample = finalize(snapshot)["samples"][0]
    position = sample["position"]
    visits = np.array(position["root_visits"])
    assert np.count_nonzero(visits) == 3
    np.testing.assert_array_equal(position["visit_policy"], visits / 3)
    assert sample["wdl"] == [0, 1, 0]
    altered = copy.deepcopy(snapshot)
    p = altered["game"]["positions"][0]
    support = np.flatnonzero(visits)
    p["visit_policy"][int(support[0])] += 1 / 6
    p["visit_policy"][int(support[1])] -= 1 / 6
    assert sum(p["visit_policy"]) == pytest.approx(1)
    with pytest.raises(ValueError, match="inconsistent"): finalize(altered)


@pytest.mark.parametrize("state", [
    make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", history_complete=True),
    make_state(moves_uci=CYCLE * 2),
])
def test_immediate_terminals_zero_positions(tmp_path, fixture_checkpoint, state):
    session = create(tmp_path, fixture_checkpoint, state, cap=len(state.board.move_stack))
    snapshot = session.run()
    assert snapshot["game"]["status"] == "completed"
    assert finalize(snapshot)["samples"] == []
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    assert buffer.ingest(snapshot)["added_samples"] == 0
    assert buffer.export()["summary"]["training_positions"] == 0


@pytest.mark.parametrize("interrupted", [True, False])
def test_unfinished_excluded_unlabeled_and_resumed_admission(tmp_path, fixture_checkpoint, interrupted):
    session = create(tmp_path, fixture_checkpoint, cap=4 if interrupted else 1,
                     budget=1, phases=(Phase(0, 0., 0.),))
    session.evaluator = controlled(BLACK_WIN[0])
    session.run(stop_after_moves=1)
    snapshot = session.snapshot()
    result = finalize(snapshot)
    status = "interrupted" if interrupted else "truncated"
    assert result["status"] == status and result["samples"] == []
    assert snapshot["game"]["winner"] is None and snapshot["game"]["outcome"] is None
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    assert buffer.ingest(snapshot)["disposition"] == "excluded-" + status
    assert not buffer.export()["samples"]
    if interrupted:
        for uci in BLACK_WIN[1:]:
            session.evaluator = controlled(uci)
            session.run(stop_after_moves=1)
        assert buffer.ingest(session.snapshot())["added_samples"] == 4
        assert buffer.export()["sources"][0]["snapshot"] == snapshot


@pytest.mark.parametrize("mutation", ["mask", "mask-int", "visits", "negative-visits", "bool-visits",
    "zero-visits", "policy", "nonfinite-policy", "bool-policy", "illegal-visit", "ply", "bool-ply",
    "history", "root-fen", "root-player", "outcome", "winner", "status", "final-fen", "move",
    "config-hash", "contracts", "checkpoint-contracts", "checkpoint-hash", "checkpoint-config", "extra-wdl"])
def test_corruption_rejected_before_any_publication(tmp_path, fixture_checkpoint, mutation):
    snapshot = played(tmp_path, fixture_checkpoint).snapshot()
    game, cfg = snapshot["game"], snapshot["configuration"]
    p = game["positions"][0]
    selected = p["move_id"]
    if mutation == "mask": p["legal_mask"][selected] = False
    elif mutation == "mask-int": p["legal_mask"][selected] = 1
    elif mutation == "visits": p["root_visits"][selected] += 1
    elif mutation == "negative-visits": p["root_visits"][selected] = -1
    elif mutation == "bool-visits": p["root_visits"][selected] = True
    elif mutation == "zero-visits": p["root_visits"] = [0] * 4352
    elif mutation == "policy": p["visit_policy"][selected] = .5
    elif mutation == "nonfinite-policy": p["visit_policy"][selected] = float("nan")
    elif mutation == "bool-policy": p["visit_policy"][selected] = True
    elif mutation == "illegal-visit": p["root_visits"][0] = 1
    elif mutation == "ply": p["ply"] = 1
    elif mutation == "bool-ply": p["ply"] = False
    elif mutation == "history": cfg["opening"]["history_complete"] = False
    elif mutation == "root-fen": p["root_fen"] = game["final_fen"]
    elif mutation == "root-player": p["side_to_move"] = "black"
    elif mutation == "outcome": game["outcome"] = "1-0"
    elif mutation == "winner": game["winner"] = "white"
    elif mutation == "status": game["status"] = "truncated"
    elif mutation == "final-fen": game["final_fen"] = p["root_fen"]
    elif mutation == "move": game["moves_uci"][0] = "e2e5"
    elif mutation == "config-hash": game["configuration_sha256"] = "0" * 64
    elif mutation == "contracts": cfg["contracts"]["wdl_order"] = ["loss", "draw", "win"]
    elif mutation == "checkpoint-contracts": cfg["checkpoint"]["contracts"]["ending_policy"] = "old"
    elif mutation == "checkpoint-hash": cfg["checkpoint"]["sha256"] = "bad"
    elif mutation == "checkpoint-config": cfg["checkpoint"]["model_config"]["relative_enabled"] = False
    elif mutation == "extra-wdl": p["wdl"] = [1, 0, 0]
    if mutation != "config-hash": game["configuration_sha256"] = digest(cfg)
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    with pytest.raises(ValueError): buffer.ingest(snapshot)
    assert buffer.export()["summary"]["source_counts"] == dict(
        admitted=0, duplicate=0, **{"excluded-truncated": 0, "excluded-interrupted": 0})


def test_duplicates_first_commit_wins_stable_ids_and_provenance(tmp_path, fixture_checkpoint):
    snapshot = played(tmp_path, fixture_checkpoint).snapshot()
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    assert buffer.ingest(snapshot)["added_samples"] == 4
    original = buffer.export()["samples"]
    assert buffer.ingest(snapshot)["repeated"]
    duplicate = copy.deepcopy(snapshot)
    duplicate["configuration"]["checkpoint"]["checkpoint_id"] = "another-checkpoint"
    duplicate["configuration"]["checkpoint"]["sha256"] = "a" * 64
    duplicate["game"]["configuration_sha256"] = digest(duplicate["configuration"])
    assert buffer.ingest(duplicate)["disposition"] == "duplicate"
    assert buffer.export()["sources"][1]["snapshot"] == duplicate
    assert buffer.export()["samples"] == original
    assert ReplayBuffer(buffer.path).export() == buffer.export()
    another = ReplayBuffer.create(tmp_path / "other.sqlite")
    another.ingest(snapshot)
    assert another.export()["samples"] == original
    # A genuinely different trajectory remains admissible.
    different = played(tmp_path, fixture_checkpoint, WHITE_WIN, name="white").snapshot()
    assert buffer.ingest(different)["added_samples"] == 5


def test_atomic_rollback_and_fresh_process_restart(tmp_path, fixture_checkpoint):
    snapshot = played(tmp_path, fixture_checkpoint).snapshot()
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    with sqlite3.connect(buffer.path) as connection:
        connection.execute("CREATE TRIGGER injected BEFORE INSERT ON samples BEGIN SELECT RAISE(ABORT,'injected failure'); END")
    with pytest.raises(sqlite3.IntegrityError): buffer.ingest(snapshot)
    assert buffer.export()["sources"] == []
    with sqlite3.connect(buffer.path) as connection: connection.execute("DROP TRIGGER injected")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(snapshot))
    code = "from chess_rl.replay import ReplayBuffer; import json,sys; b=ReplayBuffer(sys.argv[1]); print(json.dumps(b.ingest(json.load(open(sys.argv[2]))),sort_keys=True))"
    first = subprocess.run([sys.executable, "-B", "-c", code, str(buffer.path), str(source)],
                           check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(first.stdout)["added_samples"] == 4
    exported = buffer.export()
    second = subprocess.run([sys.executable, "-B", "-c", code, str(buffer.path), str(source)],
                            check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(second.stdout)["repeated"]
    assert buffer.export() == exported


@pytest.mark.parametrize("after_commit", [False, True])
def test_sigkill_atomic_admission_and_restart(tmp_path, fixture_checkpoint, after_commit):
    snapshot = played(tmp_path, fixture_checkpoint).snapshot()
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(snapshot))
    code = '''import json,os,signal,sqlite3,sys
from chess_rl.replay import ReplayBuffer
class CrashConnection(sqlite3.Connection):
    def execute(self, sql, *args):
        result = super().execute(sql, *args)
        if sql.startswith("INSERT INTO samples"):
            os.kill(os.getpid(), signal.SIGKILL)
        return result
if sys.argv[3] == "before":
    def connect(self):
        connection = sqlite3.connect(self.path, factory=CrashConnection)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection
    ReplayBuffer._connect = connect
b=ReplayBuffer(sys.argv[1])
b.ingest(json.load(open(sys.argv[2])))
os.kill(os.getpid(), signal.SIGKILL)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(buffer.path), str(source),
        "after" if after_commit else "before"], capture_output=True, text=True, timeout=30)
    assert result.returncode == -9, result.stderr
    restored = ReplayBuffer(buffer.path)
    assert restored.export()["summary"]["training_positions"] == (4 if after_commit else 0)
    receipt = restored.ingest(snapshot)
    assert receipt["repeated"] is after_commit
    assert restored.export()["summary"]["training_positions"] == 4


@pytest.mark.parametrize("table", ["samples", "sources", "games", "metadata"])
def test_restart_reaudits_database_corruption(tmp_path, fixture_checkpoint, table):
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    buffer.ingest(played(tmp_path, fixture_checkpoint).snapshot())
    with sqlite3.connect(buffer.path) as connection:
        if table == "games": connection.execute("UPDATE games SET sample_count=999")
        elif table == "samples":
            sid, payload = connection.execute("SELECT sample_id,payload FROM samples LIMIT 1").fetchone()
            sample = json.loads(payload); sample["wdl"] = [0, 1, 0]
            connection.execute("UPDATE samples SET payload=? WHERE sample_id=?", (json.dumps(sample), sid))
        elif table == "sources": connection.execute("UPDATE sources SET disposition='duplicate'")
        else: connection.execute("UPDATE metadata SET payload='{}'")
    with pytest.raises(ValueError): ReplayBuffer(buffer.path)


def test_source_envelope_corruption_and_no_overwrite(tmp_path, fixture_checkpoint):
    session = played(tmp_path, fixture_checkpoint)
    path = session.directory / "boundary.json"
    envelope = json.loads(path.read_text()); envelope["content"]["winner"] = "white"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"): read_source(session.directory)
    # Even recomputing the envelope checksum cannot bypass semantic audit.
    _write(path, envelope["content"], replace=True)
    with pytest.raises(ValueError): read_source(session.directory)
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    before = buffer.path.read_bytes()
    with pytest.raises(FileExistsError): ReplayBuffer.create(buffer.path)
    assert before == buffer.path.read_bytes()


def test_open_audit_keeps_one_snapshot_during_concurrent_commit(tmp_path, fixture_checkpoint, monkeypatch):
    donor = ReplayBuffer.create(tmp_path / "donor.sqlite")
    donor.ingest(played(tmp_path, fixture_checkpoint).snapshot())
    with sqlite3.connect(donor.path) as connection:
        rows = {table: connection.execute("SELECT * FROM " + table).fetchall()
                for table in ("sources", "games", "samples")}
    buffer = ReplayBuffer.create(tmp_path / "buffer.sqlite")
    with sqlite3.connect(buffer.path) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    committed = False
    class Interleaved(sqlite3.Connection):
        def execute(self, sql, *args):
            nonlocal committed
            cursor = super().execute(sql, *args)
            if sql == "SELECT * FROM sources ORDER BY ordinal" and not committed:
                # WAL allows a writer to commit while this reader retains its
                # empty source snapshot. Subsequent tables must see that same
                # snapshot, not falsely report target/admission corruption.
                with sqlite3.connect(buffer.path) as writer:
                    for table, records in rows.items():
                        placeholders = ",".join("?" for _ in records[0])
                        writer.executemany("INSERT INTO " + table + " VALUES(" + placeholders + ")", records)
                committed = True
            return cursor
    def connect(self):
        connection = sqlite3.connect(self.path, factory=Interleaved)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection
    monkeypatch.setattr(ReplayBuffer, "_connect", connect)
    restored = ReplayBuffer(buffer.path)
    assert committed
    assert restored.export()["summary"]["training_positions"] == 4
