"""Strength-check preparation and fixture accounting; no pilot matches run."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from chess_rl.adapter import make_state
from chess_rl.evaluation import Checkpoint, Opening, SearchPlayer, SearchSettings, play_game, summarize
from chess_rl.evaluation import pilot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return dict(sha256="2" * 64, match_id=pilot.MATCH_ID, seed=pilot.SEED,
        checkpoints={role: dict(checkpoint_id=f"pilot-step-{step}", path=f"artifacts/p02-cpu-001/step-{step:06d}.pt",
            sha256=str(index) * 64, kind=role)
            for index, (role, step) in enumerate((("supervised-candidate", 1410), ("untrained", 0)))},
        openings=pilot.authored_openings(), ply_cap=512, runtime=pilot.runtime_identity(),
        statistics=dict(bootstrap_seed=123, bootstrap_samples=2000))


def fixture_players(plan):
    def forbidden(states):
        raise AssertionError("Fixture accounting must never call search/evaluation")
    return tuple(SearchPlayer(role, role, f"real-evaluator:{raw['sha256']}", forbidden,
        SearchSettings(32), Checkpoint(raw["checkpoint_id"], "/fixture-not-real.pt", raw["sha256"], role))
        for role, raw in plan["checkpoints"].items())


def terminal_job_record(plan, name="fixture", leg=0, fen="7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"):
    # Terminal rule fixture: no decisions/moves/matches are executed.
    o = Opening.from_state(name, make_state(fen), source="terminal-accounting-fixture")
    job = dict(game_id=f"{name}:leg{leg}", pair_id=f"fixture:{name}", leg=leg,
               game_seed=42 + leg, opening=o.to_dict())
    players = fixture_players(plan)
    white, black = players if leg == 0 else players[::-1]
    record = play_game(o, white, black, pair_id=job["pair_id"], leg=leg, game_seed=job["game_seed"], ply_cap=512)
    assert not record["attempts"]
    return job, record


def terminal_plan(plan, pairs=2):
    values = [terminal_job_record(plan, f"fixture{i}", leg) for i in range(pairs) for leg in (0, 1)]
    plan["openings"] = [values[i][0]["opening"] for i in range(0, len(values), 2)]
    # Use production seed/pair identity even for synthetic terminal fixtures.
    records = []
    for job in pilot.jobs(plan):
        white, black = fixture_players(plan)[::1 if job["leg"] == 0 else -1]
        record = play_game(pilot.opening(job["opening"]), white, black,
            pair_id=job["pair_id"], leg=job["leg"], game_seed=job["game_seed"], ply_cap=512)
        records.append((job, record))
    return records


def test_freeze_diversity_pairing_and_seeds(plan):
    openings = pilot.authored_openings()
    assert len(openings) == len({pilot.opening(o).rule_key() for o in openings}) == 64
    assert len({o["family"] for o in openings}) == 16
    assert {len(o["moves_uci"]) for o in openings} == {5, 7, 9, 11}
    assert all(o["history_complete"] for o in openings)
    schedule = list(pilot.jobs(plan))
    assert len(schedule) == len({j["game_id"] for j in schedule}) == 128
    assert len({j["game_seed"] for j in schedule}) == 128
    for first, second in zip(schedule[::2], schedule[1::2], strict=True):
        assert first["opening"] == second["opening"]
        assert first["pair_id"] == second["pair_id"]
        assert (first["leg"], second["leg"]) == (0, 1)
    assert list(pilot.jobs(plan)) == schedule


def test_worker_checkpoint_colour_routing_without_games(monkeypatch, plan):
    players = fixture_players(plan)
    monkeypatch.setattr(pilot, "_WORKER", (plan, "/unused", players, plan["runtime"]))
    routed, publications = [], []
    def fake_game(o, white, black, **kwargs):
        routed.append((o.to_dict(), white, black, kwargs))
        return dict(status="truncated")
    monkeypatch.setattr(pilot, "play_game", fake_game)
    monkeypatch.setattr(pilot, "publish", lambda *args: publications.append(args))
    pair = list(pilot.jobs(plan))[:2]
    for job in pair:
        pilot.execute_job(job)
    assert routed[0][0] == routed[1][0]
    assert routed[0][1:3] == players
    assert routed[1][1:3] == players[::-1]
    assert all(r[3]["ply_cap"] == 512 for r in routed)
    assert publications[0][1] == pair[0] and publications[1][1] == pair[1]
    assert players[0].metadata()["role"] == "supervised-candidate"
    assert players[1].metadata()["role"] == "untrained"
    assert players[0].settings == players[1].settings == SearchSettings(32)
    assert players[0].metadata()["settings"]["exploration_noise"] is False


def test_worker_one_torch_thread_setup(monkeypatch, tmp_path, plan):
    import chess_rl.runtime
    calls = []
    monkeypatch.setattr(chess_rl.runtime, "configure", lambda device, **kw: calls.append((device, kw)))
    monkeypatch.setattr(pilot, "load_players", lambda *args: fixture_players(plan))
    pilot.initialize_worker(plan, ROOT, tmp_path / "results.sqlite3")
    assert calls == [("cpu", dict(threads=1))]
    assert pilot._WORKER[3]["interop_threads"] == 1
    pilot._WORKER_LOCK.close()
    monkeypatch.setattr(pilot, "_WORKER", None)


def test_completed_scores_colour_pair_uncertainty_and_unfinished_exclusion(plan):
    # Pair scores 1 and 0: CI must resample two pairs, not four independent legs.
    records = []
    for name, winners in (("a", ("white", "black")), ("b", ("black", "white"))):
        for leg, winner in enumerate(winners):
            _, record = terminal_job_record(plan, name, leg)
            record.update(winner=winner, outcome="1-0" if winner == "white" else "0-1")
            records.append(record)
    for leg, status in enumerate(("truncated", "failed")):
        _, record = terminal_job_record(plan, "c", leg)
        record.update(status=status, outcome=None, winner=None)
        records.append(record)
    result = summarize(records, "supervised-candidate", seed=12, bootstrap_samples=10000)
    assert result["completed_game_wdl"] == dict(win=2, draw=0, loss=2)
    assert result["completed_game_denominator"] == 4
    assert result["completed_game_score"] == result["complete_pair_score"] == 0.5
    assert result["complete_pairs"] == 2 and result["incomplete_pairs"] == 1
    assert result["paired_uncertainty"]["interval"] == [0, 1]
    broken = deepcopy(records)
    broken[-2]["outcome"] = "1/2-1/2"
    with pytest.raises(ValueError, match="Unfinished"):
        summarize(broken, "supervised-candidate")


def test_partial_pair_single_leg_draw_and_duplicates(plan):
    _, win = terminal_job_record(plan)
    result = summarize([win], "supervised-candidate", allow_partial=True)
    assert result["completed_game_wdl"] == dict(win=1, draw=0, loss=0)
    assert result["completed_game_score"] == 1 and result["complete_pair_score"] is None
    assert result["paired_uncertainty"] is None
    _, draw = terminal_job_record(plan, "draw", fen="7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    assert summarize([draw], "supervised-candidate", allow_partial=True)["completed_game_score"] == 0.5
    with pytest.raises(ValueError):
        summarize([win, win], "supervised-candidate", allow_partial=True)


def test_durable_restart_and_no_overwrite(tmp_path, plan):
    records = terminal_plan(plan)
    db = tmp_path / "results.sqlite3"
    pilot.initialize_store(db, plan)
    job, record = records[0]
    pilot.publish(db, job, plan, record, 0, plan["runtime"])
    saved = pilot.read_results(db, plan)
    assert list(saved) == [job["game_id"]]
    assert len(pilot.pending_jobs(plan, saved)) == 3
    assert job not in pilot.pending_jobs(plan, saved)
    with pytest.raises(sqlite3.IntegrityError):
        pilot.publish(db, job, plan, record, 0, plan["runtime"])
    pilot.initialize_store(db, plan)
    assert pilot.read_results(db, plan) == saved
    report = pilot.report(db, plan)
    assert report["recorded_games"] == 1 and report["pending_games"] == 127
    assert report["status_rates_among_recorded"]["completed"] == 1
    assert report["complete_pairs"] == 0 and report["paired_uncertainty"] is None


def test_all_recorded_statuses_skipped_on_restart(plan):
    scheduled = list(pilot.jobs(plan))
    values = {j["game_id"]: dict(record=dict(status=status)) for j, status in
        zip(scheduled[:4], ("completed", "truncated", "failed", "interrupted"), strict=True)}
    pending = pilot.pending_jobs(plan, values)
    assert len(pending) == 124 and not any(j in pending for j in scheduled[:4])


def test_four_process_publication_and_crash_after_commit(tmp_path, plan):
    records = terminal_plan(plan)
    db = tmp_path / "results.sqlite3"
    pilot.initialize_store(db, plan)
    code = """
import json, os, sys
from chess_rl.evaluation.pilot import publish
p=json.load(open(sys.argv[1]))
publish(p['db'],p['job'],p['plan'],p['record'],0,p['plan']['runtime'])
os._exit(23)  # commit survives a worker disappearing before parent notification
"""
    processes = []
    for i, (job, record) in enumerate(records):
        path = tmp_path / f"payload{i}.json"
        path.write_text(json.dumps(dict(db=str(db), job=job, record=record, plan=plan)))
        processes.append(subprocess.Popen([sys.executable, "-c", code, str(path)],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 23, stderr.decode()
    assert len(pilot.read_results(db, plan)) == 4
    assert pilot.pending_jobs(plan, pilot.read_results(db, plan)) == []
    result = pilot.report(db, plan)
    assert result["complete_pairs"] == 2 and result["complete_pair_score"] == 0.5
    # A conservative pair interval remains nonzero for identical observed scores.
    assert result["pair_hoeffding_95_interval"] == [0, 1]


def test_uncommitted_crash_is_pending(tmp_path, plan):
    terminal_plan(plan)
    db = tmp_path / "results.sqlite3"
    pilot.initialize_store(db, plan)
    code = """
import sqlite3, os, sys
db=sqlite3.connect(sys.argv[1]); db.execute('BEGIN IMMEDIATE')
db.execute("INSERT INTO games VALUES ('uncommitted','bad','bad')")
os._exit(24)
"""
    assert subprocess.run([sys.executable, "-c", code, str(db)], timeout=30).returncode == 24
    assert pilot.read_results(db, plan) == {}
    assert len(pilot.pending_jobs(plan, {})) == 4


def test_database_and_result_corruption_fail_closed(tmp_path, plan):
    job, record = terminal_plan(plan)[0]
    db = tmp_path / "results.sqlite3"
    pilot.initialize_store(db, plan)
    changed = deepcopy(plan)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="different frozen plan"):
        pilot.initialize_store(db, changed)
    pilot.publish(db, job, plan, record, 0, plan["runtime"])
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE games SET sha256='bad'")
    with pytest.raises(ValueError, match="digest"):
        pilot.read_results(db, plan)


@pytest.mark.parametrize("field", ["route", "outcome", "leg", "opening", "seed"])
def test_record_validation_rejects_wrong_routes_and_adjudication(plan, field):
    job, record = terminal_job_record(plan)
    if field == "route": record["players"]["black"]["role"] = "R"
    if field == "outcome": record.update(status="truncated", outcome="1/2-1/2")
    if field == "leg": record["leg"] = 1
    if field == "opening": record["opening"]["history_complete"] = True
    if field == "seed": record["game_seed"] += 1
    with pytest.raises(ValueError):
        pilot.validate_game(record, job, plan)


def test_output_locks_coordinator_and_orphan_workers(tmp_path):
    import fcntl
    with pilot.run_lock(tmp_path):
        with pytest.raises(ValueError, match="active"):
            with pilot.run_lock(tmp_path):
                pass
    with (tmp_path / "workers.lock").open("a+") as orphan:
        fcntl.flock(orphan, fcntl.LOCK_SH)
        with pytest.raises(ValueError, match="active"):
            with pilot.run_lock(tmp_path):
                pass


def test_real_checkpoint_routing_and_deterministic_forwards(plan):
    paths = [ROOT / raw["path"] for raw in plan["checkpoints"].values()]
    if not all(path.exists() for path in paths):
        pytest.skip("Local completed pilot checkpoints are not distributed with the repository")
    for role, raw in plan["checkpoints"].items():
        raw["sha256"] = pilot.sha_file(ROOT / raw["path"])
    players = pilot.load_players(plan, ROOT)
    from chess_rl.search.real_evaluator import RealEvaluator
    assert all(isinstance(p.evaluator, RealEvaluator) for p in players)
    assert [p.evaluator.checkpoint.checkpoint_id for p in players] == ["pilot-step-1410", "pilot-step-0"]
    assert all(not p.evaluator.checkpoint.model.training for p in players)
    state = make_state()
    outputs = [p.evaluator([state])[0] for p in players]
    assert not np.array_equal(outputs[0].priors, outputs[1].priors)
    for player, output in zip(players, outputs, strict=True):
        again = player.evaluator([state])[0]
        assert np.array_equal(output.priors, again.priors) and np.array_equal(output.wdl, again.wdl)


def test_cli_prepare_verify_report_cannot_launch_matches(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_p02_strength", ROOT / "scripts/check_p02_strength.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "prepare", lambda *args: dict(sha256="fake", planned_games=128, planned_pairs=64))
    monkeypatch.setattr(cli, "run", lambda *args: pytest.fail("prepare launched matches"))
    monkeypatch.setattr(sys, "argv", ["check_p02_strength.py", "prepare", "--plan", str(tmp_path / "plan.json")])
    cli.main()


def test_plan_freeze_and_source_checkpoint_tamper_rejection(tmp_path):
    root = tmp_path / "repo"
    archive = root / "artifacts/p02-cpu-001-preserved"
    archive.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/source.py").write_text("# frozen source\n")
    (root / "scripts").mkdir()
    (root / "scripts/check_p02_strength.py").write_text("# frozen CLI\n")
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("fixture\n")
    checkpoints = {}
    for step in (0, 1410):
        relative = f"artifacts/p02-cpu-001/step-{step:06d}.pt"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(str(step).encode())
        checkpoints[relative] = dict(sha256=pilot.sha_file(path))
    (archive / "manifest.json").write_text(json.dumps(dict(checkpoints=checkpoints, files={})))
    path = root / "plan.json"
    plan = pilot.prepare(root, path)
    assert plan["schedule"] == list(pilot.jobs(plan))
    assert len(plan["schedule"]) == 128
    with pytest.raises(ValueError, match="never overwrite"):
        pilot.prepare(root, path)
    source = root / "src/source.py"
    source.write_text("# altered\n")
    with pytest.raises(ValueError, match="source differs"):
        pilot.load_plan(path, root)
    source.write_text("# frozen source\n")
    checkpoint = root / "artifacts/p02-cpu-001/step-000000.pt"
    checkpoint.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="routing/hash"):
        pilot.load_plan(path, root)


def test_coordinator_uses_four_spawn_workers_and_skips_finished(monkeypatch, tmp_path, plan):
    from concurrent.futures import Future
    records = terminal_plan(plan)
    saved_records = {job["game_id"]: (job, record) for job, record in records}
    db = tmp_path / "results.sqlite3"
    pilot.initialize_store(db, plan)
    job, record = records[0]
    pilot.publish(db, job, plan, record, 0, plan["runtime"])
    submissions, options = [], []
    class FixturePool:
        def __init__(self, **kwargs): options.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, job):
            submissions.append(job["game_id"])
            _, record = saved_records[job["game_id"]]
            pilot.publish(db, job, plan, record, 0, plan["runtime"])
            future = Future()
            future.set_result((job["game_id"], record["status"]))
            return future
    monkeypatch.setattr(pilot, "ProcessPoolExecutor", FixturePool)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "test-previous-value")
    pilot.run(plan, ROOT, tmp_path)
    assert options[0]["max_workers"] == 4
    assert options[0]["mp_context"].get_start_method() == "spawn"
    assert options[0]["initializer"] is pilot.initialize_worker
    assert len(submissions) == 3 and job["game_id"] not in submissions
    # A second run does not even construct a worker pool.
    pilot.run(plan, ROOT, tmp_path)
    assert len(options) == 1 and len(submissions) == 3
