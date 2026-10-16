"""Self-play campaign driver: real checkpoint, real worker processes, crash recovery, streaming ingest."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")

from chess_rl import selfplay_campaign as campaign
from chess_rl.evaluation.matches import Opening
from chess_rl.replay import ReplayBuffer
from chess_rl.selfplay import Phase, Session, Settings
from .test_checkpoints import saved

ROOT = Path(__file__).resolve().parents[1]
# Every legal move from here reaches the fifty-move rule: each game completes in one ply.
FIFTY = Opening("fifty", "4k3/8/8/8/8/8/8/R3K3 w - - 99 1", (), True,
                "test setup position; game genuinely begins here")
SMALL = Settings(2, 40, phases=(Phase(0, 1.0, 0.25), Phase(4, 0.0, 0.25)))


def make(tmp_path, saved, name, *, games, settings=SMALL, opening=campaign.STANDARD_OPENING):
    return tmp_path / name, campaign.create(tmp_path / name, saved["path"], games=games, seed=11,
        settings=settings, opening=opening, purpose="test campaign")


def tree(directory):
    """Bytes of every committed game file, keyed by relative path."""
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(Path(directory, "games").rglob("*.json"))
            if not p.name.startswith(".")}


def cli(*args):
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_frozen_settings_match_the_round_plan():
    settings = campaign.frozen_settings(64)
    assert (settings.simulation_budget, settings.ply_cap, settings.c_puct, settings.unvisited_value) == (64, 512, 1.5, 0.0)
    assert [(p.start_ply, p.temperature, p.noise_fraction, p.dirichlet_alpha) for p in settings.phases] == [
        (0, 1.0, 0.25, 0.3), (30, 0.0, 0.25, 0.3)]


def test_create_freezes_plan_and_refuses_reuse(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "c", games=3)
    assert campaign.load(directory) == plan
    assert len({g["seed"] for g in plan["games"]}) == 3
    assert not any((directory / g["directory"]).exists() for g in plan["games"])
    with pytest.raises(FileExistsError):
        make(tmp_path, saved, "c", games=3)
    with pytest.raises(ValueError):
        campaign.create(tmp_path / "x", saved["path"], games=1, seed=1, purpose="p", simulations=8, settings=SMALL)


def test_run_archive_and_streaming_ingest_match_whole_ingest(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "fifty", games=4, opening=FIFTY)
    report = campaign.run(directory, workers=2, heavy_lock=tmp_path / "heavy.lock", log=lambda *_: None)
    assert report["status_counts"] == dict(completed=4, truncated=0, interrupted=0, not_started=0)
    summary = campaign.archive(directory)
    lines = (directory / "games.jsonl").read_text().splitlines()
    assert len(lines) == 4 and summary["status_counts"] == {"completed": 4}
    assert [json.loads(l)["game"]["configuration_sha256"] for l in lines]  # one snapshot per line
    assert campaign.archive(directory) == summary  # idempotent once complete
    # Same games through the list path and the streaming path give identical buffers.
    (tmp_path / "games.json").write_text(json.dumps([json.loads(l) for l in lines]))
    for records, buffer in (("games.json", "whole.sqlite"), (str(directory / "games.jsonl"), "stream.sqlite")):
        result = cli("scripts/run_r02.py", "--records", str(tmp_path / records), "--buffer",
                     str(tmp_path / buffer), "--create", "--report", str(tmp_path / (buffer + ".report.json")))
        assert result.returncode == 0, result.stderr
    whole, stream = ReplayBuffer(tmp_path / "whole.sqlite"), ReplayBuffer(tmp_path / "stream.sqlite")
    assert whole.export() == stream.export()
    assert stream.summary() == stream.export()["summary"]
    assert list(stream.iter_samples()) == stream.export()["samples"]
    receipts = json.loads((tmp_path / "stream.sqlite.report.json").read_text())["receipts"]
    assert [r["disposition"] for r in receipts].count("admitted") == summary["status_counts"]["completed"] - summary["duplicate_completed_games"]


def test_rerun_of_finished_campaign_changes_nothing(tmp_path, saved):
    directory, _ = make(tmp_path, saved, "done", games=2, settings=Settings(2, 6))
    campaign.run(directory, workers=2, log=lambda *_: None)
    before = tree(directory)
    report = campaign.run(directory, workers=2, log=lambda *_: None)
    assert tree(directory) == before and report["status_counts"]["truncated"] == 2


def test_sigkill_mid_campaign_keeps_finished_games_and_resumes(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "kill", games=6,
                           settings=Settings(2, 10, phases=(Phase(0, 1.0, 0.25), Phase(4, 0.0, 0.25))))
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    worker = subprocess.Popen([sys.executable, "scripts/run_selfplay_campaign.py", "run", "--directory",
        str(directory), "--workers", "2", "--heavy-lock", str(tmp_path / "heavy.lock")], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    events = directory / "events.jsonl"
    deadline = time.time() + 600
    try:
        while time.time() < deadline:
            ended = [json.loads(l) for l in events.read_text().splitlines()] if events.exists() else []
            if sum(e["event"] == "game-end" for e in ended) >= 1:
                break
            assert worker.poll() is None, worker.stderr.read().decode()
            time.sleep(0.02)
        os.killpg(worker.pid, signal.SIGKILL)  # parent and workers, no cleanup
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
    finished = {g["directory"] for g in plan["games"] if campaign._finished(directory / g["directory"])}
    assert 1 <= len(finished) < 6
    kept = {k: v for k, v in tree(directory).items() if "/".join(k.split("/")[:2]) in finished}
    assert kept
    report = campaign.run(directory, workers=2, log=lambda *_: None)
    assert sum(report["status_counts"][s] for s in campaign.FINAL) == 6
    after = tree(directory)
    assert all(after[k] == v for k, v in kept.items())
    summary = campaign.archive(directory)
    assert summary["games"] == 6 and summary["positions"] == sum(
        len(json.loads(l)["game"]["moves_uci"]) for l in (directory / "games.jsonl").read_text().splitlines())


def test_changed_checkpoint_and_held_locks_are_refused(tmp_path, saved):
    copy = tmp_path / "copy.pt"
    copy.write_bytes(Path(saved["path"]).read_bytes())
    directory = tmp_path / "changed"
    campaign.create(directory, copy, games=1, seed=1, settings=SMALL, purpose="test")
    with campaign._lock(tmp_path / "heavy.lock", "held"):
        with pytest.raises(RuntimeError, match="heavy job"):
            campaign.run(directory, workers=1, heavy_lock=tmp_path / "heavy.lock", log=lambda *_: None)
    copy.write_bytes(copy.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        campaign.run(directory, workers=1, log=lambda *_: None)
    assert not (directory / "games/0000").exists()


def test_archive_refuses_unfinished_games(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "partial", games=2)
    with pytest.raises(ValueError, match="2 games unfinished"):
        campaign.archive(directory)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    game = plan["games"][0]
    session = Session.create(directory / game["directory"], saved["path"], Opening(**plan["opening"]),
        settings=Settings.from_dict(plan["settings"]), seed=game["seed"], opening_selection_seed=plan["seed"],
        journal=True)
    session.run(stop_after_moves=1)
    with pytest.raises(ValueError, match="2 games unfinished"):
        campaign.archive(directory)
    assert not (directory / "games.jsonl").exists()


def finish(directory):
    """The job scripts' archive + ingest step, run exactly as sourced by the sbatch files."""
    env = dict(os.environ, PY=sys.executable, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}",
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["bash", "-euo", "pipefail", "-c",
        'source jobs/b200-selfplay/finish.sh; finish_campaign "$0"', str(directory)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_archive_and_ingest_each_resume_after_a_stop(tmp_path, saved):
    directory, _ = make(tmp_path, saved, "finish", games=3, opening=FIFTY)
    campaign.run(directory, workers=2, log=lambda *_: None)
    archive = directory / "games.jsonl"
    # Stopped after the archive was published but before its summary: redone, same bytes.
    summary = campaign.archive(directory)
    first = archive.read_bytes()
    (directory / "games.summary.json").unlink()
    assert campaign.archive(directory)["archive_sha256"] == summary["archive_sha256"]
    assert archive.read_bytes() == first
    # Stopped after archiving, before any ingest (the reported bug): finish still ingests.
    # Stopped mid-ingest with a partly filled buffer: finish completes it without --create.
    buffer = ReplayBuffer.create(directory / "replay.sqlite")
    buffer.ingest(json.loads(first.decode().splitlines()[0]))
    buffer.close()
    result = finish(directory)
    assert result.returncode == 0, result.stderr
    report = json.loads((directory / "ingest-report.json").read_text())
    assert [r["repeated"] for r in report["receipts"]] == [True, False, False]
    control = ReplayBuffer.create(tmp_path / "control.sqlite")
    for line in first.decode().splitlines():
        control.ingest(json.loads(line))
    assert ReplayBuffer(directory / "replay.sqlite").export()["samples"] == control.export()["samples"]
    # Everything complete: a further finish changes nothing.
    before = (directory / "ingest-report.json").read_bytes(), archive.read_bytes()
    assert finish(directory).returncode == 0
    assert ((directory / "ingest-report.json").read_bytes(), archive.read_bytes()) == before
    # A truncated report (stop mid-write under the old non-atomic writer) is not taken as done.
    report_path = directory / "ingest-report.json"
    report_path.write_bytes(before[0][:len(before[0]) // 2])
    result = finish(directory)
    assert result.returncode == 0, result.stderr
    redone = json.loads(report_path.read_text())
    assert redone["summary"] == report["summary"] and all(r["repeated"] for r in redone["receipts"])
    assert len(list(directory.glob(".ingest-report.json.invalid-*"))) == 1
    assert not list(directory.glob(".*.pending-*"))


def test_ingest_outputs_are_published_whole_and_never_overwritten(tmp_path, saved):
    directory, _ = make(tmp_path, saved, "publish", games=2, opening=FIFTY)
    campaign.run(directory, workers=2, log=lambda *_: None)
    campaign.archive(directory)
    args = ["scripts/run_r02.py", "--records", str(directory / "games.jsonl"), "--buffer",
            str(tmp_path / "b.sqlite"), "--report", str(tmp_path / "r.json"), "--samples", str(tmp_path / "s.jsonl")]
    assert cli(*args, "--create").returncode == 0
    report = json.loads((tmp_path / "r.json").read_text())
    import hashlib
    assert report["samples_jsonl_sha256"] == hashlib.sha256((tmp_path / "s.jsonl").read_bytes()).hexdigest()
    before = (tmp_path / "r.json").read_bytes()
    assert cli(*args).returncode != 0  # existing samples file: refused before anything is written
    assert (tmp_path / "r.json").read_bytes() == before
    assert not list(tmp_path.glob(".*.pending-*"))


def test_one_dead_worker_does_not_stall_the_campaign(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "dead", games=4,
                           settings=Settings(2, 10, phases=(Phase(0, 1.0, 0.25), Phase(4, 0.0, 0.25))))
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    parent = subprocess.Popen([sys.executable, "scripts/run_selfplay_campaign.py", "run", "--directory",
        str(directory), "--workers", "2", "--heavy-lock", str(tmp_path / "heavy.lock")], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.time() + 300
        while not any((directory / "games").glob("*/moves/000000.json")):
            assert time.time() < deadline and parent.poll() is None
            time.sleep(0.05)
        # Kill one pool worker only, found through the pid files workers write at start.
        workers = sorted(int(p.name) for p in (directory / ".workers").iterdir())
        assert workers, "no worker pid files"
        os.kill(workers[0], signal.SIGKILL)
        _, stderr = parent.communicate(timeout=900)
        assert parent.returncode == 0, stderr.decode()
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
    events = [json.loads(l) for l in (directory / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "pool-broken" for e in events)
    assert all(campaign._finished(directory / g["directory"]) for g in plan["games"])
    assert campaign.archive(directory)["games"] == 4


def test_ingest_archives_builds_one_buffer_from_several_campaigns_and_resumes(tmp_path, saved):
    halves = []
    for name, seed in (("a", 21), ("b", 22)):
        directory = tmp_path / name
        campaign.create(directory, saved["path"], games=2, seed=seed, settings=SMALL, opening=FIFTY, purpose="half")
        campaign.run(directory, workers=2, log=lambda *_: None)
        campaign.archive(directory)
        halves.append(directory)
    env = dict(os.environ, PY=sys.executable, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}",
               PYTHONDONTWRITEBYTECODE="1")
    def ingest():
        return subprocess.run(["bash", "-euo", "pipefail", "-c",
            'source jobs/b200-selfplay/finish.sh; ingest_archives "$@"', "_", str(tmp_path / "union"),
            *map(str, halves)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    result = ingest()
    assert result.returncode == 0, result.stderr
    control = ReplayBuffer.create(tmp_path / "control.sqlite")
    for directory in halves:
        for line in (directory / "games.jsonl").read_text().splitlines():
            control.ingest(json.loads(line))
    union = tmp_path / "union"
    assert ReplayBuffer(union / "replay.sqlite").export()["samples"] == control.export()["samples"]
    reports = sorted(p.name for p in union.glob("ingest-*.json"))
    assert reports == ["ingest-a.json", "ingest-b.json"]
    before = {p.name: p.read_bytes() for p in union.iterdir()}
    assert ingest().returncode == 0
    assert {p.name: p.read_bytes() for p in union.iterdir()} == before
    (union / "ingest-b.json").write_text("{")
    assert ingest().returncode == 0
    assert json.loads((union / "ingest-b.json").read_text())["summary"] == control.export()["summary"]
    assert len(list(union.glob(".ingest-b.json.invalid-*"))) == 1


def test_deadline_stops_running_games_and_keeps_their_moves(tmp_path, saved):
    directory, plan = make(tmp_path, saved, "deadline", games=2,
                           settings=Settings(2, 200, phases=(Phase(0, 1.0, 0.25), Phase(4, 0.0, 0.25))))
    # A deadline already past starts nothing.
    report = campaign.run(directory, workers=2, deadline=time.time() - 1, log=lambda *_: None)
    assert report["status_counts"]["not_started"] == 2
    # Lengthen the window until it closes while games are running (machine speeds differ).
    for seconds in (3, 6, 12, 24, 48, 96):
        report = campaign.run(directory, workers=2, deadline=time.time() + seconds, log=lambda *_: None)
        if report["status_counts"]["interrupted"]:
            break
    assert report["status_counts"]["interrupted"] >= 1  # stopped mid-game, not at a game boundary
    events = [json.loads(l) for l in (directory / "events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "deadline-reached" for e in events) >= 2
    kept = {k: v for k, v in tree(directory).items() if "/moves/" in k}
    campaign.run(directory, workers=2, deadline=time.time() + 5, log=lambda *_: None)
    after = tree(directory)
    assert all(after[k] == v for k, v in kept.items())  # committed moves survive and are resumed from


def test_choose_round_follows_the_deadline_rule_and_is_final(tmp_path, saved):
    first, _ = make(tmp_path, saved, "a", games=2, opening=FIFTY)
    second, _ = make(tmp_path, saved, "b", games=2, opening=FIFTY)
    kwargs = dict(first=first, second=second, first_buffer=tmp_path / "buf-a", both_buffer=tmp_path / "buf-ab")
    with pytest.raises(ValueError, match="first half"):
        campaign.choose_round(tmp_path / "c0.json", deadline=time.time() + 3600, **kwargs)
    campaign.run(first, workers=2, log=lambda *_: None)
    future, past = time.time() + 3600, time.time() - 3600
    # Second half unfinished: open before the deadline, first half alone after it.
    with pytest.raises(ValueError, match="still finish"):
        campaign.choose_round(tmp_path / "c1.json", deadline=future, **kwargs)
    alone = campaign.choose_round(tmp_path / "c2.json", deadline=past, **kwargs)
    assert alone["halves"] == [str(first)] and alone["buffer"] == str(tmp_path / "buf-a") and alone["games"] == 2
    campaign.run(second, workers=2, log=lambda *_: None)
    both = campaign.choose_round(tmp_path / "c3.json", deadline=time.time() + 60, **kwargs)
    assert both["halves"] == [str(first), str(second)] and both["games"] == 4
    # Finished only after the deadline: still the first half alone.
    late = campaign.choose_round(tmp_path / "c4.json", deadline=past, **kwargs)
    assert late["halves"] == [str(first)]
    # A recorded choice is final, even when the halves would now qualify.
    assert campaign.choose_round(tmp_path / "c2.json", deadline=future, **kwargs) == alone
    assert campaign.status(first)["colour_balance"]["decisive_games"] == 0  # fifty-move draws
