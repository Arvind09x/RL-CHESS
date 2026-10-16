"""Round extension: prefix choice under a deadline, training-buffer pointer with exact membership, disk."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")

from chess_rl import selfplay_campaign as campaign
from chess_rl import selfplay_rounds as rounds
from chess_rl.replay import ReplayBuffer
from .test_checkpoints import saved  # noqa: F401
from .test_selfplay_campaign import FIFTY, SMALL

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")


def half(tmp_path, saved, name, seed, *, games=2, play=True):
    directory = tmp_path / name
    campaign.create(directory, saved["path"], games=games, seed=seed, settings=SMALL, opening=FIFTY, purpose="half")
    if play:
        campaign.run(directory, workers=2, log=lambda *_: None)
    return directory


def ingest(buffer, *halves):
    for directory in halves:
        campaign.archive(directory)
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", 'source jobs/b200-selfplay/finish.sh; ingest_archives "$@"',
                             "_", str(buffer), *map(str, halves)], cwd=ROOT, env=dict(ENV, PY=sys.executable),
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


def core_round(tmp_path, saved, *, both=True):
    a, b = half(tmp_path, saved, "a", 21), half(tmp_path, saved, "b", 22, play=both)
    deadline = time.time() + (60 if both else -1)
    choice = campaign.choose_round(tmp_path / "core.json", first=a, second=b, deadline=deadline,
                                   first_buffer=tmp_path / "buf-a", both_buffer=tmp_path / "buf-ab")
    ingest(Path(choice["buffer"]), *map(Path, choice["halves"]))
    return tmp_path / "core.json", a, b


def test_choose_prefix_counts_only_a_finished_prefix_and_is_final(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c, d = half(tmp_path, saved, "c", 23, play=False), half(tmp_path, saved, "d", 24, play=False)
    kwargs = dict(core=core, halves=[c, d], buffers=[tmp_path / "abc", tmp_path / "abcd"])
    future, past = time.time() + 3600, time.time() - 3600
    with pytest.raises(ValueError, match="still finish"):
        rounds.choose_prefix(tmp_path / "x0.json", deadline=future, **kwargs)
    none = rounds.choose_prefix(tmp_path / "x1.json", deadline=past, **kwargs)
    assert none["counted"] == [] and none["buffer"] is None and none["halves"] == [str(a), str(b)]
    campaign.run(c, workers=2, log=lambda *_: None)
    deadline = time.time() + 30
    with pytest.raises(ValueError, match=f"{d}.*still finish"):
        rounds.choose_prefix(tmp_path / "x2.json", deadline=deadline, **kwargs)
    only_c = rounds.choose_prefix(tmp_path / "x3.json", deadline=deadline, now=deadline + 1, **kwargs)
    assert only_c["counted"] == [str(c)] and only_c["buffer"] == str(tmp_path / "abc") and only_c["games"] == 6
    campaign.run(d, workers=2, log=lambda *_: None)
    both = rounds.choose_prefix(tmp_path / "x4.json", deadline=time.time() + 30, **kwargs)
    assert both["counted"] == [str(c), str(d)] and both["buffer"] == str(tmp_path / "abcd") and both["games"] == 8
    # d finished in time but c late: neither counts, and d never counts without c.
    deadline = time.time()
    for game in campaign.load(c)["games"]:
        os.utime(c / game["directory"] / "boundary.json", (deadline + 100, deadline + 100))
    late = rounds.choose_prefix(tmp_path / "x5.json", deadline=deadline, now=deadline + 200, **kwargs)
    assert late["counted"] == [] and late["buffer"] is None
    # Final once written; skipped halves recorded; the core record is untouched.
    before = (tmp_path / "core.json").read_bytes()
    assert rounds.choose_prefix(tmp_path / "x1.json", deadline=future,
                                **kwargs) == none
    skipped = rounds.choose_prefix(tmp_path / "x6.json", core=core, halves=[c], buffers=[tmp_path / "abc"],
                                   deadline=past, skipped=[(d, "disk")])
    assert skipped["skipped"] == [dict(half=str(d), reason="disk")] and skipped["considered"] == [str(c)]
    assert (tmp_path / "core.json").read_bytes() == before


def test_choose_prefix_refuses_a_core_of_one_half(tmp_path, saved):
    core, _, _ = core_round(tmp_path, saved, both=False)
    c = half(tmp_path, saved, "c", 23)
    with pytest.raises(ValueError, match="needs both"):
        rounds.choose_prefix(tmp_path / "x.json", core=core, halves=[c], buffers=[tmp_path / "abc"],
                             deadline=time.time() + 60)
    pointer = rounds.write_training_pointer(tmp_path / "p.json", core=core)
    assert pointer["branch"] == "core-a-only" and pointer["buffer"] == str(tmp_path / "buf-a")
    assert pointer["games"] == 2 and pointer["training_positions_before_holdout"] == 2  # one ply per game


def test_pointer_takes_the_extended_set_only_when_it_loaded_exactly(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c = half(tmp_path, saved, "c", 23)
    deadline = time.time() + 60
    ext = rounds.choose_prefix(tmp_path / "ext.json", core=core, halves=[c], buffers=[tmp_path / "abc"],
                               deadline=deadline)
    # Not loaded yet: falls back to the core with the reason.
    missing = rounds.write_training_pointer(tmp_path / "p0.json", core=core, extension=tmp_path / "ext.json")
    assert missing["branch"] == "core-extension-unclean" and "no ingest report" in missing["reason"]
    assert missing["buffer"] == str(tmp_path / "buf-ab") and missing["games"] == 4
    ingest(tmp_path / "abc", a, b, c)
    pointer = rounds.write_training_pointer(tmp_path / "p1.json", core=core, extension=tmp_path / "ext.json")
    assert pointer["branch"] == "extended" and pointer["buffer"] == ext["buffer"] and pointer["games"] == 6
    assert pointer["halves"] == [str(a), str(b), str(c)]
    assert pointer["training_positions_before_holdout"] == ReplayBuffer(
        tmp_path / "abc" / "replay.sqlite").summary()["training_positions"]
    assert rounds.write_training_pointer(tmp_path / "p1.json", core=core) == pointer  # final
    # An extra game in the buffer, or a report from another archive, is unclean.
    stray = half(tmp_path, saved, "stray", 99, games=1)
    ingest(tmp_path / "abc", stray)
    extra = rounds.write_training_pointer(tmp_path / "p2.json", core=core, extension=tmp_path / "ext.json")
    assert extra["branch"] == "core-extension-unclean" and "extra 1" in extra["reason"]
    ingest(tmp_path / "abc2", a, b, c)
    report = json.loads((tmp_path / "abc2" / "ingest-c.json").read_text())
    report["receipts"] = report["receipts"][:-1]
    (tmp_path / "abc2" / "ingest-c.json").rename(tmp_path / "abc2" / ".old")
    (tmp_path / "abc2" / "ingest-c.json").write_text(json.dumps(report))
    rounds.choose_prefix(tmp_path / "ext2.json", core=core, halves=[c], buffers=[tmp_path / "abc2"],
                         deadline=deadline)
    short = rounds.write_training_pointer(tmp_path / "p3.json", core=core, extension=tmp_path / "ext2.json")
    assert short["branch"] == "core-extension-unclean" and "receipts" in short["reason"]


def test_pointer_branches_without_an_extended_set_and_an_unclean_core(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c = half(tmp_path, saved, "c", 23, play=False)
    rounds.choose_prefix(tmp_path / "ext.json", core=core, halves=[c], buffers=[tmp_path / "abc"],
                         deadline=time.time() - 1)
    late = rounds.write_training_pointer(tmp_path / "p0.json", core=core, extension=tmp_path / "ext.json")
    assert late["branch"] == "core-extension-not-counted" and late["buffer"] == str(tmp_path / "buf-ab")
    skipped = rounds.write_training_pointer(tmp_path / "p1.json", core=core, note="disk")
    assert skipped["branch"] == "core-extension-not-run" and skipped["reason"] == "disk"
    # A core buffer that does not verify writes no pointer at all.
    (tmp_path / "buf-ab" / "ingest-b.json").rename(tmp_path / "buf-ab" / ".moved")
    with pytest.raises(rounds.UncleanBuffer, match="no ingest report"):
        rounds.write_training_pointer(tmp_path / "p2.json", core=core, note="disk")
    assert not (tmp_path / "p2.json").exists()


def test_cli_finished_choose_and_pointer(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c = half(tmp_path, saved, "c", 23)

    def cli(*args):
        return subprocess.run([sys.executable, "scripts/extend_selfplay_round.py", *map(str, args)], cwd=ROOT,
                              env=ENV, capture_output=True, text=True, timeout=300)
    assert cli("finished", "--directory", c, "--deadline", "2099-01-01T00:00:00+05:30").returncode == 0
    assert cli("finished", "--directory", c, "--deadline", "2000-01-01T00:00:00+05:30").returncode == 1
    result = cli("choose", "--output", tmp_path / "ext.json", "--core", core, "--half", c, "--buffer",
                 tmp_path / "abc", "--deadline", "2099-01-01T00:00:00+05:30")
    assert result.returncode == 0, result.stderr
    ingest(tmp_path / "abc", a, b, c)
    result = cli("pointer", "--output", tmp_path / "p.json", "--core", core, "--extension", tmp_path / "ext.json")
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "p.json").read_text())["branch"] == "extended"


def test_disk_projection_nets_out_existing_output(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c = half(tmp_path, saved, "c", 23, play=False)
    kwargs = dict(core_halves=[a, b], core_buffer=tmp_path / "buf-ab", buffer=tmp_path / "abc", buffer_halves=3)
    fresh = rounds.project_need(play=[c], **kwargs)
    assert fresh["per_half_store_bytes"] > 0 and fresh["per_half_buffer_bytes"] > 0
    existing = rounds._half_store(c)  # the created but unplayed half already holds its games/ directory
    assert fresh["raw_need_bytes"] == pytest.approx(
        fresh["per_half_store_bytes"] - existing + 3 * fresh["per_half_buffer_bytes"])
    assert fresh["need_bytes"] == pytest.approx(fresh["raw_need_bytes"] * 1.3 + (1 << 30))
    campaign.run(c, workers=2, log=lambda *_: None)
    ingest(tmp_path / "abc", a, b, c)
    done = rounds.project_need(play=[c], **kwargs)
    assert done["raw_need_bytes"] < fresh["raw_need_bytes"]


def test_quota_parsing_and_du_fallback(tmp_path):
    text = ("Disk quotas for user arvind (uid 1001):\n"
            "     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace\n"
            "      /dev/sda1 58000000* 97656250 100000000          12345       0       0\n")
    assert rounds.parse_quota(text) == (58000000 * 1024, 97656250 * 1024)
    assert rounds.parse_quota("") is None
    assert rounds.parse_quota(text + "      /dev/sdb1 1 2 3 4 5 6\n") is None  # two filesystems: ambiguous
    (tmp_path / "f").write_bytes(b"x" * 100000)
    free = rounds.free_quota(10**9, home=tmp_path)
    if free["method"].startswith("du"):
        assert free["used_bytes"] >= 100000 and free["free_bytes"] == 10**9 - free["used_bytes"]


def test_extension_job_script_parses():
    result = subprocess.run(["bash", "-n", "jobs/b200-selfplay/round1-ext.sbatch"], cwd=ROOT, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


def _sandbox(tmp_path, saved, *, quota_bytes, minutes=30):
    """A repository-shaped directory that runs a CPU, 2-simulation, 2-game copy of round1-ext.sbatch
    against a finished a+b core built the way round1.sbatch builds it."""
    box = tmp_path / "box"
    box.mkdir()
    for name in ("src", "jobs", "runtime", "pyproject.toml", ".git"):
        (box / name).symlink_to(ROOT / name)
    # Halves the job creates start from the fifty-move position, so every game is one ply.
    (box / "scripts").mkdir()
    for script in (ROOT / "scripts").glob("*.py"):
        if script.name != "run_selfplay_campaign.py":
            (box / "scripts" / script.name).symlink_to(script)
    # The real script's text runs under the sandbox path, so its default heavy lock is the sandbox's.
    real = (ROOT / "scripts/run_selfplay_campaign.py").read_text()
    (box / "scripts" / "run_selfplay_campaign.py").write_text(
        "from chess_rl import selfplay_campaign as campaign\n"
        "from tests.test_selfplay_campaign import FIFTY\n"
        "campaign.create.__kwdefaults__['opening'] = FIFTY\n" + real)
    (box / "tests").symlink_to(ROOT / "tests")
    (box / ".venv-cuda-cu128" / "bin").mkdir(parents=True)
    # The B200 venv has the package installed; common.sh puts only the root on PYTHONPATH.
    wrapper = box / ".venv-cuda-cu128" / "bin" / "python"
    wrapper.write_text(f'#!/bin/sh\nPYTHONPATH="{box / "src"}:{box}${{PYTHONPATH:+:$PYTHONPATH}}" '
                       f'exec {sys.executable} "$@"\n')
    wrapper.chmod(0o755)
    (box / "artifacts" / "p-1m").mkdir(parents=True)
    (box / "artifacts" / "slurm").mkdir()
    p = box / "artifacts" / "p-1m" / "step-090000.pt"
    p.write_bytes(Path(saved["path"]).read_bytes())
    sel = box / "runs" / "selfplay"
    halves = []
    for name, seed in (("a", 20261017), ("b", 20261018)):
        directory = sel / f"round1{name}-g200-sims2"
        campaign.create(directory, p, games=2, simulations=2, seed=seed, opening=FIFTY, purpose="core half")
        campaign.run(directory, workers=2, log=lambda *_: None)
        halves.append(directory)
    campaign.choose_round(sel / "round1-sims2-choice.json", first=halves[0], second=halves[1],
                          deadline=time.time() + 60, first_buffer=sel / "round1-a-sims2",
                          both_buffer=sel / "round1-ab-sims2")
    script = (ROOT / "jobs/b200-selfplay/round1-ext.sbatch").read_text()
    deadline = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + 60 * minutes))
    for old, new in (("SIMS=128", "SIMS=2"), ("DEADLINE=2026-10-17T11:00:00+05:30", f"DEADLINE={deadline}"),
                     ("QUOTA_BYTES=100000000000", f"QUOTA_BYTES={quota_bytes}"), ("\nassert_gpu\n", "\n"),
                     ("LOAD_RETRY_SECONDS=60", "LOAD_RETRY_SECONDS=0"),
                     ("--games 200", "--games 2"), ("int(sys.argv[3]), 200", "int(sys.argv[3]), 2"),
                     ("--device cuda:0", "--device cpu")):
        assert script.count(old) == 1, old
        script = script.replace(old, new)
    (box / "ext.sbatch").write_text(script)
    return box, sel


def _job(box):
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(box), SLURM_JOB_ID="0", SLURM_JOB_PARTITION="test",
               SLURM_JOB_ACCOUNT="test", HOME=str(box), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["bash", str(box / "ext.sbatch")], cwd=box, env=env, capture_output=True, text=True,
                          timeout=900)


def test_extension_job_end_to_end_extended_and_rerun(tmp_path, saved):
    box, sel = _sandbox(tmp_path, saved, quota_bytes=10**12)
    result = _job(box)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    pointer = json.loads((sel / "round1-sims2-training-buffer.json").read_text())
    assert pointer["branch"] == "extended" and pointer["games"] == 8
    assert pointer["buffer"] == "runs/selfplay/round1-abcd-sims2"
    choice = json.loads((sel / "round1-ext-sims2-choice.json").read_text())
    assert [Path(h).name for h in choice["counted"]] == ["round1c-g200-sims2", "round1d-g200-sims2"]
    core = (sel / "round1-sims2-choice.json").read_bytes()
    assert (sel / "round1-ab-sims2" / "ingest-round1b-g200-sims2.json").exists()  # core finished by the job
    rerun = _job(box)  # resubmission: the pointer is final
    assert rerun.returncode == 0 and "already recorded" in rerun.stdout
    assert (sel / "round1-sims2-choice.json").read_bytes() == core


def test_extension_job_falls_back_to_the_core_when_disk_does_not_fit(tmp_path, saved):
    box, sel = _sandbox(tmp_path, saved, quota_bytes=1)
    result = _job(box)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    pointer = json.loads((sel / "round1-sims2-training-buffer.json").read_text())
    assert pointer["branch"] == "core-extension-not-run" and "did not pass" in pointer["reason"]
    assert pointer["buffer"].endswith("runs/selfplay/round1-ab-sims2") and pointer["games"] == 4
    assert not (sel / "round1c-g200-sims2").exists() and not (sel / "round1-ext-sims2-choice.json").exists()


def test_a_corrupt_or_malformed_extended_set_falls_back_to_the_core(tmp_path, saved):
    core, a, b = core_round(tmp_path, saved)
    c = half(tmp_path, saved, "c", 23)
    ingest(tmp_path / "abc", a, b, c)
    rounds.choose_prefix(tmp_path / "ext.json", core=core, halves=[c], buffers=[tmp_path / "abc"],
                         deadline=time.time() + 60)
    report_path = tmp_path / "abc" / "ingest-c.json"
    report = json.loads(report_path.read_text())
    report_path.rename(tmp_path / "abc" / ".kept")
    report_path.write_text(json.dumps({k: v for k, v in report.items() if k != "receipts"}))
    malformed = rounds.write_training_pointer(tmp_path / "p0.json", core=core, extension=tmp_path / "ext.json")
    assert malformed["branch"] == "core-extension-unclean" and malformed["reason"].startswith("KeyError")
    (tmp_path / "abc" / ".kept").replace(report_path)
    database = tmp_path / "abc" / "replay.sqlite"
    data = bytearray(database.read_bytes())
    data[:100] = b"\0" * 100  # destroy the SQLite header
    database.write_bytes(bytes(data))
    corrupt = rounds.write_training_pointer(tmp_path / "p1.json", core=core, extension=tmp_path / "ext.json")
    assert corrupt["branch"] == "core-extension-unclean" and corrupt["buffer"] == str(tmp_path / "buf-ab")
    assert "DatabaseError" in corrupt["reason"]


def test_an_incomplete_du_scan_is_not_trusted(tmp_path, monkeypatch):
    (tmp_path / "f").write_bytes(b"x" * 10000)
    assert rounds.allocated_bytes(tmp_path) >= 10000

    def partial(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="4096\tx\n4096\ttotal\n", stderr="du: cannot read")
    monkeypatch.setattr(rounds.subprocess, "run", partial)
    with pytest.raises(rounds.DiskUnmeasured, match="exited 1"):
        rounds.allocated_bytes(tmp_path)
    with pytest.raises(rounds.DiskUnmeasured):
        rounds.free_quota(10**9, home=tmp_path)  # quota and du both fail: no number to trust


def test_extension_job_falls_back_after_a_load_that_keeps_failing(tmp_path, saved):
    box, sel = _sandbox(tmp_path, saved, quota_bytes=10**12)
    (sel / "round1-abcd-sims2").write_text("not a directory")  # every load attempt fails at mkdir
    result = _job(box)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert result.stderr.count("loading the extended set: attempt") == 3
    pointer = json.loads((sel / "round1-sims2-training-buffer.json").read_text())
    assert pointer["branch"] == "core-extension-unclean" and "failed 3 times" in pointer["note"]
    assert pointer["buffer"].endswith("runs/selfplay/round1-ab-sims2") and pointer["games"] == 4
