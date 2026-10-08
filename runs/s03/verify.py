"""Save actual S03 checks in both existing environments; never alter installs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
BASE = "1301d033b81ca88e51c66ba27c1035a80c50fe54"
CHANGED_BASE_FILES = {"src/chess_rl/search/puct.py"}


def main():
    out = ROOT / "runs/s03" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir()
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(name, argv):
        started = datetime.now(timezone.utc).isoformat()
        begin = time.monotonic()
        result = subprocess.run([str(arg) for arg in argv], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        (out / f"{name}.stdout.txt").write_text(result.stdout)
        (out / f"{name}.stderr.txt").write_text(result.stderr)
        commands.append(dict(name=name, argv=[str(arg) for arg in argv], cwd=str(ROOT),
                             environment={key: env[key] for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
                             started_utc=started, elapsed_seconds=time.monotonic() - begin,
                             exit_code=result.returncode, stdout=f"{name}.stdout.txt", stderr=f"{name}.stderr.txt"))
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    counts, replays = {}, []
    for label, venv in [("adapter", ".venv-acceptance"), ("model", ".venv")]:
        python = SOURCE / venv / "bin/python"
        code = (
            "import importlib.metadata as m, importlib.util as u, json, platform, sys; "
            "print(json.dumps(dict(python=sys.version, executable=sys.executable, "
            "platform=platform.platform(), machine=platform.machine(), "
            "versions={name:m.version(name) for name in ['numpy','python-chess','chess','pytest']}, "
            "torch_available=u.find_spec('torch') is not None), indent=2))"
        )
        run(f"environment-{label}", [python, "-c", code])
        for name, tests in [
            (f"focused-{label}", ["tests/test_backup.py", "tests/test_puct.py", "tests/test_evaluator.py"]),
            (f"regressions-{label}", []),
            (f"accepted-review-probes-{label}", ["runs/d02-s02-review/test_review_cases.py"]),
        ]:
            xml = out / f"{name}.xml"
            plugin = ["-p", "runs.m06.diagnostic_plugin"] if label == "model" and not tests else []
            run(name, [python, "-m", "pytest", *tests, *plugin, "-p", "no:cacheprovider", f"--junitxml={xml}"])
            counts[name] = dict(ET.parse(xml).getroot().find("testsuite").attrib)
        replays.append(json.loads(run(f"replay-{label}", [python, "scripts/check_adapter.py"])))
    assert replays[0] == replays[1] and len(replays[0]) == 13
    run("diff-check", ["git", "diff", "--check"])
    tracked = run("baseline-files", ["git", "ls-tree", "-r", "--name-only", BASE]).splitlines()
    preserved = {}
    for relative in tracked:
        if relative in CHANGED_BASE_FILES:
            continue
        original = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        assert current == original, f"Protected accepted-base file changed: {relative}"
        preserved[relative] = hashlib.sha256(current).hexdigest()
    (out / "preservation.json").write_text(json.dumps(dict(
        base_commit=BASE, scoped_modified_base_files=sorted(CHANGED_BASE_FILES),
        byte_identical_files=preserved), indent=2) + "\n")
    run("branch", ["git", "status", "--short", "--branch"])
    run("base-head", ["git", "rev-parse", "HEAD"])
    (out / "summary.json").write_text(json.dumps(dict(
        base_commit=BASE, branch="codex/s03-value-backup", worktree=str(ROOT), checks=counts,
        replay_records_per_environment=13, replay_reports_equal=True,
        byte_identical_accepted_base_files=len(preserved),
        conventions=dict(formula="Q + c_puct * P * sqrt(max(1, sum outgoing N)) / (1 + N)",
                         c_puct=1.5, unvisited_value=0.0, tie="smallest shared move ID; exact float equality",
                         api="backup(resolved) -> int: 1 for completed nonempty path; 0 for terminal root",
                         completion="at most once per traverse-issued ticket; duplicate raises ValueError",
                         edge_value_perspective="parent player", leaf_value_perspective="leaf side to move"),
        limitations=["CPU macOS arm64/Python 3.12 only; sequential search-owned records",
                     "Dummy/controlled evaluators; no strength or performance measurement",
                     "No S04 policy/move selection, exploration, matches, self-play, real evaluator or training",
                     "S03 independent review pending; Gate S remains unaccepted"],
    ), indent=2) + "\n")
    print(f"Evidence: {out}", flush=True)


if __name__ == "__main__":
    main()
