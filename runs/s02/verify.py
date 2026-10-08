"""Run S02 checks once and save actual evidence in a fresh directory."""
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
BASE = "0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f"


def main():
    out = ROOT / "runs/s02" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir()
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(name, argv):
        begin = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run([str(arg) for arg in argv], cwd=ROOT, env=env, capture_output=True, text=True)
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

    counts = {}
    replays = []
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
        for name, tests in [(f"focused-{label}", ["tests/test_puct.py"]), (f"regressions-{label}", [])]:
            xml = out / f"{name}.xml"
            plugin = ["-p", "runs.m06.diagnostic_plugin"] if label == "model" and not tests else []
            run(name, [python, "-m", "pytest", *tests, *plugin, "-p", "no:cacheprovider", f"--junitxml={xml}"])
            suite = ET.parse(xml).getroot().find("testsuite")
            counts[name] = dict(suite.attrib)
        replays.append(json.loads(run(f"replay-{label}", [python, "scripts/check_adapter.py"])))
    assert replays[0] == replays[1] and len(replays[0]) == 13
    run("diff-check", ["git", "diff", "--check"])
    tracked = run("baseline-files", ["git", "ls-tree", "-r", "--name-only", BASE]).splitlines()
    preserved = {}
    for relative in tracked:
        original = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        if relative == "docs/s01-evaluator.md":
            assert current.startswith(original), "S01 documentation must retain original bytes as a prefix"
            mode = "original bytes retained as prefix; review acceptance appended"
        else:
            assert current == original, f"Protected baseline changed: {relative}"
            mode = "byte-identical"
        preserved[relative] = dict(mode=mode, base_sha256=hashlib.sha256(original).hexdigest(),
                                   current_sha256=hashlib.sha256(current).hexdigest())
    (out / "preservation.json").write_text(json.dumps(dict(base_commit=BASE, files=preserved), indent=2) + "\n")
    run("branch", ["git", "status", "--short", "--branch"])
    run("base-head", ["git", "rev-parse", "HEAD"])
    (out / "summary.json").write_text(json.dumps(dict(
        base_commit=BASE, branch="codex/s02-puct-tree", worktree=str(ROOT), checks=counts,
        replay_records=13, replay_reports_equal=True,
        byte_identical_baseline_files=len(preserved) - 1, appended_document="docs/s01-evaluator.md",
        conventions=dict(formula="Q + c_puct * P * sqrt(max(1, sum outgoing N)) / (1 + N)",
                         c_puct=1.5, unvisited_value=0.0, tie="smallest shared move ID; exact float equality",
                         visits="completed backups only; root preparation and S02 traversal never increment N/W",
                         edge_value_perspective="parent player", leaf_value_perspective="leaf side to move"),
        limitations=["CPU macOS arm64/Python 3.12 only", "Dummy evaluator; no strength/performance claim",
                     "S03 backup and S04 policies/move selection remain unimplemented",
                     "No exploration, matches, self-play, real evaluator, training, GPU or paid compute",
                     "S02 independent review pending; Gate S not accepted"],
    ), indent=2) + "\n")
    print(f"Evidence: {out}", flush=True)


if __name__ == "__main__":
    main()
