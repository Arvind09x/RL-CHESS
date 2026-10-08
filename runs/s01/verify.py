"""Save S01 checks in a fresh directory; never overwrite previous evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = "4765a8cb3d6ae9cf50619be283176ab9ac17cca6"
SOURCE = Path("/Users/arvindsrinivass/Desktop/RL-Chess")


def main():
    out = ROOT / "runs/s01" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir()  # exclusive, even on a same-time repeat
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(name, argv):
        started = datetime.now(timezone.utc).isoformat()
        begin = time.monotonic()
        result = subprocess.run([str(arg) for arg in argv], cwd=ROOT, env=env, capture_output=True, text=True)
        (out / f"{name}.stdout.txt").write_text(result.stdout)
        (out / f"{name}.stderr.txt").write_text(result.stderr)
        record = dict(name=name, argv=[str(arg) for arg in argv], cwd=str(ROOT),
                      environment={key: env[key] for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
                      started_utc=started, elapsed_seconds=time.monotonic() - begin,
                      exit_code=result.returncode, stdout=f"{name}.stdout.txt", stderr=f"{name}.stderr.txt")
        commands.append(record)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    replay_results = []
    for label, venv in [("adapter", ".venv-acceptance"), ("model", ".venv")]:
        python = SOURCE / venv / "bin/python"
        version_code = (
            "import importlib.metadata as m, importlib.util as u, json, platform, sys; "
            "print(json.dumps(dict(python=sys.version, executable=sys.executable, "
            "platform=platform.platform(), machine=platform.machine(), "
            "versions={name:m.version(name) for name in ['numpy','python-chess','chess','pytest']}, "
            "torch_available=u.find_spec('torch') is not None), indent=2))"
        )
        run(f"environment-{label}", [python, "-c", version_code])
        run(f"focused-{label}", [python, "-m", "pytest", "tests/test_evaluator.py",
                                "-p", "no:cacheprovider", f"--junitxml={out}/focused-{label}.xml"])
        if label == "adapter":
            run("adapter-only-regressions", [python, "-m", "pytest", "-p", "no:cacheprovider",
                                           f"--junitxml={out}/adapter-only-regressions.xml"])
        replay_results.append(json.loads(run(f"replay-{label}", [python, "scripts/check_adapter.py"])))
    assert replay_results[0] == replay_results[1]
    assert len(replay_results[0]) == 13
    run("diff-check", ["git", "diff", "--check"])
    tracked = run("baseline-file-list", ["git", "ls-tree", "-r", "--name-only", BASE]).splitlines()
    preserved = {}
    for relative in tracked:
        original = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        assert current == original, f"Protected baseline file changed: {relative}"
        preserved[relative] = hashlib.sha256(current).hexdigest()
    (out / "preservation.json").write_text(json.dumps(dict(base_commit=BASE, protected_files_sha256=preserved), indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(
        base_commit=BASE, branch="codex/s01-dummy-evaluator", worktree=str(ROOT),
        checks="All recorded commands exited zero; both 13-record replays agree",
        protected_baseline_files=len(preserved), synthetic_wdl=[0.25, 0.5, 0.25],
        limitations=["CPU macOS arm64 and Python 3.12.12 only", "Synthetic plumbing outputs; no strength claim",
                     "No model inference, search, data, engine, training or paid compute", "S01 independent review pending"],
    ), indent=2) + "\n")
    print(f"Evidence: {out}", flush=True)


if __name__ == "__main__":
    main()
