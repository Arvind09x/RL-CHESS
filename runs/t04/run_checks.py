"""Record bounded T04 commands, outputs, environment and immutable sources."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/t04"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
MODEL = str(PRIMARY / ".venv/bin/python")
ADAPTER = str(PRIMARY / ".venv-acceptance/bin/python")
CHECKPOINT = ROOT.parent / "external-artifacts/t02-integration-cpu-continuation-20261009/step-1.pt"


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preservation():
    return dict(primary_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PRIMARY, text=True).strip(),
        primary_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=PRIMARY, text=True),
        primary_plan=identity(PRIMARY / "coding-plan.md"),
        checkpoint=identity(CHECKPOINT))


def write(name, value):
    with (OUT / name).open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    mode = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else mode
    overrides = dict(PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    environment = dict(os.environ, **overrides)
    if mode == "focused":
        if not (OUT / "preservation-before.json").exists():
            write("preservation-before.json", preservation())
        argv = [MODEL, "-m", "pytest", "tests/test_real_evaluator.py", "tests/test_checkpoints.py",
                "-q", "-p", "no:cacheprovider", f"--junitxml=runs/t04/{name}.xml"]
    elif mode in ("model", "adapter"):
        argv = [MODEL if mode == "model" else ADAPTER, "-m", "pytest", "tests", "-q",
                "-p", "no:cacheprovider", f"--junitxml=runs/t04/{mode}.xml"]
        if mode == "model": argv.extend(["-p", "runs.m06.diagnostic_plugin"])
    elif mode == "fixture":
        argv = [MODEL, "scripts/check_t04.py", "--checkpoint", str(CHECKPOINT),
                "--output", "runs/t04/fixture.json"]
    elif mode == "replay":
        argv = [ADAPTER, "scripts/check_adapter.py", "--fixtures", "tests/fixtures/adapter_records.json"]
    else: raise ValueError(mode)
    sources = {str(path.relative_to(ROOT)): identity(path) for directory in ("src", "tests")
               for path in sorted((ROOT / directory).rglob("*.py"))}
    sources["scripts/check_t04.py"] = identity(ROOT / "scripts/check_t04.py")
    sources["runs/t04/run_checks.py"] = identity(Path(__file__))
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    run = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, text=True)
    write(f"{name}.command.json", dict(argv=argv, cwd=str(ROOT), environment_overrides=overrides,
          started_at_utc=start, finished_at_utc=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic() - clock, returncode=run.returncode,
          executable=identity(Path(argv[0]).resolve()), sources=sources))
    with (OUT / f"{name}.stdout.txt").open("x") as handle: handle.write(run.stdout)
    with (OUT / f"{name}.stderr.txt").open("x") as handle: handle.write(run.stderr)
    after = preservation()
    write(f"preservation-after-{name}.json", after)
    assert after == json.loads((OUT / "preservation-before.json").read_text())
    print(run.stdout[-3000:])
    print(run.stderr[-1000:])
    raise SystemExit(run.returncode)


if __name__ == "__main__": main()
