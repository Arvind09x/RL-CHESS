"""Append-only R01 commands, frozen real suite and preservation receipts."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/r01"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
PYTHON = PRIMARY / ".venv/bin/python"
ADAPTER = PRIMARY / ".venv-acceptance/bin/python"
CHECKPOINT = ROOT.parent / "external-artifacts/t03-tiny-overfit-20261009/final.pt"


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest())


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def snapshot():
    prior = json.loads((ROOT / "runs/t03-t04-integration/preservation-final.json").read_text())
    external = {name: identity(name) for name in prior["external_files"]}
    assert external == prior["external_files"]
    files = git("ls-files", "-z", cwd=PRIMARY).decode().split("\0")[:-1]
    return dict(primary_head=git("rev-parse", "HEAD", cwd=PRIMARY).decode().strip(),
        primary_branch=git("branch", "--show-current", cwd=PRIMARY).decode().strip(),
        primary_status=git("status", "--porcelain=v1", "-uall", cwd=PRIMARY).decode(),
        primary_patch_base64=base64.b64encode(git("diff", "--binary", cwd=PRIMARY)).decode(),
        primary_index_base64=base64.b64encode(git("diff", "--cached", "--binary", cwd=PRIMARY)).decode(),
        primary_tracked_files={name: identity(PRIMARY / name) for name in files}, external_files=external)


def run(name, command):
    overrides = dict(PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sources = {str(p.relative_to(ROOT)): identity(p) for directory in ("src", "tests", "scripts")
               for p in sorted((ROOT / directory).rglob("*.py"))}
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    result = subprocess.run([str(c) for c in command], cwd=ROOT, env=os.environ | overrides, capture_output=True)
    elapsed = time.monotonic() - tick
    for label, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
        save(OUT / f"{name}.{label}.json", dict(encoding="base64", data=base64.b64encode(raw).decode(),
             bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    save(OUT / f"{name}.command.json", dict(argv=[str(c) for c in command], cwd=str(ROOT),
        environment_overrides=overrides, started_at_utc=started, duration_seconds=elapsed,
        exit_code=result.returncode, tested_sources=sources))
    assert all(identity(ROOT / name) == value for name, value in sources.items())
    print(f"{name}: exit={result.returncode}, {elapsed:.3f}s", flush=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
    assert result.returncode == 0, name


def main():
    mode = sys.argv[1]
    if mode == "prepare":
        save(OUT / "preservation-before.json", snapshot())
        save(OUT / "setup.json", dict(base_commit=git("rev-parse", "HEAD").decode().strip(),
            branch=git("branch", "--show-current").decode().strip(), checkout=str(ROOT),
            accepted_combined_main=True, applicable_agents="none found in repositories/ancestors",
            instructions_read=["interface.md", "coding-plan.md", "docs/s05-search-smoke.md",
                "docs/d05-s05-review.md", "docs/t02-checkpoints.md", "docs/t02-integration.md",
                "docs/t03-tiny-overfit.md", "docs/t04-real-evaluator.md", "docs/t03-t04-integration.md",
                "docs/e01-match-protocol.md", "docs/d06-d07-e01-review.md"],
            scope="Only R01; independent review pending; no integration, R02/R03/R04, P or strength claim"))
        save(OUT / "development.json", dict(initial_unrecorded_command="pytest tests/test_selfplay.py -q -p no:cacheprovider",
            initial_result=dict(passed=52, failed=3), fixture_corrections=[
                "Replace historical LONG_CYCLE repetition fixture that continues after adjudication with exact reached threefold prefix",
                "Use halfmove 99 for last-ply mate fixtures; 149 is already terminal under actual-occurrence-v1"],
            production_changes_for_these_failures=False))
    elif mode == "preserve":
        current = snapshot()
        assert current == json.loads((OUT / "preservation-before.json").read_text())
        # All base files except deliberate documentation appends stay exact.
        changed = git("diff", "--name-only", "HEAD").decode().splitlines()
        assert set(changed) <= {"README.md", "coding-plan.md"}, changed
        save(OUT / "preservation-after.json", current)
        print("Primary, all accepted tracked files and 76 indexed external files preserved")
    elif mode == "focused":
        run("focused", [PYTHON, "-B", "-m", "pytest", "tests/test_selfplay.py", "tests/test_search_runner.py",
            "tests/test_game_runner.py", "tests/test_real_evaluator.py", "tests/test_checkpoints.py",
            "-p", "no:cacheprovider", "--junitxml=runs/r01/focused.xml"])
    elif mode in ("model", "adapter"):
        python = PYTHON if mode == "model" else ADAPTER
        plugin = ["-p", "runs.m06.diagnostic_plugin"] if mode == "model" else []
        run(f"regression-{mode}", [python, "-B", "-m", "pytest", "--import-mode=importlib",
            "-p", "no:cacheprovider", *plugin, f"--junitxml=runs/r01/regression-{mode}.xml"])
        run(f"review-probes-{mode}", [python, "-B", "-m", "pytest", "--import-mode=importlib", "-p", "no:cacheprovider",
            *[f"runs/{n}/test_review_cases.py" for n in ("d02-s02-review", "d03-s03-review", "d04-s04-review", "d05-s05-review")],
            f"--junitxml=runs/r01/review-probes-{mode}.xml"])
        run(f"adapter-replay-{mode}", [python, "-B", "scripts/check_adapter.py"])
        if mode == "model":
            run("m06-numerical", [python, "-B", "scripts/check_m06.py"])
    elif mode in ("freeze", "fixtures"):
        run(mode, [PYTHON, "-B", "scripts/check_r01.py", mode, "--checkpoint", CHECKPOINT,
                   "--output", OUT / "real"])
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
