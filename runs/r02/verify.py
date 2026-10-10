"""Record R02 checks and compare all accepted bytes without changing them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/r02"
BASE = "48bce1ecb469d07d9d5a57aff7d9cb4896a7e281"
IMPLEMENTATION = "2e31236c33a114490c59be837459803c7b4f7299"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
R01 = ROOT.parent / "r01-self-play"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def save(name, value):
    with (OUTPUT / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def preservation(name):
    baseline = json.loads((ROOT / "runs/r01/preservation-before.json").read_text())
    external = {path: dict(bytes=Path(path).stat().st_size, sha256=sha(path))
                for path in baseline["external_files"]}
    require(external == baseline["external_files"], "historical external bytes changed")
    require(git(PRIMARY, "rev-parse", "HEAD").decode().strip() == BASE, "primary base changed")
    require(git(R01, "rev-parse", "HEAD").decode().strip() == IMPLEMENTATION, "original R01 moved")
    require(not git(R01, "status", "--short"), "original R01 dirty")
    tracked = {}
    for line in git(R01, "ls-tree", "-r", IMPLEMENTATION).decode().splitlines():
        info, relative = line.split("\t")
        blob = info.split()[2]
        actual = hashlib.sha1(b"blob " + str((R01 / relative).stat().st_size).encode() + b"\0"
                              + (R01 / relative).read_bytes()).hexdigest()
        require(blob == actual, "original R01 tracked bytes changed: " + relative)
        tracked[relative] = sha(R01 / relative)
    primary_files = {}
    for line in git(PRIMARY, "ls-tree", "-r", BASE).decode().splitlines():
        info, relative = line.split("\t")
        content = (ROOT / relative).read_bytes()
        if relative in ("README.md", "coding-plan.md"):
            require(content.startswith(git(PRIMARY, "show", BASE + ":" + relative)), "base documentation edited")
        else:
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            require(actual == info.split()[2], "accepted main bytes changed: " + relative)
        primary_files[relative] = sha(PRIMARY / relative)
    primary = dict(head=BASE, branch=git(PRIMARY, "branch", "--show-current").decode().strip(),
        status=git(PRIMARY, "status", "--short").decode(), tracked_files=primary_files,
        patch_sha256=hashlib.sha256(git(PRIMARY, "diff", "--binary")).hexdigest(),
        index_sha256=sha(PRIMARY / ".git/index"))
    integration = json.loads((ROOT / "runs/r01-integration/main-integration.json").read_text())
    require(sha(PRIMARY / "coding-plan.md") == integration["restored_plan"]["sha256"],
            "accepted original user planning edits changed")
    if name == "preservation-after.json":
        prior = json.loads((OUTPUT / "preservation-before.json").read_text())
        require(primary == prior["primary"], "primary changed during R02")
    save(name, dict(base=BASE, r01_tracked_files=tracked, external_files=external,
        primary=primary, all_preserved=True))


def run(name, command):
    start = time.monotonic()
    env = dict(os.environ, PYTHONPATH="src:.", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    save(name + ".command.json", dict(argv=command, cwd=str(ROOT),
        environment=dict(PYTHONPATH=env["PYTHONPATH"], PYTHONDONTWRITEBYTECODE="1"),
        exit_code=result.returncode, duration_seconds=time.monotonic() - start,
        sources={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))
                 + [ROOT / "tests/test_replay.py", ROOT / "scripts/run_r02.py"]}))
    (OUTPUT / (name + ".stdout.txt")).write_text(result.stdout)
    (OUTPUT / (name + ".stderr.txt")).write_text(result.stderr)
    print(result.stdout[-3000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}: {result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("model", "adapter", "model-final", "before", "after"))
    args = parser.parse_args()
    if args.mode in ("before", "after"):
        preservation("preservation-" + args.mode + ".json")
        print("All accepted tracked/external bytes and primary user patch preserved")
    else:
        python = PRIMARY / (".venv/bin/python" if args.mode.startswith("model") else ".venv-acceptance/bin/python")
        command = [str(python), "-B", "-m", "pytest", "--import-mode=importlib", "-p", "no:cacheprovider",
                   "--junitxml=runs/r02/regression-" + args.mode + ".xml"]
        if args.mode.startswith("model"): command += ["-p", "runs.m06.diagnostic_plugin"]
        run("regression-" + args.mode, command)


if __name__ == "__main__":
    main()
