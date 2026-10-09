"""Verify evidence bytes, immutable base tree and retained external artifacts."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/t02"
BASE = "b26cf38f04a09fd04283023795c0752753ce3106"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
EXTERNAL = ROOT.parent / "external-artifacts/t02-cpu-continuation"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def identity(path): return dict(bytes=path.stat().st_size, sha256=sha(path))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def decoded(path):
    data = json.loads(path.read_text())
    raw = base64.b64decode(data["data"], validate=True)
    assert data["encoding"] == "base64" and len(raw) == data["bytes"]
    assert hashlib.sha256(raw).hexdigest() == data["sha256"]
    return raw


def main():
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    assert git("branch", "--show-current").decode().strip() == "codex/t02-checkpoints"
    original = {}
    for line in git("ls-tree", "-r", BASE).decode().splitlines():
        meta, name = line.split("\t")
        original[name] = meta.split()[2]
    names = list(original)
    actual = subprocess.check_output(["git", "hash-object", "--stdin-paths"], cwd=ROOT,
        input=("\n".join(names) + "\n").encode()).decode().splitlines()
    preserved = {}
    for name, digest in zip(names, actual):
        if name in ("README.md", "coding-plan.md"):
            base = git("show", BASE + ":" + name)
            assert (ROOT / name).read_bytes().startswith(base)
            preserved[name] = dict(base_blob=original[name], exact_original_prefix=True, sha256=sha(ROOT / name))
        else:
            assert digest == original[name], name
            preserved[name] = dict(base_blob=digest, unchanged=True)
    integration = json.loads((ROOT / "runs/d08-t01-integration/main-integration.json").read_text())
    assert git("rev-parse", "HEAD", cwd=PRIMARY).decode().strip() == BASE
    assert sha(PRIMARY / "coding-plan.md") == integration["restored_primary_plan_sha256"]
    status = git("status", "--porcelain", cwd=PRIMARY).decode()
    assert status == " M coding-plan.md\n"
    write("preservation.json", dict(base_commit=BASE, original_files=len(names), files=preserved,
        primary_head=BASE, primary_status=status, primary_plan_sha256=sha(PRIMARY / "coding-plan.md"),
        primary_plan_matches_accepted_integration_record=True))
    records = []
    for path in sorted(OUT.glob("*.command.json")):
        record = json.loads(path.read_text())
        assert record["exit_code"] == 0 and record["tested_source_unchanged_during_command"]
        for name, digest in record["tested_source_sha256"].items(): assert sha(ROOT / name) == digest, name
        for output in record["outputs"].values():
            container = OUT / output["path"]
            assert sha(container) == output["container_sha256"]
            raw = decoded(container)
            assert len(raw) == output["decoded_bytes"] and hashlib.sha256(raw).hexdigest() == output["decoded_sha256"]
        records.append(record)
    assert len(records) == 8
    write("commands.json", records)
    tests = {}
    for path in sorted(OUT.glob("*.xml")):
        suite = ET.parse(path).getroot().find("testsuite")
        assert int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == 0
        tests[path.stem] = dict(collected=int(suite.attrib["tests"]), skipped=int(suite.attrib["skipped"]),
            passed=int(suite.attrib["tests"]) - int(suite.attrib["skipped"]), sha256=sha(path))
    assert tests["model-regressions"]["passed"] == 1344
    assert tests["adapter-regressions"]["passed"] == 1023 and tests["adapter-regressions"]["skipped"] == 9
    assert tests["checkpoint-tests"]["passed"] == 60
    for kind in ("model", "adapter"): assert tests["accepted-review-probes-" + kind]["passed"] == 84
    frozen = decoded(OUT / "frozen-adapter.stdout.json")
    assert frozen == decoded(ROOT / "runs/d08-t01-integration/frozen-adapter-adapter.stdout.json")
    m06 = json.loads(decoded(OUT / "m06-numerical.stdout.json"))
    assert m06["status"] == "passed automated acceptance; stop for independent review"
    real = json.loads((EXTERNAL / "summary.json").read_text())
    assert real["exact_match"] and real["separate_python_process"] and real["tolerance"] == 0
    assert real["checkpoint"] == identity(EXTERNAL / "step-1.pt")
    for name, expected in real["outputs"].items(): assert identity(EXTERNAL / name) == expected
    assert (EXTERNAL / "uninterrupted.json").read_bytes() == (EXTERNAL / "resumed.json").read_bytes()
    for name in ("summary.json", "warmup.json"):
        (OUT / ("real-" + name)).write_bytes((EXTERNAL / name).read_bytes())
    warmup = json.loads((EXTERNAL / "warmup.json").read_text())
    for name, expected in warmup["provenance"]["files"].items():
        path = Path(name) if name.startswith("/") else ROOT / name
        assert identity(path) == expected
    artifacts = {str(path): identity(path) for path in sorted(EXTERNAL.iterdir())}
    write("external-artifacts.json", dict(directory=str(EXTERNAL), files=artifacts,
        source_inputs_rehashed_unchanged=warmup["provenance"]["files"]))
    setup = dict(base_commit=BASE, primary_checkout=str(PRIMARY), isolated_checkout=str(ROOT),
        branch="codex/t02-checkpoints", strategy="isolated clone; committed main state only",
        commands=[["git", "clone", "--no-hardlinks", str(PRIMARY), "t02-checkpoints"],
                  ["git", "switch", "-c", "codex/t02-checkpoints", BASE]],
        evidence_finalization_command=[str(PRIMARY / ".venv-acceptance/bin/python"), "-B", "runs/t02/finalize.py"],
        results=["clone succeeded", "branch created at reviewed integrated main"],
        primary_user_edit_hash_from_accepted_integration=integration["restored_primary_plan_sha256"],
        agents_md="No applicable AGENTS.md found in checkout or inspected ancestor paths",
        no_subagents=True, independent_review="pending")
    write("setup.json", setup)
    summary = dict(task="T02", checked_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE,
        branch=setup["branch"], isolated_checkout=str(ROOT), tests=tests,
        real_continuation=real, m06_pass=True, frozen_adapter_exact=True,
        command_records=len(records), command_index="commands.json", preservation="preservation.json",
        external_artifact_index="external-artifacts.json", existing_files_preserved=len(names),
        primary_main_and_uncommitted_plan_preserved=True, no_subagents=True, independent_review="pending",
        scope="T02 only; no T03/T04/overfit/pilot/P/self-play/RL",
        limitations="Pinned macOS arm64 CPU float32 continuation only; float64/nondefault configurations receive unit coverage. Standard Adam and explicit single-process D07 order only. No scheduler/gradient accumulation/custom RNG/worker-prefetch/accelerator restoration; incompatible runtime rejected.")
    write("summary.json", summary)
    print(json.dumps(dict(tests=tests, exact_real_continuation=True, preserved_files=len(names),
                         command_records=len(records), primary_preserved=True), indent=2))


if __name__ == "__main__": main()
