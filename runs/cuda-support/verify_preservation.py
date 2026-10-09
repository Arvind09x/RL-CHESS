"""Compare review scope with accepted Git blobs and historical artifact identities."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
BASE = "ea0f8a1"
ALLOWED = {"src/chess_rl/training/checkpoints.py", "src/chess_rl/search/real_evaluator.py"}


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def main():
    names = git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    unchanged = {}
    for name in names:
        expected = git("show", f"{BASE}:{name}")
        if name not in ALLOWED:
            assert (ROOT/name).read_bytes() == expected, name
            unchanged[name] = identity(ROOT/name)
        if name != "coding-plan.md":
            assert (PRIMARY/name).read_bytes() == expected, ("primary drift", name)
    receipt = json.loads((ROOT/"runs/t03-t04-integration/main-integration.json").read_text())
    assert identity(PRIMARY/"coding-plan.md") == receipt["restored_plan_identity"]
    assert git("rev-parse", "HEAD", cwd=PRIMARY) == git("rev-parse", BASE)
    assert git("branch", "--show-current", cwd=PRIMARY).decode().strip() == "main"
    assert git("diff", "--cached", cwd=PRIMARY) == b""
    assert git("status", "--short", cwd=PRIMARY).decode() == " M coding-plan.md\n"
    expected_external = json.loads((ROOT/"runs/t03-t04-integration/preservation-final.json").read_text())["external_files"]
    for path, expected in expected_external.items():
        assert identity(Path(path)) == expected, path
    staging = ROOT.parent/"external-artifacts/cuda-support-portable"
    report = json.loads((staging/"staging-evidence.json").read_text())
    for relative, entry in report["files"].items():
        expected = {k:entry[k] for k in ("bytes", "sha256")}
        assert identity(Path(entry["source"])) == identity(staging/relative) == expected
    plan = json.loads((ROOT/"runs/t03/expected-reload.json").read_text())
    assert identity(staging/"checkpoint.pt") == plan["checkpoint_identity"]
    result = dict(base_commit=git("rev-parse", BASE).decode().strip(),
        audited_utc=datetime.now(timezone.utc).isoformat(), allowed_existing_edits=sorted(ALLOWED),
        all_other_accepted_blobs_unchanged=unchanged, historical_external_artifacts=expected_external,
        primary_head=git("rev-parse", "HEAD", cwd=PRIMARY).decode().strip(),
        primary_original_plan=identity(PRIMARY/"coding-plan.md"),
        primary_uncommitted_patch_base64=base64.b64encode(git("diff", cwd=PRIMARY)).decode(),
        self_play_and_game_modules_unchanged=True, shared_cpu_lock_unchanged=True,
        original_frozen_bytes_and_staged_copies_identical=True)
    (ROOT/"runs/cuda-support/preservation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Accepted tracked files, main/user patch, historical external artifacts and portable bytes preserved")


if __name__ == "__main__": main()
