"""Verify D08 scope, accepted external bytes and unchanged primary main."""
import hashlib
import json
from pathlib import Path
import subprocess

from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[1]
BASE = "9565093abe40a7807e3472c1101f5f28c14b7afe"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def main():
    changed = git("diff", "--name-only", BASE).decode().splitlines()
    allowed = {"src/chess_rl/stockfish_worker.py", "README.md", "coding-plan.md"}
    assert set(changed) <= allowed
    assert git("branch", "--show-current").decode().strip() == "codex/d08-initial-labels"
    assert git("rev-parse", "main").decode().strip() == BASE
    assert git("rev-parse", "main", cwd=PRIMARY).decode().strip() == BASE
    primary_status = git("status", "--porcelain", cwd=PRIMARY).decode()
    assert primary_status in ("", " M coding-plan.md\n")
    primary_documentation = None
    if primary_status:
        original = git("show", BASE+":coding-plan.md", cwd=PRIMARY)
        expected = original
        for milestone in ("D06", "D07", "E01"):
            expected = expected.replace(("| [ ] | "+milestone+" / P").encode(),
                                        ("| [x] | "+milestone+" / P").encode(), 1)
        assert (PRIMARY/"coding-plan.md").read_bytes() == expected
        primary_documentation = dict(path=str(PRIMARY/"coding-plan.md"),
            identity=file_identity(PRIMARY/"coding-plan.md"),
            observed_change="Only D06/D07/E01 status checkboxes changed outside the D08 checkout during execution; preserved without writing",
            committed_identity=dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest()))
    accepted = git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    appended = {}
    for name in ("README.md", "coding-plan.md"):
        original = git("show", BASE+":"+name)
        assert (ROOT/name).read_bytes().startswith(original)
        appended[name] = dict(accepted_prefix_bytes=len(original),
                              accepted_prefix_sha256=hashlib.sha256(original).hexdigest())
    previous = json.loads((ROOT / "runs/d06-d07-e01-integration/preservation.json").read_text())
    external = previous["original_external_identities"]
    for path, identity in external.items():
        assert file_identity(Path(path)) == identity, path
    settings = json.loads(Path("/private/tmp/rl-chess-d08-production/settings.json").read_text())
    for name, identity in settings["production_source_identities"].items():
        assert file_identity(ROOT/name) == identity
    new = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    allowed_new = {"src/chess_rl/initial_labels.py", "tests/test_initial_labels.py",
                   "scripts/produce_d08.py", "scripts/audit_d08.py", "scripts/check_d08_loader.py",
                   "scripts/report_d08.py", "scripts/check_d08_preservation.py", "docs/d08-initial-labels.md"}
    assert all(p in allowed_new or p.startswith("runs/d08/") for p in new)
    report = dict(base_commit=BASE, branch="codex/d08-initial-labels", primary_main_preserved=True,
        primary_main_clean=not bool(primary_status), primary_working_status=primary_status,
        primary_uncommitted_documentation=primary_documentation,
        accepted_tracked_files=len(accepted), changed_accepted_paths=changed,
        byte_identical_accepted_files=len(accepted)-len(changed), append_only_documents=appended,
        worker_change="Optional root_replayer callable only; accepted default remains train-only; engine/targets/ledger unchanged",
        original_external_identities=external, protected_heldout_and_prior_bundles_preserved=True,
        production_source_identities=settings["production_source_identities"], new_paths=new,
        production_artifacts_outside_git=True)
    path = ROOT / "runs/d08/preservation.json"
    with path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ("original_external_identities", "production_source_identities", "new_paths")}, indent=2))


if __name__ == "__main__":
    main()
