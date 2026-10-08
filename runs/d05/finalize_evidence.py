"""D05 preservation, command/output identities and review-boundary evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = "ad8438acfbd859f734daaebc6115afdab0367b65"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


checks = {}
for name in git("ls-tree", "-r", "--name-only", BASE).decode().splitlines():
    original = git("show", f"{BASE}:{name}")
    current = (ROOT / name).read_bytes()
    if name == "coding-plan.md":
        assert current.startswith(original), "Historical completion log changed"
    else:
        assert original == current, name
    checks[name] = dict(base_sha256=hashlib.sha256(original).hexdigest(),
        current_sha256=hashlib.sha256(current).hexdigest(), unchanged=original == current)
primary = Path("/Users/arvindsrinivass/Desktop/RL-Chess")
assert git("rev-parse", "HEAD", cwd=primary).decode().strip() == BASE
assert git("branch", "--show-current", cwd=primary).decode().strip() == "main"
assert not git("status", "--porcelain", cwd=primary).strip()
assert git("branch", "--show-current").decode().strip() == "codex/d05-stockfish-worker"
settings = json.loads((OUT / "tiny-final/settings.json").read_text())
paths = {"manifest": ROOT / "data/manifests/elite-2025-01-d04-v1.json",
    "games.jsonl": Path("/private/tmp/rl-chess-d03-prepared/games.jsonl"),
    "split-audit.json": Path("/private/tmp/rl-chess-d03-prepared/split-audit.json")}
for name in ("overlap-audit.json", "samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl"):
    paths[name] = Path("/private/tmp/rl-chess-d04-prepared") / name
identities = {name: file_identity(path) for name, path in paths.items()}
assert identities == settings["artifact_identities"]
assert file_identity(Path(settings["teacher"]["executable"])) == settings["teacher"]["executable_identity"]
assert (OUT / "replay-adapter.stdout.txt").read_bytes() == (OUT / "replay-model.stdout.txt").read_bytes()
assert (OUT / "replay-adapter.stdout.txt").read_bytes() == (ROOT / "runs/d04-s04-review/replay-adapter.stdout.txt").read_bytes()
assert (OUT / "tiny-real/selected.jsonl").read_bytes() == (OUT / "tiny-final/selected.jsonl").read_bytes()
commands = []
for path in sorted(OUT.glob("*.command.json")):
    command = json.loads(path.read_text())
    for stream in ("stdout", "stderr"):
        identity = file_identity(OUT / command[stream]["path"])
        assert identity == {k: command[stream][k] for k in ("bytes", "sha256")}
    commands.append(dict(path=path.name, **command))
required_success = ("focused-strict-final", "final-adapter-regression", "final-model-regression",
    "reviewer-adapter", "reviewer-model", "replay-adapter", "replay-model", "sample-replay-strict",
    "sample-replay-initial", "teacher-capabilities", "teacher-bytes", "m06-check",
    "pinned-sources-local", "pinned-sources-network-retry", "failure-fixtures", "tiny-final")
for name in required_success:
    assert json.loads((OUT / f"{name}.command.json").read_text())["exit_code"] == 0, name
assert json.loads((OUT / "historical-gate.command.json").read_text())["exit_code"] == 1
tests = {}
for name in ("focused-strict-final", "final-adapter-regression", "final-model-regression", "reviewer-adapter", "reviewer-model"):
    suite = ET.parse(OUT / f"{name}.xml").getroot().find("testsuite")
    tests[name] = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    assert tests[name]["failures"] == tests[name]["errors"] == 0
summary = json.loads((OUT / "tiny-final/summary.json").read_text())
replay = json.loads((OUT / "tiny-final/independent-replay.json").read_text())
assert summary["successful_labels"] == replay["successful_labels_replayed"] == 12
preservation = dict(base_commit=BASE, base_files=len(checks),
    unchanged_base_files=sum(c["unchanged"] for c in checks.values()),
    coding_plan_append_only=True, all_original_artifacts_preserved=True,
    primary_main_clean_and_unchanged=True, protected_heldout_membership_unchanged=True,
    input_identities=identities, files=checks)
(OUT / "preservation.json").write_text(json.dumps(preservation, indent=2, sort_keys=True)+"\n")
(OUT / "commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True)+"\n")
completion = dict(task="D05 only", review_status="independent review pending", accepted=False,
    base_commit=BASE, branch="codex/d05-stockfish-worker", checkout=str(ROOT),
    active_manifest="data/manifests/elite-2025-01-d04-v1.json",
    api="await AnnotationWorker.annotate(reference, canonical_game) -> label dict | None",
    cli="scripts/annotate_d05.py", final_sample=summary,
    independent_replay=dict(successful_labels_replayed=replay["successful_labels_replayed"], coverage=replay["coverage"]),
    real_attempts_across_initial_and_final_runs=24, distinct_real_references=12,
    initial_run_preserved=True, tests=tests, input_identities=identities,
    evidence={name: file_identity(OUT / name) for name in
        ("commands.json", "preservation.json", "tiny-final/independent-replay.json", "tiny-final/uci-transport.log",
         "failure-fixtures/attempts.jsonl", "failure-fixtures/summary.json")},
    both_adapter_replays_identical=True, dependencies_resynced=False, teacher_rebuilt=False,
    main_merged=False, D06_benchmark_run=False, D08_production_run=False, training_run=False,
    search_integration=False, sampling_or_heldout_decisions_changed=False,
    limitations=["CPU macOS arm64/Python 3.12.12 only", "Tiny plumbing subset; no throughput/cost or strength claims",
        "Sequential worker; no concurrent engine diagnostic observers", "No crash recovery, resume or cross-run deduplication",
        "Two initial fake-NNUE fixture setup failures preserved before correction; sandbox DNS failure preserved before successful retry",
        "Worktree creation was recorded from the execution tool; UTC timestamps were not captured before the wrapper existed"])
(OUT / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in completion.items() if k not in ("input_identities", "evidence", "final_sample")}, indent=2))
