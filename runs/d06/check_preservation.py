"""D06 preservation against exact accepted base and original bulk identities."""
import hashlib
import json
from pathlib import Path
import subprocess

from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[2]
BASE = "4184387c7e4b67559355d6b212a2a300c86c5ce6"
PRIMARY = Path("/Users/arvindsrinivass/Desktop/RL-Chess")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


assert git("rev-parse", "HEAD").decode().strip() == BASE
assert git("branch", "--show-current").decode().strip() == "codex/d06-annotation-benchmark"
assert git("rev-parse", "HEAD", cwd=PRIMARY).decode().strip() == BASE
assert git("branch", "--show-current", cwd=PRIMARY).decode().strip() == "main"
assert not git("status", "--porcelain", cwd=PRIMARY).strip()
checks = {}
for name in git("ls-tree", "-r", "--name-only", BASE).decode().splitlines():
    original = git("show", f"{BASE}:{name}")
    current = (ROOT/name).read_bytes()
    assert (PRIMARY/name).read_bytes() == original, name
    if name in ("README.md", "coding-plan.md"):
        assert current.startswith(original), name
    else:
        assert current == original, name
    checks[name] = dict(base_sha256=hashlib.sha256(original).hexdigest(),
        current_sha256=hashlib.sha256(current).hexdigest(), unchanged=current == original)
external = json.loads((ROOT/"runs/d04/preservation.json").read_text())["external_identities"]
bulk = Path("/private/tmp/rl-chess-d04-prepared")
audit = json.loads((bulk/"overlap-audit.json").read_text())
for name, expected in audit["outputs"].items():
    external[str(bulk/name)] = expected
for name, expected in json.loads(Path("/private/tmp/rl-chess-d02-accepted-prepared/summary.json").read_text())["outputs"].items():
    external[str(Path("/private/tmp/rl-chess-d02-accepted-prepared")/name)] = expected
for path, expected in external.items():
    assert file_identity(Path(path)) == expected, path
assert (ROOT/"runs/d06/replay-adapter.stdout.txt").read_bytes() == (ROOT/"runs/d06/replay-model.stdout.txt").read_bytes()
assert (ROOT/"runs/d06/replay-adapter.stdout.txt").read_bytes() == (ROOT/"runs/d05-s05-review/adapter-frozen-replay.log").read_bytes()
report = dict(base_commit=BASE, primary_main_clean_and_unchanged=True,
    checkout=str(ROOT), branch="codex/d06-annotation-benchmark", base_files=len(checks),
    unchanged_base_files=sum(r["unchanged"] for r in checks.values()),
    status_documents_append_only=True, accepted_code_pins_teacher_historical_evidence_preserved=True,
    original_inputs_and_protected_registries_preserved=True, adapter_reports_match=True,
    external_identities=external, files=checks)
(ROOT/"runs/d06/preservation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("files", "external_identities")}, indent=2))
