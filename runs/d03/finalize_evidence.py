"""Index saved checks, verify output hashes, and inventory the scoped deliverable."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = '1301d033b81ca88e51c66ba27c1035a80c50fe54'
commands = {}
for path in sorted(OUT.glob('*.command.json')):
    record = json.loads(path.read_text())
    name = path.name.removesuffix('.command.json')
    expected = {'focused-initial': 1, 'annotation-gate': 1, 'staged-whitespace': 2}.get(name, 0)
    assert record['exit_code'] == expected, name
    for stream in ('stdout', 'stderr'):
        raw = (OUT/record[stream]['path']).read_bytes()
        assert len(raw) == record[stream]['bytes']
        assert hashlib.sha256(raw).hexdigest() == record[stream]['sha256']
    commands[name] = dict(record=path.name, exit_code=record['exit_code'], expected_exit_code=expected)
audit = json.loads((OUT/'split-audit.json').read_text())
report = dict(implementation_base=BASE, branch='codex/d03-canonical-splits', worktree=str(ROOT),
              automated_checks_complete=True, independent_review_status='pending; no acceptance claimed',
              authoritative_bulk_output='/private/tmp/rl-chess-d03-prepared',
              repeated_bulk_output='/private/tmp/rl-chess-d03-reproduced',
              commands=commands, counts=audit['counts'], actual_split_counts=audit['actual_split_counts'],
              split_audit_sha256=hashlib.sha256((OUT/'split-audit.json').read_bytes()).hexdigest(),
              limitations=['D04 sampling/phases/position overlap/protected registries unresolved',
                           'full legal replay relies on accepted D02; fresh D03 audit replays 290 sampled games',
                           'reordering and expansion tested on fixtures; full corpus prepared twice in original order',
                           'CPU macOS arm64 Python 3.12.12 only; no annotation or GPU validation'])
(OUT/'completion.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().splitlines()
names = set(git('diff', '--name-only', BASE)) | set(git('ls-files', '--others', '--exclude-standard'))
inventory = {name: dict(bytes=(ROOT/name).stat().st_size, sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
             for name in sorted(names) if name != 'runs/d03/change-manifest.json'}
(OUT/'change-manifest.json').write_text(json.dumps(dict(base_commit=BASE, files=inventory), indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(saved_commands_verified=len(commands), scoped_files=len(inventory), independent_review='pending'), indent=2))
