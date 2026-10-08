"""D03 scope and protected bytes against the exact independently accepted base."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '1301d033b81ca88e51c66ba27c1035a80c50fe54'
ALLOWED = {'README.md', 'coding-plan.md', 'data/manifests/elite-2025-01-v1.json', 'tests/test_data_manifest.py'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


checks = {}
base_paths = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
for name in sorted(base_paths):
    prior = git('show', f'{BASE}:{name}')
    actual = (ROOT/name).read_bytes()
    if name not in ALLOWED:
        assert prior == actual, name
    checks[name] = dict(base_sha256=hashlib.sha256(prior).hexdigest(), current_sha256=hashlib.sha256(actual).hexdigest(),
                        byte_identical=prior==actual)
path = 'data/manifests/elite-2025-01-v1.json'
prior = json.loads(git('show', f'{BASE}:{path}'))
actual = json.loads((ROOT/path).read_text())
old_prep, new_prep = prior.pop('preparation'), actual.pop('preparation')
assert prior == actual
assert {k:v for k,v in old_prep.items() if k not in ('split_algorithm', 'split_audit_sha256')} == {k:v for k,v in new_prep.items() if k not in ('split_algorithm', 'split_audit_sha256')}
assert (ROOT/'runs/d03/replay-adapter.stdout.txt').read_bytes() == (ROOT/'runs/d03/replay-model.stdout.txt').read_bytes()
assert git('merge-base', BASE, 'HEAD').decode().strip() == BASE
assert git('branch', '--show-current').decode().strip() == 'codex/d03-canonical-splits'
changed = set(git('diff', '--name-only', BASE).decode().splitlines())
assert changed & base_paths <= ALLOWED, changed
new = (changed - base_paths) | set(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
allowed_new = {'src/chess_rl/canonical_games.py', 'scripts/prepare_d03.py', 'tests/test_canonical_games.py', 'docs/d03-canonical-splits.md'}
assert all(name in allowed_new or name.startswith('runs/d03/') for name in new), new
report = dict(base_commit=BASE, branch='codex/d03-canonical-splits', worktree=str(ROOT),
              protected_files=checks, byte_identical_files=sum(c['byte_identical'] for c in checks.values()),
              allowed_existing_changes=sorted(ALLOWED), manifest_changes_only_d03_split_fields=True,
              adapter_reports_byte_identical=True, accepted_code_dependencies_and_historical_evidence_preserved=True)
(ROOT/'runs/d03/preservation.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='protected_files'}, indent=2))
