"""Verify the combined code, prior evidence, intake and repeated bulk outputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = '1301d033b81ca88e51c66ba27c1035a80c50fe54'
D03 = '7eaba97d14d8e4d769b5db7e8a4e5a3931940d1f'
S03 = 'e4a1e8b8374d463b3389f503ba5bae29be85ef0e'


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


allowed = {'README.md', 'coding-plan.md', 'data/manifests/elite-2025-01-v1.json',
           'tests/test_data_manifest.py', 'src/chess_rl/search/puct.py'}
files = git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
checks = {}
for name in files:
    original = git('show', f'{BASE}:{name}')
    current = (ROOT/name).read_bytes()
    if name not in allowed:
        assert original == current, name
    checks[name] = dict(base_sha256=hashlib.sha256(original).hexdigest(),
                        current_sha256=hashlib.sha256(current).hexdigest(), unchanged=original == current)
for commit, worktree in [(D03, '/private/tmp/rl-chess-d03-canonical-splits'),
                         (S03, '/private/tmp/rl-chess-s03')]:
    assert git('rev-parse', 'HEAD', cwd=worktree).decode().strip() == commit
    assert not git('status', '--porcelain', cwd=worktree).strip()
    for name in git('diff', '--name-only', BASE, commit).decode().splitlines():
        original = git('show', f'{commit}:{name}')
        current = (ROOT/name).read_bytes()
        if name in {'README.md', 'coding-plan.md', 'docs/d03-canonical-splits.md', 'docs/s03-value-backup.md'}:
            # Append-only review records are allowed; coding-plan status is updated later.
            if name != 'coding-plan.md':
                assert current.startswith(original), name
        else:
            assert current == original, name
prior = json.loads(git('show', f'{BASE}:data/manifests/elite-2025-01-v1.json'))
current = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text())
for name in ('split_algorithm', 'split_audit_sha256'):
    prior['preparation'][name] = current['preparation'][name]
assert prior == current
bulk = {}
for name in ('games.jsonl', 'split-audit.json'):
    first = Path('/private/tmp/rl-chess-d03-prepared')/name
    second = Path('/private/tmp/rl-chess-d03-reproduced')/name
    assert first.stat().st_size == second.stat().st_size and sha(first) == sha(second)
    bulk[name] = dict(bytes=first.stat().st_size, sha256=sha(first))
assert (OUT/'replay-adapter.stdout.txt').read_bytes() == (OUT/'replay-model.stdout.txt').read_bytes()
report = dict(base_commit=BASE, base_files=len(files),
              unchanged_base_files=sum(item['unchanged'] for item in checks.values()),
              original_task_worktrees_clean=True, original_task_code_and_evidence_preserved=True,
              manifest_only_two_D03_decisions_changed=True, repeated_bulk_output_identities=bulk,
              both_adapter_replays_identical=True, files=checks)
(OUT/'preservation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
