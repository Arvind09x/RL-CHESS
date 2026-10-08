"""Review preservation check against the common base and original task commits."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f'
ALLOWED = {'README.md', 'coding-plan.md', 'data/manifests/elite-2025-01-v1.json',
           'tests/test_data_manifest.py', 'docs/s01-evaluator.md'}

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

checks = {}
for name in git('ls-tree', '-r', '--name-only', BASE).decode().splitlines():
    original = git('show', f'{BASE}:{name}')
    current = (ROOT/name).read_bytes()
    if name not in ALLOWED:
        assert current == original, name
    elif name == 'docs/s01-evaluator.md':
        assert current.startswith(original), name
    checks[name] = dict(base_sha256=sha(original), current_sha256=sha(current),
        mode='byte-identical' if current == original else 'allowed scoped change')

name = 'data/manifests/elite-2025-01-v1.json'
prior = json.loads(git('show', f'{BASE}:{name}'))
current = json.loads((ROOT/name).read_text())
assert {k:v for k,v in prior.items() if k != 'archive'} == {k:v for k,v in current.items() if k != 'archive'}
for key in current['archive']:
    if key not in ('downloaded_bytes', 'downloaded_sha256', 'retrieved_at_utc', 'verification_evidence'):
        assert current['archive'][key] == prior['archive'][key]

for commit, checkout, appended in (
    ('cb37b0df79f85ee7615d89778efbad75c1b19576', '/private/tmp/rl-chess-d02', 'docs/d02-pgn-replay.md'),
    ('88e62a418bfd8cce54bc8c7e90bae60bb5719c6a', '/private/tmp/rl-chess-s02', 'docs/s02-puct.md'),
):
    assert git('rev-parse', 'HEAD', cwd=checkout).decode().strip() == commit
    assert not git('status', '--porcelain', cwd=checkout).strip()
    for name in git('diff', '--name-only', BASE, commit).decode().splitlines():
        original = git('show', f'{commit}:{name}')
        current_bytes = (ROOT/name).read_bytes()
        if name == appended:
            assert current_bytes.startswith(original), name
        elif name not in ('README.md', 'coding-plan.md'):
            assert current_bytes == original, name

report = dict(base_commit=BASE, protected_files=checks, original_worktrees_clean=True,
              both_task_implementations_and_prior_evidence_preserved=True,
              manifest_changes_only_archive_retrieval_provenance=True)
(Path(__file__).parent/'preservation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(protected_base_files=len(checks),
                     byte_identical=sum(v['mode']=='byte-identical' for v in checks.values()),
                     original_worktrees_clean=True, task_code_and_evidence_preserved=True), indent=2))
