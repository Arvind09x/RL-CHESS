"""Audit D01 scope and preservation against the user's exact M06 base commit."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/d01'
BASE = '4765a8cb3d6ae9cf50619be283176ab9ac17cca6'
ALLOWED_EDITS = {'README.md', 'coding-plan.md', 'docs/m06-relative.md'}
ALLOWED_NEW = {'data/manifests/elite-2025-01-v1.json', 'docs/d01-data-manifest.md',
               'src/chess_rl/data_manifest.py', 'scripts/check_d01.py', 'tests/test_data_manifest.py'}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')

protected = {}
files = git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
for name in files:
    before = git('show', BASE+':'+name)
    current = (ROOT/name).read_bytes()
    if name not in ALLOWED_EDITS:
        assert current == before, name
        protected[name] = hashlib.sha256(current).hexdigest()
    elif name == 'docs/m06-relative.md':
        assert current.startswith(before), name
    elif name == 'coding-plan.md':
        historical_log = before.split(b'## Completion log\n', 1)[1]
        assert current.split(b'## Completion log\n', 1)[1].startswith(historical_log)
old = json.loads((ROOT/'runs/m01/original-plan-preservation.json').read_text())
original = {name: hashlib.sha256((Path(old['directory'])/name).read_bytes()).hexdigest()
            for name in old['sha256']}
assert original == old['sha256']
status = git('status', '--porcelain=v1').decode()
changed = set(git('diff', '--name-only', BASE).decode().splitlines())
new = set(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
changed |= new
assert all(name in ALLOWED_EDITS | ALLOWED_NEW or name.startswith('runs/d01/') for name in changed)
for line in (ROOT/'coding-plan.md').read_text().splitlines():
    if any('| '+task+' /' in line for task in ('D02','D03','D04','D05','D06','D07','D08','S01')):
        assert '| [ ] |' in line
commands = json.loads((OUT/'commands.json').read_text())
assert all(c['exit_code'] == c['expected_exit_code'] for c in commands)
report = {'baseline_commit': BASE, 'protected_file_count': len(protected),
          'protected_repository_sha256': protected, 'allowed_existing_edits': sorted(ALLOWED_EDITS),
          'historical_completion_log_retained': True, 'm06_document_exact_prefix_retained': True,
          'original_planning_directory': old['directory'], 'original_plan_sha256': original,
          'adapter_model_contracts_dependency_source_pins_previous_tests_checkers_runs_unchanged': True,
          'd02_d08_and_s01_unchecked': True}
save('preservation-check.json', report)
# The evidence itself and scope audit are included in the focused change inventory.
changed |= {'runs/d01/preservation-check.json', 'runs/d01/change-manifest.json'}
save('change-manifest.json', {'baseline_commit': BASE, 'branch': git('branch','--show-current').decode().strip(),
                            'worktree': str(ROOT), 'scope': 'D01 only plus shared status synchronization',
                            'review_status': 'D01 automated preparation; stop for independent review',
                            'changed_files': sorted(changed),
                            'sha256_excluding_self': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                      for name in sorted(changed) if name != 'runs/d01/change-manifest.json'},
                            'protected_baseline_files': len(protected)})
print('Preserved', len(protected), 'baseline files, historical completion entries, M06 prefix and three original plans')
print('Focused change inventory:', len(changed), 'files')
