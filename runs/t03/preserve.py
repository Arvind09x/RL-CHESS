"""Exact inherited tree, external trust roots and untouched primary snapshot."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
primary = Path('/Users/arvindsrinivass/Desktop/RL-Chess')

def git(cwd, *args):
    return subprocess.check_output(['git', *args], cwd=cwd)

def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for b in iter(lambda: handle.read(1024 * 1024), b''): h.update(b)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())

def snapshot():
    inherited = git(root, 'ls-tree', '-r', '--name-only', '4fa48d4').decode().splitlines()
    plan = json.loads((root / 'runs/t03/frozen-plan.json').read_text())
    return dict(inherited={n: identity(root / n) for n in inherited},
        external={n: identity(Path(n)) for n in plan['inputs'] if n.startswith('/private/tmp/')},
        primary_head=git(primary, 'rev-parse', 'HEAD').decode().strip(),
        primary_status=git(primary, 'status', '--porcelain').decode(),
        primary_patch_base64=base64.b64encode(git(primary, 'diff', '--binary')).decode(),
        primary_files={n: identity(primary / n) for n in git(primary, 'ls-files').decode().splitlines()},
        frozen_plan=identity(root / 'runs/t03/frozen-plan.json'))

mode = sys.argv[1]
path = root / 'runs/t03' / f'preservation-{mode}.json'
current = snapshot()
if mode == 'after':
    before = json.loads((root / 'runs/t03/preservation-before.json').read_text())
    for key in ('external', 'primary_head', 'primary_status', 'primary_patch_base64', 'primary_files', 'frozen_plan'):
        assert before[key] == current[key], key
    changed = [n for n in before['inherited'] if before['inherited'][n] != current['inherited'][n]]
    assert set(changed) <= {'README.md', 'coding-plan.md'}
    for n in changed:
        original = git(root, 'show', '4fa48d4:' + n)
        assert (root / n).read_bytes().startswith(original), 'original documentation must be retained'
    current['verified'] = dict(inherited_files=len(before['inherited']), allowed_append_only_changes=changed,
        accepted_sources_and_historical_evidence_unchanged=True, external_trust_roots_unchanged=True,
        primary_head_and_exact_uncommitted_edits_unchanged=True, frozen_plan_unchanged=True)
with path.open('x') as handle: handle.write(json.dumps(current, indent=2, sort_keys=True) + '\n')
print(mode, len(current['inherited']), 'inherited files', len(current['external']), 'external files')
