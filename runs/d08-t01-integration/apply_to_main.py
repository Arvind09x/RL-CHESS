"""Apply only the verified integration, preserving the pre-existing primary plan edits.

Run after the isolated evidence commit. Refuse changed source or primary state.
The temporary Git stash is retained for recovery if an operation fails.
"""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
OUT = ROOT / 'runs/d08-t01-integration'


def git(*args, cwd=PRIMARY):
    return subprocess.check_output(['git', *args], cwd=cwd)


def main(target):
    setup = json.loads((OUT / 'setup.json').read_text())
    assert git('rev-parse', 'HEAD', cwd=ROOT).decode().strip() == target
    assert git('status', '--porcelain=v1', '-uall', cwd=ROOT) == b''
    assert git('rev-parse', 'HEAD').decode().strip() == setup['base_commit']
    assert git('branch', '--show-current').decode().strip() == 'main'
    assert git('status', '--porcelain=v1', '-uall').decode() == setup['primary_status']
    assert not git('diff', '--cached', '--name-only')
    current = (PRIMARY / 'coding-plan.md').read_bytes()
    assert hashlib.sha256(current).hexdigest() == setup['primary_coding_plan_identity']['sha256']
    patch = base64.b64decode(json.loads((OUT / 'primary-coding-plan-patch.json').read_text())['data'], validate=True)
    assert git('diff', '--binary', '--', 'coding-plan.md') == patch
    original = git('show', setup['base_commit'] + ':coding-plan.md')
    original_lines, current_lines = original.splitlines(keepends=True), current.splitlines(keepends=True)
    assert len(original_lines) == len(current_lines)
    edits = [(a, b) for a, b in zip(original_lines, current_lines) if a != b]
    assert len(edits) == 3
    projected = git('show', target + ':coding-plan.md', cwd=ROOT)
    for (a, b), milestone in zip(edits, ('D06', 'D07', 'E01')):
        assert a.startswith(f'| [ ] | {milestone} /'.encode())
        assert b == a.replace(b'| [ ] |', b'| [x] |', 1)
        assert projected.count(a) == 1
        projected = projected.replace(a, b, 1)
    transcript = []
    def record(*args):
        started = datetime.now(timezone.utc).isoformat()
        raw = git(*args)
        transcript.append(dict(argv=['git', *args], cwd=str(PRIMARY), started_at_utc=started,
                               finished_at_utc=datetime.now(timezone.utc).isoformat(), stdout=raw.decode()))
        return raw
    record('fetch', str(ROOT), 'refs/heads/codex/d08-t01-integration')
    assert git('rev-parse', 'FETCH_HEAD').decode().strip() == target
    record('stash', 'push', '-m', 'Preserve primary coding-plan edits for D08/T01 integration', '--', 'coding-plan.md')
    stash = git('rev-parse', 'stash@{0}').decode().strip()
    assert git('show', stash + ':coding-plan.md') == current
    record('merge', '--ff-only', target)
    # Restore exactly the three saved user edits on top of the verified append-only plan.
    (PRIMARY / 'coding-plan.md').write_bytes(projected)
    assert git('status', '--porcelain=v1', '-uall').decode() == setup['primary_status']
    assert (PRIMARY / 'coding-plan.md').read_bytes() == projected
    verified_files = 0
    for line in git('ls-tree', '-r', target).decode().splitlines():
        metadata, name = line.split('\t')
        if name == 'coding-plan.md':
            continue
        raw = (PRIMARY / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == metadata.split()[-1], name
        verified_files += 1
    assert git('rev-parse', 'stash@{0}').decode().strip() == stash
    record('stash', 'drop', 'stash@{0}')
    receipt = dict(integrated_at_utc=datetime.now(timezone.utc).isoformat(), primary_checkout=str(PRIMARY),
                   base_commit=setup['base_commit'], verified_result_commit=target,
                   source_commits={'D08': 'bcb208d73271451b05d1409d4bf340aaad82eabf',
                                   'T01': 'f669fc8b4de397a31768321b9105e255b0ff0e0e'},
                   primary_user_edit_milestones=['D06', 'D07', 'E01'],
                   primary_edits_remain_uncommitted=True, exact_integrated_files_verified=verified_files,
                   restored_primary_plan_sha256=hashlib.sha256(projected).hexdigest(),
                   commands=transcript, scope='D08/T01 integration only; stop before T02 or later implementation')
    path = PRIMARY / 'runs/d08-t01-integration/main-integration.json'
    with path.open('x') as f:
        f.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    record('add', '--', 'runs/d08-t01-integration/main-integration.json')
    assert git('diff', '--cached', '--name-only') == b'runs/d08-t01-integration/main-integration.json\n'
    record('commit', '-m', 'Record verified D08 T01 integration into main and preservation of primary edits')
    assert git('status', '--porcelain=v1', '-uall').decode() == setup['primary_status']
    assert (PRIMARY / 'coding-plan.md').read_bytes() == projected
    print(json.dumps(dict(main_commit=git('rev-parse', 'HEAD').decode().strip(),
                          verified_result_commit=target, primary_status=setup['primary_status'],
                          primary_edits_preserved_and_uncommitted=True, receipt=str(path)), indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
