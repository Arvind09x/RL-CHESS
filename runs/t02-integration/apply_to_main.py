"""Fast-forward verified T02 integration, retaining the user's exact plan changes.

Refuse drift. Keep the recovery stash until restoration and receipt commit succeed.
Requires filesystem authorization because primary is outside this chat's writable roots.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from run_checks import ROOT, OUT, PRIMARY, BASE, SOURCE, identity

def git(*args, cwd=PRIMARY):
    return subprocess.check_output(['git', *args], cwd=cwd)

def main(target):
    setup = json.loads((OUT/'setup.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['integration_verification']['all_commands_exit_zero']
    assert summary['real_continuation']['exact_match']
    assert summary['source_commit'] == SOURCE and summary['base_commit'] == BASE
    assert git('rev-parse', 'HEAD', cwd=ROOT).decode().strip() == target
    assert git('status', '--porcelain=v1', '-uall', cwd=ROOT) == b''
    assert git('merge-base', '--is-ancestor', SOURCE, target, cwd=ROOT) == b''
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    assert git('branch', '--show-current') == b'main\n'
    assert git('status', '--porcelain=v1', '-uall').decode() == setup['primary_status']
    assert not git('diff', '--cached', '--name-only')
    current = (PRIMARY/'coding-plan.md').read_bytes()
    assert identity(PRIMARY/'coding-plan.md') == setup['primary_coding_plan_identity']
    patch = base64.b64decode(json.loads((OUT/'primary-coding-plan-patch.json').read_text())['data'], validate=True)
    assert git('diff', '--binary', '--', 'coding-plan.md') == patch
    original = git('show', BASE+':coding-plan.md')
    a, b = original.splitlines(keepends=True), current.splitlines(keepends=True)
    assert len(a) == len(b)
    edits = [(x,y) for x,y in zip(a,b) if x != y]
    assert len(edits) == 3
    projected = git('show', target+':coding-plan.md', cwd=ROOT)
    for (old,new), milestone in zip(edits, ('D06','D07','E01')):
        assert old.startswith(f'| [ ] | {milestone} /'.encode())
        assert new == old.replace(b'| [ ] |', b'| [x] |', 1)
        assert projected.count(old) == 1
        projected = projected.replace(old,new,1)
    transcript = []
    def record(*args):
        started = datetime.now(timezone.utc).isoformat()
        raw = git(*args)
        transcript.append(dict(argv=['git',*args], cwd=str(PRIMARY), started_at_utc=started,
                               finished_at_utc=datetime.now(timezone.utc).isoformat(), stdout=raw.decode()))
        return raw
    record('fetch', str(ROOT), 'refs/heads/codex/t02-integration')
    assert git('rev-parse', 'FETCH_HEAD').decode().strip() == target
    record('stash', 'push', '-m', 'Preserve primary coding-plan edits for verified T02 integration', '--', 'coding-plan.md')
    stash = git('rev-parse', 'stash@{0}').decode().strip()
    assert git('show', stash+':coding-plan.md') == current
    record('merge', '--ff-only', target)
    (PRIMARY/'coding-plan.md').write_bytes(projected)
    assert git('status', '--porcelain=v1', '-uall').decode() == setup['primary_status']
    assert (PRIMARY/'coding-plan.md').read_bytes() == projected
    verified = 0
    for line in git('ls-tree', '-r', target).decode().splitlines():
        metadata, name = line.split('\t')
        if name == 'coding-plan.md': continue
        raw = (PRIMARY/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == metadata.split()[-1], name
        verified += 1
    # Confirm the restored patch has exactly the original three line changes.
    new_committed = git('show', target+':coding-plan.md')
    new_edits = [(x,y) for x,y in zip(new_committed.splitlines(keepends=True),projected.splitlines(keepends=True)) if x != y]
    assert new_edits == edits
    preserved = json.loads((OUT/'preservation-after.json').read_text())
    for path, expected in preserved['external_files'].items():
        assert identity(Path(path)) == expected, path
    workspace = json.loads((OUT/'workspace-external-after.json').read_text())
    for path, expected in workspace['files'].items():
        assert identity(Path(path)) == expected, path
    artifacts = json.loads((OUT/'external-artifacts.json').read_text())
    for path, expected in artifacts['files'].items():
        assert identity(Path(path)) == expected, path
    receipt = dict(integrated_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE,
        reviewed_source_commit=SOURCE, verified_result_commit=target, primary_checkout=str(PRIMARY),
        review_type='Independent AI review supplied by user; no human review claimed',
        exact_integrated_files_verified=verified, original_user_changes=['D06','D07','E01'],
        original_user_changes_exact_and_uncommitted=True, restored_primary_plan_identity=identity(PRIMARY/'coding-plan.md'),
        all_indexed_external_and_historical_workspace_artifacts_rehashed_unchanged=True,
        fresh_correctness_artifacts_rehashed=True, integration_evidence='summary.json', commands=transcript,
        receipt_commit_scope='only runs/t02-integration/main-integration.json; no source or user plan edits',
        scope='Stop after T02 integration; no T03, T04, pilot training, P selection, self-play or RL')
    path = PRIMARY/'runs/t02-integration/main-integration.json'
    with path.open('x') as f:
        f.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    record('add','--','runs/t02-integration/main-integration.json')
    assert git('diff','--cached','--name-only') == b'runs/t02-integration/main-integration.json\n'
    record('commit','-m','Record verified T02 integration into main and preservation of primary edits')
    assert git('status','--porcelain=v1','-uall').decode() == setup['primary_status']
    assert (PRIMARY/'coding-plan.md').read_bytes() == projected
    assert git('merge-base','--is-ancestor',SOURCE,'HEAD') == b''
    assert git('diff-tree','--no-commit-id','--name-only','-r','HEAD') == b'runs/t02-integration/main-integration.json\n'
    assert git('rev-parse','stash@{0}').decode().strip() == stash
    record('stash','drop','stash@{0}')
    print(json.dumps(dict(main_commit=git('rev-parse','HEAD').decode().strip(), verified_result_commit=target,
        primary_status=setup['primary_status'], exact_user_patch_preserved_and_uncommitted=True,
        verified_files=verified, receipt=str(path)),indent=2))

if __name__ == '__main__': main(sys.argv[1])
