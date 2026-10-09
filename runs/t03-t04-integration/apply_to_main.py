"""Apply verified integration to main, preserving the exact three user line edits.

Primary is outside writable roots; execute only with filesystem authorization.
Recovery stash is retained until restoration and the receipt commit succeed.
"""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from run_checks import ROOT, OUT, PRIMARY, BASE, SOURCES, identity, primary_snapshot

def git(*args,cwd=PRIMARY):
    return subprocess.check_output(['git',*args],cwd=cwd)

def verify_tree(target, projected):
    count=0
    for row in git('ls-tree','-r',target).decode().splitlines():
        metadata,name=row.split('\t')
        raw=(PRIMARY/name).read_bytes()
        if name=='coding-plan.md':
            assert raw==projected
        else:
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==metadata.split()[-1],name
        count+=1
    return count

def main(target):
    setup=json.loads((OUT/'setup.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['all_commands_exit_zero'] and summary['acceptance']['T03'] and summary['acceptance']['T04']
    assert summary['t03_checkpoint']['exact_all_model_tensor_hashes']
    assert summary['t03_checkpoint']['exact_final_metrics']
    assert summary['t04_actual_t03_checkpoint']['input_histories_and_rng_unchanged']
    assert primary_snapshot()==setup['primary'], 'Primary changed since verification; refuse integration'
    assert git('rev-parse','HEAD',cwd=ROOT).decode().strip()==target
    assert git('status','--porcelain=v1','-uall',cwd=ROOT)==b''
    for c in [BASE,*SOURCES]:
        assert not git('merge-base','--is-ancestor',c,target,cwd=ROOT)
    # Rehash every indexed external artifact immediately before changing main.
    external=json.loads((OUT/'preservation-after.json').read_text())['external_files']
    for name,expected in external.items(): assert identity(Path(name))==expected,name
    original=git('show',BASE+':coding-plan.md')
    current=(PRIMARY/'coding-plan.md').read_bytes()
    a,b=original.splitlines(keepends=True),current.splitlines(keepends=True)
    assert len(a)==len(b)
    edits=[(x,y) for x,y in zip(a,b) if x!=y]
    assert len(edits)==3
    projected=git('show',target+':coding-plan.md',cwd=ROOT)
    for (old,new),milestone in zip(edits,('D06','D07','E01')):
        assert old.startswith(f'| [ ] | {milestone} /'.encode())
        assert new==old.replace(b'| [ ] |',b'| [x] |',1) and projected.count(old)==1
        projected=projected.replace(old,new,1)
    transcript=[]
    def record(*args):
        start=datetime.now(timezone.utc).isoformat()
        raw=git(*args)
        transcript.append(dict(argv=['git',*args],cwd=str(PRIMARY),started_at_utc=start,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),stdout=raw.decode()))
        return raw
    stash_before=git('stash','list')
    record('fetch',str(ROOT),'refs/heads/codex/t03-t04-integration')
    assert git('rev-parse','FETCH_HEAD').decode().strip()==target
    record('stash','push','-m','Preserve primary plan edits for verified T03 T04 integration','--','coding-plan.md')
    stash=git('rev-parse','stash@{0}').decode().strip()
    assert git('show',stash+':coding-plan.md')==current
    assert git('status','--porcelain=v1','-uall')==b''
    record('merge','--ff-only',target)
    (PRIMARY/'coding-plan.md').write_bytes(projected)
    assert git('status','--porcelain=v1','-uall').decode()==setup['primary']['status']
    new_committed=git('show',target+':coding-plan.md')
    actual_edits=[(x,y) for x,y in zip(new_committed.splitlines(keepends=True),projected.splitlines(keepends=True)) if x!=y]
    assert actual_edits==edits
    verified=verify_tree(target,projected)
    for name,expected in external.items(): assert identity(Path(name))==expected,name
    receipt=dict(integrated_at_utc=datetime.now(timezone.utc).isoformat(),primary_checkout=str(PRIMARY),
        base_commit=BASE,reviewed_source_commits=SOURCES,verified_result_commit=target,
        exact_integrated_files_verified=verified,all_indexed_external_artifacts_unchanged=True,
        review_type='Supplied independent AI review; no human or newly commissioned review',
        accepted_tasks=['T03','T04'],original_user_changes=['D06','D07','E01'],
        original_user_changes_exact_and_uncommitted=True,
        original_primary_plan_identity=setup['primary']['plan'],restored_plan_identity=identity(PRIMARY/'coding-plan.md'),
        frozen_t03_plan_and_historical_evidence_preserved=True,original_overfit_rerun=False,
        integration_evidence='runs/t03-t04-integration/summary.json',commands=transcript,
        receipt_commit_scope='only runs/t03-t04-integration/main-integration.json',
        scope='Stop after integration; no R01, GPU work or training campaign')
    receipt_path=PRIMARY/'runs/t03-t04-integration/main-integration.json'
    with receipt_path.open('x') as f: f.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    record('add','--','runs/t03-t04-integration/main-integration.json')
    assert git('diff','--cached','--name-only')==b'runs/t03-t04-integration/main-integration.json\n'
    record('commit','-m','Record verified T03 T04 integration and preservation of primary edits')
    assert git('status','--porcelain=v1','-uall').decode()==setup['primary']['status']
    assert git('diff-tree','--no-commit-id','--name-only','-r','HEAD')==b'runs/t03-t04-integration/main-integration.json\n'
    verify_tree(target,projected)
    for c in SOURCES: assert not git('merge-base','--is-ancestor',c,'HEAD')
    assert git('rev-parse','stash@{0}').decode().strip()==stash
    record('stash','drop','stash@{0}')
    assert git('stash','list')==stash_before
    print(json.dumps(dict(main_commit=git('rev-parse','HEAD').decode().strip(),verified_result_commit=target,
        original_user_edits_exact_and_uncommitted=True,original_stash_list_preserved=True,
        reviewed_commits_preserved=SOURCES,verified_files=verified,receipt=str(receipt_path)),indent=2))

if __name__=='__main__': main(sys.argv[1])
