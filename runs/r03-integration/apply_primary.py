"""Apply verified R03 and restore exact unstaged user edits; refuse drift."""
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from checks import ROOT,OUT,PRIMARY,ENV,git,identity,save,external_preserved

BEFORE=json.loads((OUT/'preservation-before.json').read_text())
P=BEFORE['primary']
def normalize(patch):
    return b'\n'.join(line for line in patch.split(b'\n') if not line.startswith(b'index '))
def expected_user_plan(final_plan):
    result=final_plan
    for task in ('D06','D07','E01'):
        old=f'| [ ] | {task} /'.encode(); assert result.count(old)==1
        result=result.replace(old,f'| [x] | {task} /'.encode(),1)
    return result
def check_restored(target,expected,patch,receipt=False):
    names=git('ls-tree','-r','--name-only',target).decode().splitlines()
    for name in names:
        if name=='coding-plan.md': assert (PRIMARY/name).read_bytes()==expected
        else: assert identity(PRIMARY/name)==identity(ROOT/name),name
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert git('write-tree',cwd=PRIMARY)==git('rev-parse','HEAD^{tree}',cwd=PRIMARY)
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY)==b' M coding-plan.md\n'
    assert normalize(git('diff','--binary',cwd=PRIMARY))==normalize(patch)
    for task in ('R03',): assert f'| [x] | {task} /'.encode() in git('show','HEAD:coding-plan.md',cwd=PRIMARY)
    assert b'| [ ] | R04 /' in git('show','HEAD:coding-plan.md',cwd=PRIMARY)
def main():
    target=git('rev-parse','HEAD').decode().strip()
    assert git('branch','--show-current')==b'codex/r03-integration\n'
    assert not git('status','--porcelain=v1','-uall'),'isolated result must be committed and clean'
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['verdict']=='ACCEPTED' and summary['implementation']==BEFORE['source_head']
    assert all(v['exit_code']==0 for v in summary['commands'].values())
    assert all(json.loads((OUT/'comparisons.json').read_text()).values())
    acceptance=json.loads((ROOT/'runs/r03-acceptance/acceptance.json').read_text())
    assert acceptance['reviewed']==BEFORE['source_head'] and acceptance['base']==P['head']
    assert acceptance['actionable_findings']==[] and acceptance['verdict']=='ACCEPTED'
    for name,item in json.loads((OUT/'artifact-index.json').read_text()).items(): assert identity(OUT/name)==item,name
    for name,item in json.loads((OUT/'new-external-artifact-index.json').read_text()).items(): assert identity(name)==item,name
    for name,item in json.loads((OUT/'accepted-tree-inventory.json').read_text()).items():
        if name not in ('README.md','coding-plan.md'): assert identity(ROOT/name)==item,name
    candidate_plan=git('show',BEFORE['source_head']+':coding-plan.md')
    final_plan=(ROOT/'coding-plan.md').read_bytes()
    assert final_plan.startswith(candidate_plan.replace(b'| [ ] | R03 /',b'| [x] | R03 /',1))
    assert (ROOT/'README.md').read_bytes().startswith(git('show',BEFORE['source_head']+':README.md'))
    original_plan=base64.b64decode(P['plan_base64']); patch=base64.b64decode(P['patch_base64'])
    expected=expected_user_plan(final_plan)
    assert expected[:len(original_plan)].replace(b'| [x] | R03 /',b'| [ ] | R03 /',1)==original_plan
    assert git('branch','--show-current',cwd=PRIMARY)==b'main\n'
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()==P['head']
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode()==P['status']
    assert (PRIMARY/'.git/index').read_bytes()==base64.b64decode(P['index_base64'])
    assert git('ls-files','--stage','-z',cwd=PRIMARY)==base64.b64decode(P['index_entries_base64'])
    assert git('diff','--binary',cwd=PRIMARY)==patch
    assert git('diff','--cached','--binary',cwd=PRIMARY)==base64.b64decode(P['staged_patch_base64'])==b''
    assert (PRIMARY/'coding-plan.md').read_bytes()==original_plan
    for name,item in P['tracked'].items(): assert identity(PRIMARY/name)==item,name
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    external_preserved()
    subprocess.run(['git','apply','--check','-'],cwd=ROOT,input=patch,env=ENV,check=True)
    if '--verify-only' in sys.argv:
        print(json.dumps(dict(target=target,primary_drift=False,raw_index_exact=True,
            original_patch_applies=True,artifacts_verified=True),indent=2)); return
    git('fetch','--no-tags',str(ROOT),'codex/r03-integration',cwd=PRIMARY)
    assert git('rev-parse','FETCH_HEAD',cwd=PRIMARY).decode().strip()==target
    for commit in (P['head'],acceptance['freeze'],BEFORE['source_head'],summary['merge']):
        subprocess.run(['git','merge-base','--is-ancestor',commit,target],cwd=PRIMARY,env=ENV,check=True)
    git('stash','push','-m','R03 integration exact coding-plan preservation '+target[:12],'--','coding-plan.md',cwd=PRIMARY)
    stash=git('rev-parse','refs/stash',cwd=PRIMARY).decode().strip()
    assert git('show',stash+':coding-plan.md',cwd=PRIMARY)==original_plan
    assert git('show',stash+'^2:coding-plan.md',cwd=PRIMARY)==git('show',P['head']+':coding-plan.md',cwd=PRIMARY)
    assert not git('status','--porcelain=v1','-uall',cwd=PRIMARY)
    git('merge','--ff-only',target,cwd=PRIMARY)
    subprocess.run(['git','apply','--check','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    subprocess.run(['git','apply','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    check_restored(target,expected,patch)
    external_preserved()
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==[stash]+P['stash_list']
    git('stash','drop','stash@{0}',cwd=PRIMARY)
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    receipt=dict(applied_at_utc=datetime.now(timezone.utc).isoformat(),local_date='2026-10-10',timezone='Asia/Kolkata',
        primary=str(PRIMARY),branch='main',previous_head=P['head'],verified_integration_commit=target,
        freeze_retained=acceptance['freeze'],implementation_retained=BEFORE['source_head'],merge=summary['merge'],
        fast_forward=True,r03_accepted=True,acceptance_scope=acceptance['scope'],suites=summary['suites'],
        exact_mixed_continuation=summary['mixed_continuation'],source_audit=summary['source_audit'],
        original_plan=P['tracked']['coding-plan.md'],restored_plan=identity(PRIMARY/'coding-plan.md'),
        original_patch_base64=P['patch_base64'],restored_patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        original_file_projection_exact=True,original_patch_hunks_exact_except_blob_hash_headers=True,
        original_raw_index_saved_losslessly=True,original_raw_index_exact_until_application=True,
        original_index=P['index'],original_staged_patch_preserved=True,new_index_matches_integrated_head=True,
        D06_D07_E01_edits_exact_and_uncommitted=True,only_uncommitted_path='coding-plan.md',
        recovery_stash_verified_then_removed=stash,original_stash_inventory_preserved=True,
        integrated_tree_matches_isolated=True,historical_artifacts_unchanged=external_preserved(),
        r04_started=False,cuda_integrated=False,receipt_commit_scope='runs/r03-integration/main-integration.json')
    save(PRIMARY/'runs/r03-integration/main-integration.json',receipt)
    git('add','--','runs/r03-integration/main-integration.json',cwd=PRIMARY)
    assert git('diff','--cached','--name-only',cwd=PRIMARY)==b'runs/r03-integration/main-integration.json\n'
    git('diff','--cached','--check',cwd=PRIMARY)
    git('commit','-m','Record verified R03 main integration and exact preservation of user planning edits',cwd=PRIMARY)
    final=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()
    assert git('diff','--name-only',target,final,cwd=PRIMARY)==b'runs/r03-integration/main-integration.json\n'
    check_restored(target,expected,patch); external_preserved()
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    print(json.dumps(dict(final_main=final,verified_integration=target,freeze_retained=acceptance['freeze'],
        implementation_retained=BEFORE['source_head'],status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),
        exact_original_user_edits_restored=True,clean_index=True,r04_started=False,cuda_integrated=False),indent=2))
if __name__=='__main__': main()
