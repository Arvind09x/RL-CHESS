"""Apply committed verified R02, retaining user edits and refusing primary drift."""
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from checks import ROOT,OUT,PRIMARY,SOURCE,ENV,git,identity,save

BEFORE=json.loads((OUT/'preservation-before.json').read_text())
P=BEFORE['primary']
SUMMARY=json.loads((OUT/'summary.json').read_text())
def external_preserved():
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip()==BEFORE['source_head']
    assert git('status','--porcelain=v1','-uall',cwd=SOURCE).decode()==BEFORE['source_status']
    for n,item in BEFORE['source_tracked'].items(): assert identity(SOURCE/n)==item,n
    for n,item in BEFORE['review'].items(): assert identity(SOURCE/'runs/r02-review-20261010'/n)==item,n
    for n,item in BEFORE['r01_tracked'].items(): assert identity(ROOT.parent/'r01-self-play'/n)==item,n
    for n,item in BEFORE['external'].items(): assert identity(n)==item,n
def normalize_patch(patch):
    return b'\n'.join(line for line in patch.split(b'\n') if not line.startswith(b'index '))
def verify_primary(target,expected_plan,patch):
    for n in git('ls-tree','-r','--name-only',target).decode().splitlines():
        if n=='coding-plan.md': assert (PRIMARY/n).read_bytes()==expected_plan
        else: assert identity(PRIMARY/n)==identity(ROOT/n),n
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert git('write-tree',cwd=PRIMARY)==git('rev-parse','HEAD^{tree}',cwd=PRIMARY)
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY)==b' M coding-plan.md\n'
    assert normalize_patch(git('diff','--binary',cwd=PRIMARY))==normalize_patch(patch)
def main():
    target=git('rev-parse','HEAD').decode().strip()
    assert git('branch','--show-current')==b'codex/r02-integration\n'
    assert not git('status','--porcelain=v1','-uall'),'isolated result must be committed and clean'
    assert SUMMARY['verdict']=='ACCEPTED' and SUMMARY['accepted_implementation_commit']==BEFORE['source_head']
    assert all(c['exit_code']==0 for c in SUMMARY['commands'].values()) and len(SUMMARY['commands'])==6
    review=json.loads((OUT/'supplied-review/summary.json').read_bytes())
    assert review['commit']==BEFORE['source_head'] and review['accepted_base']==P['head']
    assert review['actionable_findings']==[]
    for n,item in json.loads((OUT/'artifact-index.json').read_text()).items(): assert identity(OUT/n)==item,n
    for n,item in json.loads((OUT/'accepted-tree-inventory.json').read_text()).items():
        if n not in ('README.md','coding-plan.md'): assert identity(ROOT/n)==item,n
    candidate_plan=git('show',BEFORE['source_head']+':coding-plan.md')
    final_plan=(ROOT/'coding-plan.md').read_bytes()
    assert final_plan.startswith(candidate_plan.replace(b'| [ ] | R02 /',b'| [x] | R02 /',1))
    assert (ROOT/'README.md').read_bytes().startswith(git('show',BEFORE['source_head']+':README.md'))
    assert b'| [ ] | R03 /' in final_plan and b'| [x] | R02 /' in final_plan
    expected_plan=final_plan
    for task in ('D06','D07','E01'):
        old=f'| [ ] | {task} /'.encode(); assert expected_plan.count(old)==1
        expected_plan=expected_plan.replace(old,f'| [x] | {task} /'.encode(),1)
    original_plan=base64.b64decode(P['plan_base64']); patch=base64.b64decode(P['patch_base64'])
    assert expected_plan[:len(original_plan)].replace(b'| [x] | R02 /',b'| [ ] | R02 /',1)==original_plan
    assert git('branch','--show-current',cwd=PRIMARY)==b'main\n'
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()==P['head']
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode()==P['status']
    assert (PRIMARY/'.git/index').read_bytes()==base64.b64decode(P['index_base64'])
    assert git('ls-files','--stage','-z',cwd=PRIMARY)==base64.b64decode(P['index_entries_base64'])
    assert git('diff','--binary',cwd=PRIMARY)==patch
    assert git('diff','--cached','--binary',cwd=PRIMARY)==base64.b64decode(P['staged_patch_base64'])==b''
    assert (PRIMARY/'coding-plan.md').read_bytes()==original_plan
    for n,item in P['tracked'].items(): assert identity(PRIMARY/n)==item,n
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    external_preserved()
    subprocess.run(['git','apply','--check','-'],cwd=ROOT,input=patch,env=ENV,check=True)
    if '--verify-only' in sys.argv:
        print(json.dumps(dict(verified_target=target,primary_drift=False,original_patch_applies=True,
            original_index_exact=True,artifacts_verified=True),indent=2))
        return
    git('fetch','--no-tags',str(ROOT),'codex/r02-integration',cwd=PRIMARY)
    assert git('rev-parse','FETCH_HEAD',cwd=PRIMARY).decode().strip()==target
    for commit in (P['head'],BEFORE['source_head'],SUMMARY['integration_merge']):
        subprocess.run(['git','merge-base','--is-ancestor',commit,target],cwd=PRIMARY,env=ENV,check=True)
    git('stash','push','-m','r02-integration exact original coding-plan preservation '+target[:12],'--','coding-plan.md',cwd=PRIMARY)
    stash=git('rev-parse','refs/stash',cwd=PRIMARY).decode().strip()
    assert git('show',stash+':coding-plan.md',cwd=PRIMARY)==original_plan
    assert git('show',stash+'^2:coding-plan.md',cwd=PRIMARY)==git('show',P['head']+':coding-plan.md',cwd=PRIMARY)
    assert not git('status','--porcelain=v1','-uall',cwd=PRIMARY)
    git('merge','--ff-only',target,cwd=PRIMARY)
    subprocess.run(['git','apply','--check','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    subprocess.run(['git','apply','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    verify_primary(target,expected_plan,patch)
    external_preserved()
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==[stash]+P['stash_list']
    git('stash','drop','stash@{0}',cwd=PRIMARY)
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    receipt=dict(applied_at_utc=datetime.now(timezone.utc).isoformat(),local_date='2026-10-10',timezone='Asia/Kolkata',
        primary=str(PRIMARY),branch='main',previous_head=P['head'],verified_integration_commit=target,
        retained_implementation_commit=BEFORE['source_head'],integration_merge=SUMMARY['integration_merge'],
        fast_forward=True,review_acceptance=SUMMARY['scope'],suites=SUMMARY['suites'],fresh_commands_verified=6,
        original_plan=P['tracked']['coding-plan.md'],restored_plan=identity(PRIMARY/'coding-plan.md'),
        original_patch_base64=P['patch_base64'],restored_patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        patch_hunks_exact_except_required_blob_hash_headers=True,original_file_projection_exact=True,
        original_index=P['index'],original_raw_index_exact_until_application=True,original_raw_index_saved_losslessly=True,
        original_staged_patch_and_clean_staged_state_preserved=True,new_index_matches_integrated_head=True,
        original_D06_D07_E01_edits_exact_and_uncommitted=True,only_uncommitted_path='coding-plan.md',
        recovery_stash_verified_then_removed=stash,original_stash_inventory_preserved=True,
        full_integrated_tree_matches_isolated=True,original_r01_r02_and_review_artifacts_unchanged=True,
        external_artifacts_verified=len(BEFORE['external']),review_artifacts_verified=len(BEFORE['review']),
        archive_verification=SUMMARY['archive_verification'],r02_marked_accepted=True,r03_started=False,cuda_integrated=False,
        receipt_commit_scope='Only runs/r02-integration/main-integration.json')
    path=PRIMARY/'runs/r02-integration/main-integration.json'; save(path,receipt)
    git('add','--','runs/r02-integration/main-integration.json',cwd=PRIMARY)
    assert git('diff','--cached','--name-only',cwd=PRIMARY)==b'runs/r02-integration/main-integration.json\n'
    git('diff','--cached','--check',cwd=PRIMARY)
    git('commit','-m','Record verified R02 main integration and exact preservation of user planning edits',cwd=PRIMARY)
    final_head=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()
    assert git('diff','--name-only',target,final_head,cwd=PRIMARY)==b'runs/r02-integration/main-integration.json\n'
    verify_primary(target,expected_plan,patch)
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==P['stash_list']
    external_preserved()
    print(json.dumps(dict(final_main=final_head,verified_integration=target,retained_implementation=BEFORE['source_head'],
        status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),exact_original_user_edits_restored=True,
        clean_index=True,r03_started=False,cuda_integrated=False),indent=2))
if __name__=='__main__': main()
