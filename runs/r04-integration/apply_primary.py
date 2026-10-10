"""Apply accepted R04 only after exact preflight; restore original unstaged patch."""
import base64,json,subprocess,sys
from datetime import datetime,timezone
from checks import *
def main():
    target=git('rev-parse','HEAD').decode().strip(); b=before();p=b['primary']
    assert git('branch','--show-current')==b'codex/r04-integration\n'
    assert not git('status','--porcelain=v1','-uall')
    summary=json.loads((OUT/'summary.json').read_text())
    acceptance=json.loads((ROOT/'runs/r04-acceptance/acceptance.json').read_text())
    assert summary['verdict']=='VERIFIED' and acceptance['verdict']=='ACCEPTED bounded CPU Gate I'
    assert summary['reviewed_head']==b['source_head'] and acceptance['actionable_findings']==[]
    assert all(v['exit_code']==0 and v['sources_unchanged'] for v in summary['commands'].values())
    for n,v in json.loads((OUT/'artifact-index.json').read_text()).items():assert identity(OUT/n)==v,n
    for n,v in json.loads((OUT/'new-external-artifact-index.json').read_text()).items():assert identity(n)==v,n
    accepted_tree();primary_unchanged();historical()
    original=base64.b64decode(p['plan_base64']);patch=base64.b64decode(p['patch_base64'])
    expected=expected_plan()
    subprocess.run(['git','apply','--check','-'],cwd=ROOT,input=patch,env=ENV,check=True)
    if '--verify-only' in sys.argv:
        print(json.dumps(dict(target=target,raw_index_exact=True,original_plan_and_patch_exact=True,artifacts_exact=True,patch_applies=True),indent=2));return
    git('fetch','--no-tags',str(ROOT),'codex/r04-integration',cwd=PRIMARY)
    assert git('rev-parse','FETCH_HEAD',cwd=PRIMARY).decode().strip()==target
    for commit in [p['head'],*summary['sequence'],summary['merge']]:
        subprocess.run(['git','merge-base','--is-ancestor',commit,target],cwd=PRIMARY,env=ENV,check=True)
    git('stash','push','-m','R04 integration exact plan preservation '+target[:12],'--','coding-plan.md',cwd=PRIMARY)
    stash=git('rev-parse','refs/stash',cwd=PRIMARY).decode().strip()
    assert git('show',stash+':coding-plan.md',cwd=PRIMARY)==original
    assert git('show',stash+'^2:coding-plan.md',cwd=PRIMARY)==git('show',p['head']+':coding-plan.md',cwd=PRIMARY)
    assert not git('status','--porcelain=v1','-uall',cwd=PRIMARY)
    git('merge','--ff-only',target,cwd=PRIMARY)
    subprocess.run(['git','apply','--check','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    subprocess.run(['git','apply','-'],cwd=PRIMARY,input=patch,env=ENV,check=True)
    restored(target);hist=historical()
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==[stash]+p['stash_list']
    git('stash','drop','stash@{0}',cwd=PRIMARY)
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==p['stash_list']
    receipt=dict(applied_at_utc=datetime.now(timezone.utc).isoformat(),local_date='2026-10-10',timezone='Asia/Kolkata',
        primary=str(PRIMARY),branch='main',previous_head=p['head'],verified_integration=target,
        retained_sequence=summary['sequence'],merge=summary['merge'],fast_forward=True,r04_accepted=True,gate_i_accepted=True,
        acceptance_scope=acceptance['scope'],suites=summary['suites'],complete_chain_verified=True,
        original_plan=p['tracked']['coding-plan.md'],restored_plan=identity(PRIMARY/'coding-plan.md'),
        original_patch_base64=p['patch_base64'],restored_patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        original_file_projection_exact=True,original_patch_hunks_exact_except_blob_hash_headers=True,
        original_raw_index_saved_losslessly=True,original_raw_index_exact_until_application=True,
        original_index=p['raw_index'],original_staged_entries_saved_losslessly=True,original_empty_staged_state_preserved=True,
        new_index_matches_integrated_head=True,user_D06_D07_E01_edits_exact_and_unstaged=True,
        recovery_stash_verified_then_removed=stash,original_stash_inventory_preserved=True,
        integrated_tree_matches_isolated=True,historical_preservation=hist,
        cuda_integrated=False,pilot_executed=False,pretrained_P_selected=False)
    save(PRIMARY/'runs/r04-integration/main-integration.json',receipt)
    git('add','--','runs/r04-integration/main-integration.json',cwd=PRIMARY)
    assert git('diff','--cached','--name-only',cwd=PRIMARY)==b'runs/r04-integration/main-integration.json\n'
    git('diff','--cached','--check',cwd=PRIMARY)
    git('commit','-m','Record verified R04 main integration and exact preservation of user planning edits',cwd=PRIMARY)
    final=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()
    assert git('diff','--name-only',target,final,cwd=PRIMARY)==b'runs/r04-integration/main-integration.json\n'
    restored(target);historical()
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==p['stash_list']
    print(json.dumps(dict(final_main=final,verified_integration=target,retained_sequence=summary['sequence'],
        status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),
        original_user_edits_exact=True,index_matches_main=True,staged_diff_empty=True,
        r04_accepted=True,gate_i_accepted=True,cuda_integrated=False,pilot_executed=False,pretrained_P_selected=False),indent=2))
if __name__=='__main__':main()

