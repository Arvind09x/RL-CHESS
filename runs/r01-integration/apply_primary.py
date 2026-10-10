"""Apply the verified committed integration, refusing drift and preserving user edits."""
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from run_checks import ROOT, OUT, PRIMARY, identity, git, save

SOURCE = Path('/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/r01-self-play')
REVIEW = SOURCE.parent / 'independent-review-20261010'
BEFORE = json.loads((OUT/'preservation-before.json').read_text())

def check_shared_artifacts():
    for n, item in BEFORE['external_files'].items():
        assert identity(n) == item, n
    for n, item in BEFORE['review_artifacts'].items():
        assert identity(REVIEW/n) == item, n
    assert identity(REVIEW/'artifact-index.json') == BEFORE['review_index_identity']
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip() == BEFORE['source_head']
    assert not git('status','--porcelain=v1','-uall',cwd=SOURCE)

def verify_tree(target, expected_plan):
    for n in git('ls-tree','-r','--name-only',target,cwd=PRIMARY).decode().splitlines():
        if n == 'coding-plan.md':
            assert (PRIMARY/n).read_bytes() == expected_plan
        else:
            assert identity(PRIMARY/n) == identity(ROOT/n), n
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY) == b' M coding-plan.md\n'
    assert git('diff','--name-only',cwd=PRIMARY) == b'coding-plan.md\n'

def main():
    target = git('rev-parse','HEAD').decode().strip()
    assert not git('status','--porcelain=v1','-uall'), 'isolated checkout must be committed and clean'
    assert git('branch','--show-current').decode().strip() == 'codex/r01-integration'
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip() == BEFORE['head']
    assert git('branch','--show-current',cwd=PRIMARY).decode().strip() == 'main'
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode() == BEFORE['status']
    patch = base64.b64decode(BEFORE['patch_base64'])
    assert git('diff','--binary',cwd=PRIMARY) == patch
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    for n, item in BEFORE['tracked_files'].items():
        assert identity(PRIMARY/n) == item, n
    check_shared_artifacts()
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['verdict'] == 'ACCEPTED'
    assert summary['accepted_implementation_commit'] == BEFORE['source_head']
    assert len(summary['commands']) == 13 and all(c['exit_code']==0 for c in summary['commands'].values())
    for n, item in json.loads((OUT/'artifact-index.json').read_text()).items():
        assert identity(OUT/n) == item, n
    accepted_inventory = json.loads((OUT/'accepted-tree-inventory.json').read_text())
    for n, item in accepted_inventory.items():
        if n not in ('README.md','coding-plan.md'):
            assert identity(ROOT/n) == item, n
    original_plan = git('show',BEFORE['source_head']+':coding-plan.md')
    final_plan = (ROOT/'coding-plan.md').read_bytes()
    accepted_prefix = original_plan.replace(b'| [ ] | R01 /', b'| [x] | R01 /',1)
    assert final_plan.startswith(accepted_prefix)
    assert (ROOT/'README.md').read_bytes().startswith(git('show',BEFORE['source_head']+':README.md'))
    assert b'| [ ] | R02 /' in final_plan
    assert not (ROOT/'runtime/cuda').exists()
    expected_plan = final_plan
    for task in ('D06','D07','E01'):
        old = f'| [ ] | {task} /'.encode()
        assert expected_plan.count(old) == 1
        expected_plan = expected_plan.replace(old,f'| [x] | {task} /'.encode(),1)
    saved_original = base64.b64decode(BEFORE['original_plan_base64'])
    original_projection = expected_plan[:len(saved_original)].replace(b'| [x] | R01 /',b'| [ ] | R01 /',1)
    assert original_projection == saved_original, 'all original file bytes must survive acceptance additions'
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines() == BEFORE['stash_list']
    git('fetch','--no-tags',str(ROOT),'codex/r01-integration',cwd=PRIMARY)
    assert git('rev-parse','FETCH_HEAD',cwd=PRIMARY).decode().strip() == target
    subprocess.run(['git','merge-base','--is-ancestor',BEFORE['head'],target],cwd=PRIMARY,check=True)
    subprocess.run(['git','merge-base','--is-ancestor',BEFORE['source_head'],target],cwd=PRIMARY,check=True)
    stash_message = 'r01-integration exact original coding-plan preservation ' + target[:12]
    git('stash','push','-m',stash_message,'--','coding-plan.md',cwd=PRIMARY)
    stash = git('rev-parse','refs/stash',cwd=PRIMARY).decode().strip()
    assert git('show',stash+':coding-plan.md',cwd=PRIMARY) == base64.b64decode(BEFORE['original_plan_base64'])
    assert not git('status','--porcelain=v1','-uall',cwd=PRIMARY)
    git('merge','--ff-only',target,cwd=PRIMARY)
    # Reapply the original lossless patch; never replace the integrated plan with
    # the old complete file or commit the user's three checkbox changes.
    subprocess.run(['git','apply','--check','-'],cwd=PRIMARY,input=patch,check=True)
    subprocess.run(['git','apply','-'],cwd=PRIMARY,input=patch,check=True)
    verify_tree(target,expected_plan)
    check_shared_artifacts()
    assert git('rev-parse','refs/stash',cwd=PRIMARY).decode().strip() == stash
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines() == [stash]+BEFORE['stash_list']
    # Recovery bytes also remain in preservation-before.json and this chat's work/.
    git('stash','drop','stash@{0}',cwd=PRIMARY)
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines() == BEFORE['stash_list']
    receipt = dict(applied_at_utc=datetime.now(timezone.utc).isoformat(),local_date='2026-10-10',
        primary=str(PRIMARY),branch='main',previous_head=BEFORE['head'],verified_integration_commit=target,
        retained_implementation_commit=BEFORE['source_head'],fast_forward=True,
        integration_merge=summary['integration_merge'],review_acceptance='ACCEPTED within documented CPU R01 scope',
        full_committed_tree_matches_isolated_except_exact_original_user_patch=True,
        original_plan_before=BEFORE['original_plan'],restored_plan=identity(PRIMARY/'coding-plan.md'),
        original_patch_base64=BEFORE['patch_base64'],restored_patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        exact_original_D06_D07_E01_substitutions_verified=True,
        original_file_projection_exact=True,clean_index=True,only_uncommitted_path='coding-plan.md',
        recovery_stash=stash,recovery_stash_verified_and_removed=True,original_stash_inventory_preserved=True,
        external_files_verified=76,original_review_artifacts_verified=213,
        original_implementation_checkout_clean_and_unchanged=True,
        tests=summary['suites'],successful_fresh_commands=13,
        real_snapshots_equal=20,crash_resume_core_sha256=summary['real_results']['continuation']['resumed_core_sha256'],
        no_R02_or_CUDA_integration=True,receipt_commit_scope='Only this main-integration.json; user plan remains uncommitted')
    receipt_path=PRIMARY/'runs/r01-integration/main-integration.json'
    save(receipt_path,receipt)
    git('add','--','runs/r01-integration/main-integration.json',cwd=PRIMARY)
    assert git('diff','--cached','--name-only',cwd=PRIMARY) == b'runs/r01-integration/main-integration.json\n'
    git('diff','--cached','--check',cwd=PRIMARY)
    git('commit','-m','Record verified R01 main integration and exact preservation of user planning edits',cwd=PRIMARY)
    final_head=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()
    assert git('diff','--name-only',target,final_head,cwd=PRIMARY) == b'runs/r01-integration/main-integration.json\n'
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY) == b' M coding-plan.md\n'
    assert (PRIMARY/'coding-plan.md').read_bytes() == expected_plan
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines() == BEFORE['stash_list']
    check_shared_artifacts()
    print(json.dumps(dict(final_main=final_head,verified_integration=target,retained_implementation=BEFORE['source_head'],
        final_status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),original_user_modification_exact=True),indent=2))

if __name__ == '__main__':
    main()
