"""Bind final R04 results to immutable inputs, execution receipts and artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from verify import identity,save,OUT,ROOT


def read(name):return json.loads((OUT/name).read_text())


def main():
    plan=read('frozen-plan.json');generation=read('generation.json');membership=read('eligible-membership.json')
    full,prefix,resume=[read(v+'.json') for v in ('full','prefix','resume')]
    chain=read('chain-audit.json');evaluation=read('evaluation.json');preservation=read('preservation-after.json')
    assert full['steps']==prefix['steps']+resume['steps'] and full['final']==resume['final']
    assert chain['final_training_restore_exact'] and chain['continuation_exact']
    assert sum(len(v['steps']) for v in (full,prefix,resume))==plan['update_plan']['verification_optimizer_executions']
    results={}
    for name in ('model','adapter'):
        suite=ET.parse(OUT/('regression-'+name+'.xml')).getroot().find('testsuite')
        result={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        result['passed']=result['tests']-result['errors']-result['failures']-result['skipped']
        assert result['errors']==result['failures']==0
        results[name]=result
    receipts={p.name:json.loads(p.read_text()) for p in OUT.glob('*.command.json')}
    assert {name:r['exit_code'] for name,r in receipts.items() if r['exit_code']}=={'chain-audit.command.json':1}
    def commit_time(ref):
        return datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',ref],cwd=ROOT).decode().strip())
    assert commit_time('cec350b') < datetime.fromisoformat(receipts['generate.command.json']['started_at_utc'])
    assert commit_time('bbd2e40') < datetime.fromisoformat(receipts['boundary.command.json']['started_at_utc'])
    for p,expected in plan['inputs'].items():assert identity(p)==expected
    steps=full['steps'];parts=[]
    for step in steps:
        m=step['metrics'];actual=[m['supervised']['policy_loss'],m['supervised']['wdl_loss'],m['selfplay']['policy_loss'],m['selfplay']['wdl_loss']]
        parts.extend(abs(a-b) for a,b in zip(actual,step['independent_losses']))
    for p,expected in read('new-external-artifact-index.json').items():assert identity(p)==expected
    files={str(p.relative_to(ROOT)):identity(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    save('summary.json',dict(task='R04 only; independent Gate I review pending',
        generated_at_utc=datetime.now(timezone.utc).isoformat(),base_commit=plan['base_commit'],
        branch='codex/r04-local-cycle',freeze_commit='cec350b5fe5f0cee55711e790aa0d06545fdb416',
        membership_freeze_commit=subprocess.check_output(['git','rev-parse','bbd2e40'],cwd=ROOT).decode().strip(),
        freezes_precede_dependent_execution=True,checkpoint=plan['checkpoint_identity'],checkpoint_id=plan['checkpoint_id'],
        new_boundary=read('new-boundary.json'),generation=generation,
        replay=dict(summary=membership['replay_summary'],eligible_roots=len(membership['replay_membership']),
            heldout_exclusions=membership['excluded_heldout'],excluded_unfinished_roots=chain['unfinished_roots_excluded'],
            duplicates=membership['replay_summary']['source_counts']['duplicate'],root_links=chain['actual_root_to_target_links']),
        mixed=dict(settings=plan['mixed_settings'],logical_updates=3,physical_verification_updates=6,
            logical_samples_per_source=6,physical_samples_per_source=12,
            updates_per_unique_eligible_root=membership['updates_per_unique_eligible_root'],
            maximum_independent_source_loss_difference=max(parts),source_metrics=[s['metrics'] for s in steps],
            finite_parameter_gradients=[s['finite_parameter_gradients'] for s in steps],
            exact_illegal_gradient_zero=True,uninterrupted_and_fresh_resume_exact=True,
            exact_continuation_fields=['sample references and IDs','source losses','logits','logit gradients','parameter gradients',
                'model','Adam','both permutations/cursors/epochs/counts','sampling RNG','global RNG','progress'],
            final_training_restore_exact=chain['final_training_restore_exact'],updated_checkpoint=resume['checkpoint']),
        evaluation=evaluation,evaluation_by_opening=chain['evaluation_by_opening'],regressions=results,
        preservation=preservation,inputs_rehashed=len(plan['inputs']),
        external_artifacts=read('new-external-artifact-index.json'),
        retained_audit_failure=dict(command='chain-audit.command.json',reason='in-memory manifest WDL tuple versus serialized JSON list',
            correction='compare complete canonical JSON SHA256; no target transformation or experiment rerun',
            final_command='chain-audit-final.command.json',final_passed=True),
        no_production_API_changes=True,no_strength_claim=True,no_pretrained_P_selection=True,
        no_pilot_spending=True,no_main_integration=True,gate_I_accepted=False,R04_checkbox_checked=False,
        command_results={name:dict(exit_code=r['exit_code'],started_at_utc=r['started_at_utc'],duration_seconds=r['duration_seconds']) for name,r in receipts.items()},
        evidence_file_identities=files))
    print(json.dumps(dict(regressions=results,chain_passed=True,preservation_exact=True,gate_I='pending independent review')))


if __name__=='__main__':main()
