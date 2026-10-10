"""Verify independent-process R03 equality and preserve accepted inputs/main."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/r03'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')

def read(name): return json.loads((OUT/name).read_text())

def identity(path):
    h = hashlib.sha256(); count = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            count += len(block); h.update(block)
    return dict(bytes=count,sha256=h.hexdigest())

def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd).decode()

def tests(name):
    suites = ET.parse(OUT/name).getroot().findall('testsuite')
    totals = {key:sum(int(s.attrib[key]) for s in suites) for key in ('tests','failures','errors','skipped')}
    totals['passed'] = totals['tests']-totals['failures']-totals['errors']-totals['skipped']
    assert totals['failures'] == totals['errors'] == 0
    return totals

def main():
    import torch
    plan = read('frozen-plan.json')
    full,prefix,resume = [read(name) for name in ('bounded-full-final.json','bounded-prefix.json','bounded-resume.json')]
    assert prefix['steps'] == full['steps'][:1]
    assert resume['steps'] == full['steps'][1:]
    assert full['final'] == resume['final']
    assert len(full['steps']) == 3 and len(prefix['steps']) == 1 and len(resume['steps']) == 2
    for run in (full,prefix,resume):
        assert run['settings'] == full['settings']
        assert run['supervised_source_id'] == full['supervised_source_id']
        assert run['supervised_membership_sha256'] == full['supervised_membership_sha256']
        assert run['replay_id'] == plan['replay']['export_sha256']
        assert run['replay_membership'] == plan['replay']['sample_ids']
    for path,expected in plan['inputs'].items():
        assert identity(path) == expected, f'accepted source changed: {path}'
    current_primary = dict(head=git('rev-parse','HEAD',cwd=PRIMARY).strip(),
        status=git('status','--porcelain=v1',cwd=PRIMARY),diff=git('diff','--binary',cwd=PRIMARY),
        staged=git('diff','--cached','--binary',cwd=PRIMARY),coding_plan=identity(PRIMARY/'coding-plan.md'))
    assert current_primary == plan['primary']
    unchanged_primary = 0
    for name in git('ls-files',cwd=PRIMARY).splitlines():
        if name == 'coding-plan.md': continue
        original = subprocess.check_output(['git','show',plan['base_commit']+':'+name],cwd=PRIMARY)
        assert (PRIMARY/name).read_bytes() == original,f'primary tracked file changed: {name}'
        unchanged_primary += 1
    commands = ('bounded-full-final','bounded-prefix','bounded-resume','model-fixtures','adapter-fixtures-final')
    for name in commands:
        receipt = read(name+'.command.json')
        assert receipt['exit_code'] == 0 and receipt['sources_unchanged']
        for path,expected in receipt['source_hashes_after'].items():
            assert identity(ROOT/path) == expected,f'tested source drift: {path}'
    checkpoint = Path('/private/tmp/r03-bounded-boundary-final.pt')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert payload['version'] == 'm06-checkpoint-v2-mixed'
    training = payload['training']
    assert 'data_order' not in training
    assert training['progress'] == dict(step=1,epoch=0,samples_seen=4)
    mixed = training['mixed_state']
    assert mixed['supervised']['cursor'] == mixed['selfplay']['cursor'] == 2
    assert mixed['supervised']['samples_seen'] == mixed['selfplay']['samples_seen'] == 2
    assert mixed['supervised_source_id'] == full['supervised_source_id']
    assert mixed['supervised_membership_sha256'] == full['supervised_membership_sha256']
    assert mixed['replay_id'] == full['replay_id']
    assert list(mixed['replay_membership']) == full['replay_membership']
    assert mixed['sampling_rng'] and training['rng'] and training['optimizer']['state']['state']
    summary = dict(task='R03 implemented; independent review pending; stop before R04',
        verified_at_utc=datetime.now(timezone.utc).isoformat(),base_commit=plan['base_commit'],
        freeze_commit='d223be2',branch=git('branch','--show-current').strip(),settings=full['settings'],
        model_fixtures=tests('model-fixtures.xml'),adapter_fixtures=tests('adapter-fixtures-final.xml'),
        mixed_test_cases=len(ET.parse(OUT/'model-fixtures.xml').getroot().findall(".//testcase[@classname='tests.test_mixed_update']")),
        fresh_process=dict(independent_processes=3,identical_prefix=True,identical_updates_2_and_3=True,
            identical_final_model_adam_gradients_progress_sampling_rng_and_global_rng=True,
            comparison='SHA256 over complete tensor bytes, dtype and shape; scalar metrics and state compared exactly',
            synthetic_fixture='test_exact_cpu_fresh_process_next_update crosses supervised and replay epoch boundaries'),
        per_source_counts=dict(supervised=6,selfplay=6),
        losses=[dict(update=s['update'],actual=s['metrics'],independent=s['independent']) for s in full['steps']],
        gradient_tensors=[s['finite_parameter_gradients'] for s in full['steps']],
        checkpoint=dict(path=str(checkpoint),identity=identity(checkpoint),version=payload['version'],
            preserved_fields=sorted(training),mixed_fields=sorted(mixed)),
        sources=dict(supervised_source_id=full['supervised_source_id'],
            supervised_membership_sha256=full['supervised_membership_sha256'],replay_id=full['replay_id'],
            replay_membership=full['replay_membership'],excluded_heldout=full['excluded_heldout']),
        preservation=dict(primary_exact=True,primary_tracked_files_unchanged=unchanged_primary,
            primary_head=current_primary['head'],primary_status=current_primary['status'],
            frozen_inputs_unchanged=len(plan['inputs']),interface_unchanged=identity(ROOT/'interface.md')==identity(PRIMARY/'interface.md')),
        historical_diagnostics=dict(initial_50_case_development='50 passed; later added six contract/edge cases',
            bounded_full='passing development receipt before adding frozen-input refusal to runner; final run supersedes it',
            adapter_fixtures='initial command mistakenly loaded Torch-only historical M05 diagnostic plugin in Torch-free environment; no production issue; corrected command omits plugin'))
    (OUT/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('model_fixtures','adapter_fixtures','mixed_test_cases','fresh_process','preservation')}))

if __name__ == '__main__': main()
