"""Validate lossless evidence and publish integration acceptance summary once."""
import base64
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from run_checks import ROOT, OUT, BASE, SOURCES, identity, save

commands={}
for path in sorted(OUT.glob('*.command.json')):
    d=json.loads(path.read_text())
    assert d['exit_code']==0 and d['tested_source_unchanged_during_command']
    for label,expected in d['outputs'].items():
        container=OUT/expected['path']
        c=json.loads(container.read_text())
        raw=base64.b64decode(c['data'],validate=True)
        import hashlib
        assert len(raw)==c['bytes']==expected['decoded_bytes']
        assert hashlib.sha256(raw).hexdigest()==c['sha256']==expected['decoded_sha256']
        assert identity(container)['sha256']==expected['container_sha256']
    commands[path.stem.removesuffix('.command')]=d
assert len(commands)==15, len(commands)
results={}
for path in sorted(OUT.glob('*.xml')):
    d=ET.parse(path).getroot().find('testsuite').attrib
    assert int(d['failures'])==int(d['errors'])==0
    results[path.stem]=dict(passed=int(d['tests'])-int(d['skipped']),skipped=int(d['skipped']),
        errors=int(d['errors']),failures=int(d['failures']),evidence=path.name,identity=identity(path))
assert results['regressions-model']['passed']==1398
assert results['regressions-adapter']['passed']==1023
assert results['checkpoint-training-tests']['passed']==206
assert all(results[n]['passed']==84 for n in ['accepted-review-probes-model','accepted-review-probes-adapter'])
for kind in ['model','adapter']:
    old=ROOT/'runs/t02-integration'/f'frozen-adapter-{kind}.stdout.json'
    new=OUT/f'frozen-adapter-{kind}.stdout.json'
    assert base64.b64decode(json.loads(old.read_text())['data'])==base64.b64decode(json.loads(new.read_text())['data'])
checkpoint=json.loads((OUT/'t03-checkpoint-identity-metrics.json').read_text())
t04=json.loads((OUT/'t03-through-t04.json').read_text())
assert t04['checkpoint']['sha256']==checkpoint['checkpoint_identity']['sha256']
assert t04['mixed_forward_shapes']==[[16,64,128]] and t04['terminal_searches_bypassed']==12
assert len(t04['predictions'])==31 and len(t04['bounded_search'])==8
assert all(r['budget']==r['completed']==3 for r in t04['bounded_search'])
# Existing raw failed-test logs are historical evidence and must remain exact.
diff=subprocess.run(['git','diff','--check',BASE],cwd=ROOT,capture_output=True)
assert diff.returncode==2
lines=diff.stdout.decode().splitlines()
locations=[s.split(':')[0] for s in lines if ': trailing whitespace.' in s]
assert len(locations)==4 and set(locations)=={'runs/t04/focused.stdout.txt','runs/t04/focused.xml'}
for name in set(locations):
    assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',SOURCES[1]+':'+name],cwd=ROOT)
save(OUT/'historical-whitespace-check.json',dict(argv=['git','diff','--check',BASE],exit_code=2,
    stdout=diff.stdout.decode(),stderr=diff.stderr.decode(),
    only_exact_reviewed_historical_failure_logs=True,preserved_files=sorted(set(locations))))
save(OUT/'commands.json',commands)
save(OUT/'summary.json',dict(base_commit=BASE,reviewed_source_commits=SOURCES,
    prepared_merge_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    acceptance=dict(T03=True,T04=True),review_type='Supplied independent AI review; no human review and no newly commissioned review',
    supplied_review=json.loads((OUT/'setup.json').read_text())['supplied_review'],
    all_commands_exit_zero=True,fresh_command_count=len(commands),results=results,
    frozen_adapter_fixtures_exact_in_both_environments=13,
    t03_checkpoint=checkpoint,t04_actual_t03_checkpoint=t04,
    historical_report_scope_note='Unmodified check_t04.py report retains its original review-pending scope string; this separate integration acceptance supersedes that historical status without rewriting source.',
    preserved_artifacts=identity(OUT/'preservation-after.json'),
    integration_harness={str(p.relative_to(ROOT)):identity(p) for p in sorted(OUT.glob('*.py'))},
    original_overfit_rerun=False,frozen_plan_modified=False,production_code_fixes_required=False,
    scope='Stop after primary main integration; no R01, GPU work, training campaigns, P selection or games'))
print(json.dumps(dict(commands=len(commands),results=results,acceptance={'T03':True,'T04':True}),indent=2))
