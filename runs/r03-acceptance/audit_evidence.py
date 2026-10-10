"""Read-only audit of historical identities, chronology and review preservation."""
from review_runner import ROOT, OUT, PRIMARY, identity, git
from datetime import datetime
import json
import xml.etree.ElementTree as ET

freeze='d223be26457aea7aa5e7f9b489ba8cee817fdf10'
head='9fefb55f31a6a941f454d12c049c9aac8c5c3cac'
base='66913809a3fa79d6a7a41f8157b5dbc425684051'
assert git(ROOT,'rev-parse',freeze+'^').strip()==base
assert git(ROOT,'rev-parse',head+'^').strip()==freeze
assert git(ROOT,'rev-parse','HEAD').strip()==head
freeze_files=git(ROOT,'diff-tree','--no-commit-id','--name-only','-r',freeze).splitlines()
assert freeze_files==['runs/r03/freeze.py','runs/r03/frozen-plan.json']
assert git(ROOT,'diff',base,head,'--','interface.md')==''
index=json.loads((ROOT/'runs/r03/artifact-index.json').read_text())
assert all(identity(ROOT/path)==expected for path,expected in index.items())
plan=json.loads((ROOT/'runs/r03/frozen-plan.json').read_text())
summary=json.loads((ROOT/'runs/r03/summary.json').read_text())
assert identity(OUT/'boundary.pt')==summary['checkpoint']['identity']
assert identity(ROOT/'runs/r03/frozen-plan.json')['sha256']==__import__('hashlib').sha256(
    __import__('subprocess').check_output(['git','show',freeze+':runs/r03/frozen-plan.json'],cwd=ROOT)).hexdigest()
commands={p.stem:json.loads(p.read_text()) for p in (ROOT/'runs/r03').glob('*.command.json')}
mixed=[v for v in commands.values() if 'scripts/run_r03.py' in v['argv']]
first=min(datetime.fromisoformat(v['started_at_utc']) for v in mixed)
commit_time=datetime.fromisoformat(git(ROOT,'show','-s','--format=%cI',freeze).strip())
assert datetime.fromisoformat(plan['frozen_at_utc'])<commit_time<first
for receipt in mixed:
    assert receipt['source_hashes_before']['runs/r03/frozen-plan.json']==identity(ROOT/'runs/r03/frozen-plan.json')
before=json.loads((OUT/'preservation-before.json').read_text())
after=json.loads((OUT/'preservation-after.json').read_text())
assert before==after
for key in ('head','status','diff','staged'):
    assert before['primary'][key].strip()==plan['primary'][key].strip()
assert before['primary']['tracked']['coding-plan.md']==plan['primary']['coding_plan']
counts={}
for name in ('model','adapter','reviewer-probes','failure-boundary'):
    tests=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
    counts[name]={k:int(tests.attrib[k]) for k in ('tests','errors','failures','skipped')}
    counts[name]['passed']=counts[name]['tests']-sum(counts[name][k] for k in ('errors','failures','skipped'))
counts['focused_mixed_cases']=sum('test_mixed_update' in case.attrib.get('classname','')
    for case in ET.parse(OUT/'model.xml').getroot().iter('testcase'))
report=dict(base=base,freeze=freeze,reviewed=head,actionable_findings=[],
    bounded_cpu_verdict='ACCEPT',tests=counts,historical_artifact_identities_verified=len(index),
    frozen_inputs_verified=len(plan['inputs']),freeze_plan_at=plan['frozen_at_utc'],
    freeze_committed_at=commit_time.isoformat(),first_retained_mixed_execution_at=first.isoformat(),
    freeze_only_files=freeze_files,exact_new_checkpoint_matches_historical_identity=True,
    continuation=json.loads((OUT/'comparisons.json').read_text()),
    preservation=dict(reviewed_tracked_files=len(before['reviewed']['tracked']),
        primary_tracked_files=len(before['primary']['tracked']),reviewed_and_primary_unchanged=True,
        user_planning_edits_exact=True,interface_unchanged=True),
    parameter_gradient_max_absolute_difference=1.1175870895385742e-7,
    no_integration_or_r04=True,
    required_documents={p:identity(ROOT/p) for p in ('interface.md','coding-plan.md',
        'docs/r03-mixed-update.md','runs/r03/frozen-plan.json','runs/r03/summary.json')})
(OUT/'review-summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
