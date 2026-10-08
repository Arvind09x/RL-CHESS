"""Finalize D07-only evidence; no acceptance or main integration is claimed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from chess_rl.pgn_replay import file_identity

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/d07'
BASE='4184387c7e4b67559355d6b212a2a300c86c5ce6'
commands={p.stem.removesuffix('.command'):json.loads(p.read_text()) for p in sorted(OUT.glob('*.command.json'))}
for name,row in commands.items():
    for stream in ('stdout','stderr'):
        path=OUT/row[stream]['path']
        assert file_identity(path)=={k:row[stream][k] for k in ('bytes','sha256')},(name,stream)
(OUT/'commands.json').write_text(json.dumps(commands,indent=2,sort_keys=True)+'\n')
source_paths=('src/chess_rl/supervised.py','tests/test_supervised.py','runs/d07/verify_sample.py')
sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
for name in ('adapter-release','model-release','sample-release','reviewer-adapter','reviewer-model','frozen-adapter','frozen-model','preservation','source-whitespace'):
    row=commands[name]
    assert row['exit_code']==0,name
    assert row['tested_source_unchanged_during_command'] is True,name
    assert row['tested_source_sha256']==sources,name
results={}
for name in ('adapter-release','model-release','reviewer-adapter','reviewer-model'):
    root=ET.parse(OUT/f'{name}.xml').getroot()
    suite=root.find('testsuite') if root.tag=='testsuites' else root
    results[name]={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    assert results[name]['failures']==results[name]['errors']==0
assert results['model-release']['tests']==1111 and results['model-release']['skipped']==0
assert results['adapter-release']['tests']==949 and results['adapter-release']['skipped']==7
for name,skip in (('adapter-release',1),('model-release',0)):
    cases=ET.parse(OUT/f'{name}.xml').getroot().findall('.//testcase')
    d07=[c for c in cases if c.attrib.get('classname')=='tests.test_supervised']
    assert len(d07)==127 and sum(c.find('skipped') is not None for c in d07)==skip
expected=(ROOT/'runs/d05-s05-review/adapter-frozen-replay.log').read_bytes()
for name in ('frozen-adapter','frozen-model'):
    assert (OUT/f'{name}.stdout.txt').read_bytes()==expected
sample=json.loads((OUT/'sample-release.json').read_text())
assert len(sample['records'])==12 and sample['coverage']=={
    f'{phase}:{color}':2 for phase in ('opening','middlegame','endgame') for color in ('white','black')}
assert sample==json.loads((OUT/'sample-model.json').read_text())
preservation=json.loads((OUT/'preservation.json').read_text())
assert preservation['modified_base_files']==['coding-plan.md']
changed=subprocess.check_output(['git','-C',str(ROOT),'diff',BASE,'--name-only']).decode().splitlines()
allowed={'coding-plan.md','docs/d07-record-loader.md','src/chess_rl/supervised.py','tests/test_supervised.py'}
assert all(path in allowed or path.startswith('runs/d07/') for path in changed),changed
report=dict(task='D07 only',status='implemented; stop for independent review',independent_review='pending',
    base_commit=BASE,branch='codex/d07-record-loader',isolated_checkout=str(ROOT),
    main_integrated=False,tests=results,d07_model_cases=127,d07_adapter_passes=126,d07_adapter_torch_skips=1,
    successful_existing_roots_reloaded=12,coverage=sample['coverage'],source_sha256=sources,
    frozen_adapter_replays_equal=True,primary_main_preserved=True,protected_heldout_membership_preserved=True,
    preservation_report='runs/d07/preservation.json',sample_report='runs/d07/sample-release.json',
    commands_index='runs/d07/commands.json',
    changed_files=changed,
    artifact_identities={p.name:file_identity(p) for p in OUT.glob('*') if p.is_file() and p.name not in ('completion.json','finalize_evidence.py')},
    prohibited_later_work_performed=False,
    scope_exclusions=['D06 benchmark','new teacher annotation','loss implementation','training','D08','main integration'],
    limitations=['CPU macOS arm64 / Python 3.12.12 only','12 existing roots, not a new labeled dataset',
        'Eager label validation and full audited-table scans; no cache or scale claim',
        'No GPU, concurrent mutation or training/playing-strength claim'],
    finalization=dict(argv=[sys.executable,*sys.argv],cwd=str(ROOT),recorded_at_utc=datetime.now(timezone.utc).isoformat()))
(OUT/'completion.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('artifact_identities','source_sha256')},indent=2))
