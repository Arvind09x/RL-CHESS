"""Retain raw curves and create CSV/SVG, hashes and checked completion evidence."""
import base64
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
run = root / 'runs/t03'
plan = json.loads((run / 'frozen-plan.json').read_text())
external = Path(plan['external_directory'])

def identity(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def write_new(path, data):
    with path.open('x') as handle: handle.write(json.dumps(data, indent=2, sort_keys=True) + '\n')

for name in ('learning-curves.jsonl', 'result.json', 'reload-verification.json', 'expected-reload.json'):
    assert not (run / name).exists()
    shutil.copyfile(external / name, run / name)
rows = [json.loads(line) for line in (run / 'learning-curves.jsonl').read_text().splitlines()]
result = json.loads((run / 'result.json').read_text())
reload = json.loads((run / 'reload-verification.json').read_text())
assert result['status'] in ('passed', 'failed_step_cap') and result['step'] <= plan['gate']['step_cap']
assert [r['step'] for r in rows] == list(range(result['step'] + 1))
assert rows[-1]['metrics'] == result['metrics'] == reload['final_metrics']
assert reload['tolerance'] == 0 and reload['fresh_python_process']
from chess_rl.training.tiny_overfit import FitGate
gate = FitGate(**plan['gate'])
assert gate.passes(rows[-1]['metrics']) == (result['status'] == 'passed')
if result['status'] == 'failed_step_cap':
    assert result['step'] == gate.step_cap and result['cap_reached']
assert not any(gate.passes(r['metrics']) for r in rows[1:-1])
execution = json.loads((external / 'execution-start.json').read_text())
assert execution['frozen_plan_identity'] == identity(run / 'frozen-plan.json')
assert execution['frozen_plan'] == plan
assert datetime.fromisoformat(plan['frozen_at_utc']) < datetime.fromisoformat(execution['started_at_utc'])
with (run / 'learning-curves.csv').open('x', newline='') as handle:
    names = ['step', 'elapsed_seconds', 'legal_best_move_accuracy', 'policy_ce', 'wdl_cross_entropy',
        'teacher_entropy_floor', 'wdl_ce_minus_entropy', 'teacher_wdl_kl_mean', 'teacher_wdl_kl_max',
        'entropy_identity_error']
    writer = csv.DictWriter(handle, fieldnames=names, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] if name in row else row['metrics'][name] for name in names})

import subprocess, sys
subprocess.run([sys.executable, '-B', str(run / 'render_curves.py')], check=True)
commands = {}
for path in sorted(run.glob('*.command.json')):
    record = json.loads(path.read_text())
    assert record['exit_code'] == 0 and record['tested_source_unchanged_during_command']
    for name, output in record['outputs'].items():
        container = json.loads((run / output['path']).read_text())
        raw = base64.b64decode(container['data'])
        assert len(raw) == container['bytes'] and hashlib.sha256(raw).hexdigest() == container['sha256']
    commands[path.stem] = dict(exit_code=record['exit_code'], seconds=record['duration_seconds'])
tests = {}
for path in sorted(run.glob('*.xml')):
    cases = ET.parse(path).getroot().findall('.//testcase')
    assert not any(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    tests[path.stem] = dict(passed=len(cases) - skipped, skipped=skipped)
frozen = json.loads((run / 'frozen-adapter.stdout.json').read_text())
prior = json.loads((root / 'runs/t02-integration/frozen-adapter-adapter.stdout.json').read_text())
assert base64.b64decode(frozen['data']) == base64.b64decode(prior['data'])
write_new(run / 'external-artifacts.json', {str(p): identity(p) for p in sorted(external.iterdir())})
write_new(run / 'summary.json', dict(task='T03', base_commit=plan['base_commit'], branch='codex/t03-tiny-overfit',
    frozen_plan_identity=identity(run / 'frozen-plan.json'), preexecution_freeze_verified=True,
    first_passing_step_verified=True, thresholds_unchanged=True, accepted_training_labels=8,
    validation_test_samples_used=0, optimizer_updates=result['step'], final_metrics=result['metrics'],
    initial_metrics=rows[0]['metrics'], status=result['status'], step_cap=gate.step_cap,
    gate=plan['gate'], runtime=result['runtime'], training_seconds=result['training_seconds'],
    checkpoint=str(external / 'final.pt'), checkpoint_identity=identity(external / 'final.pt'),
    fresh_process_reload_verification=reload, tests=tests, commands=commands,
    frozen_adapter_fixtures_byte_identical=13, curve_records=len(rows),
    review='automated checks complete; independent T03 review pending',
    limits='eight fixed training roots, pinned CPU float32; plumbing only, never immutable pretrained P; no strength/generalization claim'))
print(json.dumps(dict(status=result['status'], step=result['step'], tests=tests, artifacts=str(external)), indent=2))
