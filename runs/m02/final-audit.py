"""Read-only final M02 scope, archive and preservation audit; run from repo root."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

root = Path.cwd()
subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
report = json.loads((root/'runs/m02/preservation-check.json').read_text())
for name, expected in report['protected_repository_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
for name, expected in report['original_plan_sha256'].items():
    assert hashlib.sha256((Path(report['original_planning_directory'])/name).read_bytes()).hexdigest() == expected, name
old = subprocess.check_output(['git', 'show', report['baseline_commit']+':docs/m01-gab.md'])
assert (root/'docs/m01-gab.md').read_bytes().startswith(old)
assert (root/'runs/m02/05-adapter-replay-model.json').read_bytes() == (root/'runs/m02/06-adapter-replay-only.json').read_bytes()
for name, passed, skipped in [('pytest-model.xml', 157, 0), ('pytest-adapter-only.xml', 119, 2)]:
    xml = ET.parse(root/'runs/m02'/name).getroot()
    cases = xml.findall('.//testcase')
    assert len(cases) == passed + skipped
    assert len(xml.findall('.//skipped')) == skipped
    assert not xml.findall('.//failure') and not xml.findall('.//error')
backbone = json.loads((root/'runs/m02/backbone-acceptance.json').read_text())
assert backbone['status'] == 'passed' and len(backbone['cases']) == 6
assert backbone['constructors']['backbone_parameter_elements'] == 3567168
assert all(len(case['gradient_max_abs_errors']) == 183 for case in backbone['cases'])
assert len(json.loads((root/'runs/m02/m01-rerun.json').read_text())['cases']) == 12
assert all(row['exit_code'] == 0 for row in json.loads((root/'runs/m02/commands.json').read_text()))
changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
allowed = {'README.md', 'coding-plan.md', 'docs/m01-gab.md', 'docs/m02-backbone.md', 'docs/upstream-sources.md', 'scripts/check_m02.py', 'src/chess_rl/model/__init__.py', 'src/chess_rl/model/attention.py', 'src/chess_rl/model/backbone.py', 'tests/backbone_checks.py', 'tests/test_backbone.py'}
assert all(name in allowed or name.startswith('runs/m02/') for name in changed)
print('Final staged diff check: passed')
print('Scope: M01 AI acceptance record and M02 only')
print('Original planning hashes, historical evidence, adapter, M01 math, pins/notices/dependencies: unchanged')
print('M01 documentation: append only')
print('Reports: 157 model passes; 119 adapter-only passes, 2 expected skips; 6 M02/12 M01 numerical cases')
print('All 182 backbone parameter gradients plus input verified in each numerical case')
print('Adapter replay outputs: byte-identical across environments')
