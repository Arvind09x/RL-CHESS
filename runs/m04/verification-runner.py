"""Exact M04 verification commands and preservation procedure used for this run."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
OUT = Path('/private/tmp/rl-chess-m04-change/runs/m04')
BASELINE = '83d8ac1bbaffe54bbf5cf3d02daf542982c03736'
OUT.mkdir(parents=True, exist_ok=True)
commands = []


def run(command, name, overrides=None):
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()
    env = os.environ.copy()
    overrides = {'PYTHONDONTWRITEBYTECODE': '1', **(overrides or {})}
    env.update(overrides)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    (OUT/name).write_text(result.stdout)
    commands.append({'command': command, 'cwd': str(ROOT), 'environment_overrides': overrides,
                     'started_at_utc': started_at, 'exit_code': result.returncode, 'output': 'runs/m04/'+name,
                     'duration_seconds': round(time.time()-start, 3)})
    (OUT/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    print(name, 'exit', result.returncode, result.stdout[-250:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run(['.venv/bin/python', 'scripts/check_m04.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'wdl-acceptance.json')], '01-m04-check.log')
run(['.venv/bin/python', 'scripts/check_m03.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m03-rerun.json')], '02-m03-check.log')
run(['.venv/bin/python', 'scripts/check_m02.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m02-rerun.json')], '03-m02-check.log')
run(['.venv/bin/python', 'scripts/check_m01.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m01-rerun.json')], '04-m01-check.log')
run(['.venv/bin/python', '-m', 'pytest', '-p', 'no:cacheprovider',
     '--junitxml='+str(OUT/'pytest-model.xml')], '05-model-tests.log')
run(['.venv-acceptance/bin/python', '-m', 'pytest', '-p', 'no:cacheprovider',
     '--junitxml='+str(OUT/'pytest-adapter-only.xml')], '06-adapter-only-tests.log')
run(['.venv/bin/python', 'scripts/check_adapter.py'], '07-adapter-replay-model.json')
run(['.venv-acceptance/bin/python', 'scripts/check_adapter.py'], '08-adapter-replay-only.json')
run(['.venv-acceptance/bin/python', '-c', "import importlib.util, sys; assert importlib.util.find_spec('torch') is None; import chess_rl.adapter; assert 'torch' not in sys.modules; print('Adapter-only: no installed or imported torch')"], '09-adapter-import.log')
run(['uv', 'pip', 'check', '--python', '.venv/bin/python'], '10-model-dependencies.log',
    {'UV_CACHE_DIR': '/private/tmp/rl-chess-m04-uv-cache'})
run(['uv', 'pip', 'check', '--python', '.venv-acceptance/bin/python'], '11-adapter-dependencies.log',
    {'UV_CACHE_DIR': '/private/tmp/rl-chess-m04-uv-cache'})
run(['.venv/bin/python', 'scripts/check_upstream.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'], '12-upstream-check.log')
run(['git', 'diff', '--check'], '13-diff-check.log')
# Every baseline file except the five intentional export/current-status docs edits.
allowed = {'src/chess_rl/model/__init__.py', 'README.md', 'coding-plan.md',
           'docs/upstream-sources.md', 'docs/m03-policy.md'}
files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT, text=True).splitlines()
checks = {}
for file in files:
    if file in allowed:
        continue
    old = subprocess.check_output(['git', 'show', BASELINE+':'+file], cwd=ROOT)
    new = (ROOT/file).read_bytes()
    assert old == new, file
    checks[file] = hashlib.sha256(new).hexdigest()
# Existing M03 evidence prose must survive as an exact prefix; acceptance is appended.
old = subprocess.check_output(['git', 'show', BASELINE+':docs/m03-policy.md'], cwd=ROOT)
assert (ROOT/'docs/m03-policy.md').read_bytes().startswith(old)
prior = json.loads((ROOT/'runs/m01/original-plan-preservation.json').read_text())
original = Path(prior['directory'])
hashes = {f: hashlib.sha256((original/f).read_bytes()).hexdigest() for f in prior['sha256']}
assert hashes == prior['sha256']
report = {'baseline_commit': BASELINE, 'protected_repository_sha256': checks,
          'intentionally_changed_existing_files': sorted(allowed),
          'm03_document_historical_content_retained_as_exact_prefix': True,
          'original_planning_directory': str(original), 'original_plan_sha256': hashes,
          'original_hashes_match_historical_m01_evidence': True,
          'repository_interface_and_plan1_unchanged': True,
          'all_prior_runs_adapter_m01_m02_m03_implementation_tests_scripts_pins_notices_dependencies_unchanged': True}
(OUT/'preservation-check.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print('Preservation passed:', len(checks), 'protected files and 3 original plans', flush=True)
