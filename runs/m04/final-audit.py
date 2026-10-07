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
commands = json.loads((OUT/'commands.json').read_text())


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


run(['git', 'diff', '--check'], '14-final-diff-check.log')
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
