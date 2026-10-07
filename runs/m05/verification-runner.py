"""Exact M05 verification commands and preservation procedure used for this run."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/private/tmp/rl-chess-m05-change')
SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
MODEL_PYTHON = str(SOURCE/'.venv/bin/python')
ADAPTER_PYTHON = str(SOURCE/'.venv-acceptance/bin/python')
OUT = Path('/private/tmp/rl-chess-m05-change/runs/m05')
BASELINE = '0fdbd1eaf5e4d1701a2627d55f87b2150a4f252a'
OUT.mkdir(parents=True, exist_ok=True)
commands = []


def run(command, name, overrides=None):
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()
    env = os.environ.copy()
    overrides = {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT), **(overrides or {})}
    env.update(overrides)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    (OUT/name).write_text(result.stdout)
    commands.append({'command': command, 'cwd': str(ROOT), 'environment_overrides': overrides,
                     'started_at_utc': started_at, 'exit_code': result.returncode, 'output': 'runs/m05/'+name,
                     'duration_seconds': round(time.time()-start, 3)})
    (OUT/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    print(name, 'exit', result.returncode, result.stdout[-250:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run([MODEL_PYTHON, 'scripts/check_m05.py', '--output', str(OUT/'network-acceptance.json')], '01-m05-check.log')
run([MODEL_PYTHON, 'scripts/check_m04.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m04-rerun.json')], '02-m04-check.log')
run([MODEL_PYTHON, 'scripts/check_m03.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m03-rerun.json')], '03-m03-check.log')
run([MODEL_PYTHON, 'scripts/check_m02.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m02-rerun.json')], '04-m02-check.log')
run([MODEL_PYTHON, 'scripts/check_m01.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--output', str(OUT/'m01-rerun.json')], '05-m01-check.log')
run([MODEL_PYTHON, '-m', 'pytest', '-p', 'no:cacheprovider',
     '--junitxml='+str(OUT/'pytest-model.xml')], '06-model-tests.log')
run([ADAPTER_PYTHON, '-m', 'pytest', '-p', 'no:cacheprovider',
     '--junitxml='+str(OUT/'pytest-adapter-only.xml')], '07-adapter-only-tests.log')
run([MODEL_PYTHON, 'scripts/check_adapter.py'], '08-adapter-replay-model.json')
run([ADAPTER_PYTHON, 'scripts/check_adapter.py'], '09-adapter-replay-only.json')
run([ADAPTER_PYTHON, '-c', "import importlib.util, sys; assert importlib.util.find_spec('torch') is None; import chess_rl.adapter; assert 'torch' not in sys.modules; print('Adapter-only: no installed or imported torch')"], '10-adapter-import.log')
run(['uv', 'pip', 'check', '--python', MODEL_PYTHON], '11-model-dependencies.log',
    {'UV_CACHE_DIR': '/private/tmp/rl-chess-m05-uv-cache'})
run(['uv', 'pip', 'check', '--python', ADAPTER_PYTHON], '12-adapter-dependencies.log',
    {'UV_CACHE_DIR': '/private/tmp/rl-chess-m05-uv-cache'})
run([MODEL_PYTHON, 'scripts/check_upstream.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'], '13-upstream-check.log')
# Repository diff check is recorded in final-audit.py after the authorized archive.
