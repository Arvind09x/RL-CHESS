"""Focused D01 and unchanged M06 regression checks; saves actual command evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/d01'
SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
MODEL = str(SOURCE / '.venv/bin/python')
ADAPTER = str(SOURCE / '.venv-acceptance/bin/python')
ENV = {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)}
records = []

def run(argv, name, expected=0):
    started = datetime.now(timezone.utc).isoformat()
    t = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, env={**os.environ, **ENV}, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Readable outputs normalize only trailing newlines; actual exit is retained.
    if name == '06-m06-check.log' and result.returncode == 0:
        (OUT / name).write_text('Complete M06 checker passed (exit 0).\nFull result: runs/d01/m06-rerun.json\n')
    else:
        (OUT / name).write_text(result.stdout.rstrip('\n') + '\n')
    records.append({'argv': argv, 'cwd': str(ROOT), 'environment_overrides': ENV,
                    'started_at_utc': started, 'duration_seconds': round(time.monotonic()-t, 3),
                    'exit_code': result.returncode, 'expected_exit_code': expected,
                    'output': 'runs/d01/' + name})
    (OUT / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
    print(name, 'exit', result.returncode, result.stdout[-350:].strip(), flush=True)
    if result.returncode != expected:
        raise SystemExit(result.returncode or 1)

run([ADAPTER, '-m', 'pytest', 'tests/test_data_manifest.py', '-p', 'no:cacheprovider'], '01-focused-tests.log')
run([ADAPTER, 'scripts/check_d01.py', '--teacher-executable', '/private/tmp/Stockfish-sf_17.1/src/stockfish',
     '--output', str(OUT / 'manifest-check.json')], '02-manifest-check.log')
run([ADAPTER, 'scripts/check_d01.py', '--require-annotation-ready'], '03-annotation-gate.log', expected=1)
run([MODEL, '-m', 'pytest', '-p', 'no:cacheprovider', '-p', 'runs.m06.diagnostic_plugin',
     '--junitxml=' + str(OUT / 'pytest-model.xml')], '04-model-tests.log')
run([ADAPTER, '-m', 'pytest', '-p', 'no:cacheprovider',
     '--junitxml=' + str(OUT / 'pytest-adapter-only.xml')], '05-adapter-only-tests.log')
run([MODEL, 'scripts/check_m06.py', '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero',
     '--output', str(OUT / 'm06-rerun.json')], '06-m06-check.log')
run([MODEL, 'scripts/check_adapter.py'], '07-adapter-replay-model.json')
run([ADAPTER, 'scripts/check_adapter.py'], '08-adapter-replay-only.json')
run([ADAPTER, '-c', "import importlib.util,sys; assert importlib.util.find_spec('torch') is None; import chess_rl.data_manifest, chess_rl.adapter; assert 'torch' not in sys.modules; print('No installed/imported Torch')"], '09-import-boundary.log')
run([MODEL, 'scripts/check_upstream.py', '--maia3-checkout', '/private/tmp/rl-chess-f03-maia3',
     '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'], '10-pinned-sources.log')
run(['git', 'diff', '--check'], '11-whitespace.log')
