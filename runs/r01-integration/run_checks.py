"""Fresh integration evidence, separate from historical implementation receipts."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
MODEL = PRIMARY / '.venv/bin/python'
ADAPTER = PRIMARY / '.venv-acceptance/bin/python'
CHECKPOINT = Path('/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/t03-tiny-overfit-20261009/final.pt')

def identity(p):
    p = Path(p)
    with p.open('rb') as f:
        return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())

def save(p, value):
    with Path(p).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)

def sources():
    paths = [p for d in ('src', 'tests', 'scripts') for p in (ROOT / d).rglob('*.py')]
    paths += [OUT / 'run_checks.py', OUT / 'supplied-review/test_r01_review.py', ROOT / 'runs/m06/diagnostic_plugin.py']
    return {str(p.relative_to(ROOT)): identity(p) for p in sorted(paths)}

def run(name, args, python=MODEL):
    command = [str(python), '-B', *map(str, args)]
    overrides = dict(PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    tested = sources()
    tick = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, env=os.environ | overrides, capture_output=True)
    for label, raw in (('stdout', result.stdout), ('stderr', result.stderr)):
        save(OUT / f'{name}.{label}.json', dict(encoding='base64', data=base64.b64encode(raw).decode(),
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    save(OUT / f'{name}.command.json', dict(argv=command, cwd=str(ROOT), environment_overrides=overrides,
        started_at_utc=started, duration_seconds=time.monotonic()-tick, exit_code=result.returncode,
        tested_head=git('rev-parse', 'HEAD').decode().strip(), tested_sources=tested))
    assert sources() == tested
    print(name, 'exit', result.returncode, 'seconds', round(time.monotonic()-tick, 3), flush=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
    assert result.returncode == 0, name
    return result.stdout

def pytest(name, paths=(), python=MODEL, plugin=False):
    args = ['-m', 'pytest', *paths, '--import-mode=importlib', '-p', 'no:cacheprovider',
        '--basetemp=' + str(OUT / 'tmp' / name), '--junitxml=' + str(OUT / (name + '.xml'))]
    if plugin:
        args += ['-p', 'runs.m06.diagnostic_plugin']
    run(name, args, python)

def main():
    mode = sys.argv[1]
    if mode in ('model', 'adapter'):
        python = MODEL if mode == 'model' else ADAPTER
        run('environment-' + mode, ['-c', '''import sys,importlib.metadata as m,importlib.util,json
versions={n:m.version(n) for n in ('numpy','python-chess','chess','pytest')}
assert sys.version_info[:3]==(3,12,12)
assert versions=={'numpy':'2.5.3','python-chess':'1.999','chess':'1.11.2','pytest':'9.1.1'}
if importlib.util.find_spec('torch'):
 import torch
 assert torch.__version__=='2.8.0'
 versions['torch']=torch.__version__
 print(json.dumps({'python':sys.version,'versions':versions,'cuda_available':torch.cuda.is_available()},indent=2))
else:
 print(json.dumps({'python':sys.version,'versions':versions,'torch_present':False},indent=2))
'''], python)
        pytest('regression-' + mode, python=python, plugin=(mode == 'model'))
        pytest('accepted-probes-' + mode, [f'runs/{n}/test_review_cases.py' for n in
            ('d02-s02-review', 'd03-s03-review', 'd04-s04-review', 'd05-s05-review')], python)
        fresh = run('adapter-replay-' + mode, ['scripts/check_adapter.py'], python)
        prior = json.loads((ROOT / f'runs/t03-t04-integration/frozen-adapter-{mode}.stdout.json').read_text())
        assert fresh == base64.b64decode(prior['data'])
        if mode == 'model':
            pytest('focused', ['tests/test_selfplay.py', 'tests/test_search_runner.py', 'tests/test_game_runner.py',
                'tests/test_real_evaluator.py', 'tests/test_checkpoints.py'])
            pytest('independent-r01', ['runs/r01-integration/supplied-review/test_r01_review.py'])
            run('m06-numerical', ['scripts/check_m06.py'])
    elif mode == 'real':
        run('real-freeze', ['scripts/check_r01.py', 'freeze', '--checkpoint', CHECKPOINT, '--output', OUT / 'real'])
        run('real-fixtures', ['scripts/check_r01.py', 'fixtures', '--checkpoint', CHECKPOINT, '--output', OUT / 'real'])
    else:
        raise ValueError(mode)

if __name__ == '__main__':
    main()
