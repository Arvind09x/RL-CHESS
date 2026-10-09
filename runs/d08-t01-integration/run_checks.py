"""Fresh combined checks; keep exact output bytes in lossless JSON containers."""
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/d08-t01-integration'
PYTHON_ROOT = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
ENV = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
PROBES = [f'runs/{name}/test_review_cases.py' for name in
          ('d02-s02-review', 'd03-s03-review', 'd04-s04-review', 'd05-s05-review')]


def sources():
    names = subprocess.check_output(['git', 'ls-files'], cwd=ROOT).decode().splitlines()
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names
            if n.endswith('.py') or n in ('pyproject.toml', 'uv.lock', '.python-version', 'runs/t01/config.json')}


def run(name, command):
    before = sources()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=os.environ | ENV, capture_output=True)
    elapsed = time.monotonic() - clock
    outputs = {}
    for label, raw in (('stdout', result.stdout), ('stderr', result.stderr)):
        path = OUT / f'{name}.{label}.json'
        # Base64 is reversible even if a diagnostic is not valid UTF-8; no trimming.
        container = dict(encoding='base64', data=base64.b64encode(raw).decode(),
                         bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        path.write_text(json.dumps(container, indent=2, sort_keys=True) + '\n')
        outputs[label] = dict(path=path.name, decoded_bytes=len(raw), decoded_sha256=container['sha256'],
                             container_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record = dict(argv=command, cwd=str(ROOT), environment_overrides=ENV, started_at_utc=started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), duration_seconds=elapsed,
                  exit_code=result.returncode, outputs=outputs, source_sha256=before,
                  source_unchanged_during_command=before == sources())
    (OUT / f'{name}.command.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(name=name, exit_code=result.returncode, seconds=elapsed)), flush=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
    assert before == sources()
    assert result.returncode == 0, name


def suite(kind):
    python = str(PYTHON_ROOT / ('.venv' if kind == 'model' else '.venv-acceptance') / 'bin/python')
    prefix = [python, '-B']
    inspection = "import importlib.metadata as m,importlib.util,platform,json,sys; print(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),executable=sys.executable,versions={n:m.version(n) for n in ['numpy','python-chess','chess','pytest']},torch_available=importlib.util.find_spec('torch') is not None),indent=2))"
    run('environment-' + kind, prefix + ['-c', inspection])
    pytest = prefix + ['-m', 'pytest', '--import-mode=importlib', '-p', 'no:cacheprovider']
    plugin = ['-p', 'runs.m06.diagnostic_plugin'] if kind == 'model' else []
    run('regressions-' + kind, pytest + plugin + [f'--junitxml={OUT}/regressions-{kind}.xml'])
    run('accepted-review-probes-' + kind, pytest + PROBES + [f'--junitxml={OUT}/accepted-review-probes-{kind}.xml'])
    run('frozen-adapter-' + kind, prefix + ['scripts/check_adapter.py'])
    run('s05-replay-' + kind, prefix + ['runs/s05/replay_games.py',
        str(ROOT / 'runs/s05/20261008T132541185409Z/games-adapter/games.jsonl'), str(OUT / f's05-replay-{kind}.json')])
    for bundle in ('adapter', 'model'):
        run(f'e01-{bundle}-in-{kind}', prefix + ['scripts/replay_e01.py',
            str(ROOT / f'runs/e01/20261008T150821584550Z/matches-{bundle}/matches.jsonl'),
            '--output', str(OUT / f'e01-{bundle}-in-{kind}.json')])
    if kind == 'model':
        run('m06-numerical', prefix + ['scripts/check_m06.py'])
        run('t01-real-batch', prefix + ['scripts/check_t01.py', '--output', str(OUT / 't01-real-batch.json')])
    else:
        bulk = str(ROOT.parent / 'external-artifacts/d08-bcb208d-production')
        reports = ROOT.parent / 'external-artifacts/d08-t01-integration-audits'
        reports.mkdir(exist_ok=False)
        run('d08-independent-replay', prefix + ['scripts/audit_d08.py', bulk, '--report', str(reports / 'd08-replay.json')])
        run('d08-loader-verification', prefix + ['scripts/check_d08_loader.py', bulk, '--report', str(reports / 'd08-loader.json')])
        run('manifest-teacher-gate', prefix + ['scripts/check_d01.py', '--manifest',
            'data/manifests/elite-2025-01-d04-v1.json', '--require-annotation-ready',
            '--teacher-executable', '/private/tmp/Stockfish-sf_17.1/src/stockfish'])
        run('pinned-upstream-sources', prefix + ['scripts/check_upstream.py', '--maia3-checkout',
            '/private/tmp/rl-chess-f03-maia3', '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'])


if __name__ == '__main__':
    suite(sys.argv[1])
