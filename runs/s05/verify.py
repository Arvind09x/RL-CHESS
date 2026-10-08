"""Save real commands/results in both existing pinned environments."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
BASE = 'ad8438acfbd859f734daaebc6115afdab0367b65'


def main():
    focused_only = '--focused-only' in sys.argv
    out = ROOT / 'runs/s05' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir()
    env = dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1')
    commands, counts, replays, games = [], {}, [], []

    def run(name, argv):
        begin = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run([str(a) for a in argv], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        for suffix, content in [('stdout', result.stdout), ('stderr', result.stderr)]:
            (out / f'{name}.{suffix}.txt').write_text(content)
        commands.append(dict(name=name, argv=[str(a) for a in argv], cwd=str(ROOT),
                             environment={k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')},
                             started_utc=started, elapsed_seconds=time.monotonic() - begin,
                             exit_code=result.returncode,
                             stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                             stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest()))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(f'{name}: exit {result.returncode}', flush=True)
        if result.returncode:
            print(result.stdout + result.stderr, flush=True)
            raise RuntimeError(f'Failed {name}; evidence: {out}')
        return result.stdout

    for label, venv in [('adapter', '.venv-acceptance'), ('model', '.venv')]:
        python = SOURCE / venv / 'bin/python'
        code = (
            'import importlib.metadata as m, importlib.util as u, json, platform, sys; '
            'print(json.dumps(dict(python=sys.version, executable=sys.executable, '
            'platform=platform.platform(), machine=platform.machine(), '
            "versions={n:m.version(n) for n in ['numpy','python-chess','chess','pytest']}, "
            "torch_available=u.find_spec('torch') is not None), indent=2))"
        )
        run(f'environment-{label}', [python, '-B', '-c', code])
        checks = [('focused', ['tests/test_game_runner.py', 'tests/test_search_runner.py',
                               'tests/test_backup.py', 'tests/test_puct.py', 'tests/test_evaluator.py'])]
        if not focused_only:
            checks += [('regressions', []), ('accepted-review-probes', [
                'runs/d02-s02-review/test_review_cases.py',
                'runs/d03-s03-review/test_review_cases.py',
                'runs/d04-s04-review/test_review_cases.py'])]
        for name, tests in checks:
            key = f'{name}-{label}'
            xml = out / f'{key}.xml'
            plugin = ['-p', 'runs.m06.diagnostic_plugin'] if label == 'model' and not tests else []
            run(key, [python, '-B', '-m', 'pytest', *tests, *plugin, '--import-mode=importlib',
                      '-p', 'no:cacheprovider', f'--junitxml={xml}'])
            counts[key] = dict(ET.parse(xml).getroot().find('testsuite').attrib)
        if not focused_only:
            replays.append(json.loads(run(f'adapter-replay-{label}', [python, '-B', 'scripts/check_adapter.py'])))
            game_dir = out / f'games-{label}'
            run(f'smoke-{label}', [python, '-B', 'runs/s05/smoke.py', game_dir])
            games.append((game_dir / 'games.jsonl').read_bytes())
            for replay_label, replay_env in [('adapter', '.venv-acceptance'), ('model', '.venv')]:
                run(f'game-replay-{label}-in-{replay_label}', [SOURCE / replay_env / 'bin/python',
                    '-B', 'runs/s05/replay_games.py', game_dir / 'games.jsonl',
                    out / f'game-replay-{label}-in-{replay_label}.json'])
    preserved = {}
    if not focused_only:
        assert replays[0] == replays[1] and len(replays[0]) == 13
        assert games[0] == games[1], 'Deterministic smoke records differ across environments'
        tracked = run('baseline-files', ['git', 'ls-tree', '-r', '--name-only', BASE]).splitlines()
        for relative in tracked:
            original = subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT)
            current = (ROOT / relative).read_bytes()
            assert original == current, f'Accepted file changed: {relative}'
            preserved[relative] = hashlib.sha256(current).hexdigest()
        (out / 'preservation.json').write_text(json.dumps(dict(
            base_commit=BASE, byte_identical_files=preserved), indent=2) + '\n')
        run('diff-check', ['git', 'diff', '--check'])
        run('branch', ['git', 'status', '--short', '--branch'])
        run('base-head', ['git', 'rev-parse', 'HEAD'])
        assert run('primary-head', ['git', '-C', str(SOURCE), 'rev-parse', 'main']).strip() == BASE
    summary = dict(base_commit=BASE, branch='codex/s05-search-smoke', checkout=str(ROOT),
                   checks=counts, focused_only=focused_only,
                   frozen_adapter_replays_equal=not focused_only,
                   deterministic_game_records_equal=not focused_only,
                   byte_identical_accepted_base_files=len(preserved),
                   independent_review='pending', gate_s='unaccepted')
    if not focused_only:
        summary['smoke'] = json.loads((out / 'games-adapter/summary.json').read_text())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Evidence: {out}', flush=True)


if __name__ == '__main__':
    main()
