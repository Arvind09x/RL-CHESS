"""Record exact bounded E01 commands and regressions in both pinned envs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
BASE = '4184387c7e4b67559355d6b212a2a300c86c5ce6'


def main():
    out = ROOT / 'runs/e01' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir()
    env = dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1')
    commands, counts, adapter_reports = [], {}, []
    def run(name, argv):
        begin = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run([str(a) for a in argv], cwd=ROOT, env=env, capture_output=True, text=True)
        for suffix, content in [('stdout', result.stdout), ('stderr', result.stderr)]:
            (out / f'{name}.{suffix}.txt').write_text(content)
        commands.append(dict(name=name, argv=[str(a) for a in argv], cwd=str(ROOT),
            environment={k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')}, started_utc=started,
            elapsed_seconds=time.monotonic() - begin, exit_code=result.returncode,
            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest()))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(f'{name}: exit {result.returncode}', flush=True)
        if result.returncode:
            print(result.stdout + result.stderr, flush=True)
            raise RuntimeError(f'Failed {name}: {out}')
        return result.stdout
    for label, venv in [('adapter', '.venv-acceptance'), ('model', '.venv')]:
        python = SOURCE / venv / 'bin/python'
        run(f'environment-{label}', [python, '-B', '-c',
            "import importlib.metadata as m, platform, json, sys; print(json.dumps(dict(python=sys.version, "
            "executable=sys.executable, platform=platform.platform(), versions={n:m.version(n) for n in "
            "['numpy','python-chess','chess','pytest']}),indent=2))"])
        for name, tests in [('focused', ['tests/test_matches.py', 'tests/test_game_runner.py', 'tests/test_search_runner.py']),
                            ('regressions', []), ('accepted-review-probes', [
                            'runs/d02-s02-review/test_review_cases.py', 'runs/d03-s03-review/test_review_cases.py',
                            'runs/d04-s04-review/test_review_cases.py', 'runs/d05-s05-review/test_review_cases.py'])]:
            key = f'{name}-{label}'
            xml = out / f'{key}.xml'
            plugin = ['-p', 'runs.m06.diagnostic_plugin'] if label == 'model' and not tests else []
            run(key, [python, '-B', '-m', 'pytest', *tests, *plugin, '--import-mode=importlib',
                      '-p', 'no:cacheprovider', f'--junitxml={xml}'])
            counts[key] = dict(ET.parse(xml).getroot().find('testsuite').attrib)
        adapter_reports.append(json.loads(run(f'adapter-replay-{label}', [python, '-B', 'scripts/check_adapter.py'])))
        run(f's05-accepted-replay-{label}', [python, '-B', 'runs/s05/replay_games.py',
            ROOT / 'runs/s05/20261008T132541185409Z/games-adapter/games.jsonl', out / f's05-replay-{label}.json'])
        game_dir = out / f'matches-{label}'
        run(f'smoke-{label}', [python, '-B', 'runs/e01/smoke.py', game_dir])
        for target, target_venv in [('adapter', '.venv-acceptance'), ('model', '.venv')]:
            run(f'e01-replay-{label}-in-{target}', [SOURCE / target_venv / 'bin/python', '-B',
                'scripts/replay_e01.py', game_dir / 'matches.jsonl', '--output', out / f'e01-replay-{label}-in-{target}.json'])
    assert adapter_reports[0] == adapter_reports[1] and len(adapter_reports[0]) == 13
    # Engine timing/node accounting are measured, so compare deterministic non-engine fixtures exactly.
    def records(label):
        return [json.loads(line) for line in (out / f'matches-{label}/matches.jsonl').read_text().splitlines()]
    a, b = records('adapter'), records('model')
    assert len(a) == len(b) == 24
    assert [r for r in a if r['configuration']['right_player']['kind'] != 'stockfish'] == [
        r for r in b if r['configuration']['right_player']['kind'] != 'stockfish']
    tracked = run('baseline-files', ['git', 'ls-tree', '-r', '--name-only', BASE]).splitlines()
    preserved = {}
    for relative in tracked:
        original = subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        assert original == current, f'Accepted file changed: {relative}'
        preserved[relative] = hashlib.sha256(current).hexdigest()
    (out / 'preservation.json').write_text(json.dumps(dict(base_commit=BASE, byte_identical_files=preserved), indent=2) + '\n')
    assert run('primary-main', ['git', '-C', str(SOURCE), 'rev-parse', 'main']).strip() == BASE
    assert run('checkout-head', ['git', 'rev-parse', 'HEAD']).strip() == BASE
    run('diff-check', ['git', 'diff', '--check'])
    summary = dict(base_commit=BASE, branch='codex/e01-match-protocol', checkout=str(ROOT), checks=counts,
        byte_identical_accepted_files=len(preserved), adapter_reports_equal=True,
        deterministic_nonengine_records_equal=True, matches_per_environment=24, games_per_environment=48,
        independent_review='pending', real_checkpoints=False, training=False, t04=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Evidence: {out}', flush=True)


if __name__ == '__main__': main()
