"""Fresh, append-only T02 integration checks in the two accepted environments."""
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
OUT = ROOT / 'runs/t02-integration'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
ARTIFACTS = ROOT.parent / 'external-artifacts/t02-integration-cpu-continuation-20261009'
BASE = 'b26cf38f04a09fd04283023795c0752753ce3106'
SOURCE = '27731e32d8aa64d852b439945f30e7238e87a824'

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)

def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())

def save(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def sources():
    paths = [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'tests').rglob('*.py')),
             *sorted((ROOT/'scripts').glob('*.py')), *sorted((ROOT/'runs/m06').glob('*.py')),
             Path(__file__), ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'.python-version']
    return {str(p.relative_to(ROOT)): identity(p)['sha256'] for p in paths}

def run(name, command):
    assert not (OUT / f'{name}.command.json').exists(), name
    env = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT/'src') + ':' + str(ROOT),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    before = sources()
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=os.environ | env, capture_output=True)
    elapsed = time.monotonic() - tick
    outputs = {}
    for label, raw in [('stdout', result.stdout), ('stderr', result.stderr)]:
        path = OUT / f'{name}.{label}.json'
        save(path, dict(encoding='base64', data=base64.b64encode(raw).decode(),
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        outputs[label] = dict(path=path.name, decoded_bytes=len(raw),
                              decoded_sha256=hashlib.sha256(raw).hexdigest(),
                              container_sha256=identity(path)['sha256'])
    save(OUT / f'{name}.command.json', dict(argv=command, cwd=str(ROOT), environment_overrides=env,
        started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=elapsed, exit_code=result.returncode, outputs=outputs,
        tested_source_sha256=before, tested_source_unchanged_during_command=before == sources()))
    print(f'{name}: exit {result.returncode}, {elapsed:.2f}s', flush=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr)
    assert before == sources(), name
    assert result.returncode == 0, name

def prepare():
    assert git('rev-parse', 'HEAD', cwd=PRIMARY).decode().strip() == BASE
    assert git('branch', '--show-current', cwd=PRIMARY) == b'main\n'
    status = git('status', '--porcelain=v1', '-uall', cwd=PRIMARY).decode()
    assert status == ' M coding-plan.md\n'
    assert not git('diff', '--cached', '--name-only', cwd=PRIMARY)
    patch = git('diff', '--binary', '--', 'coding-plan.md', cwd=PRIMARY)
    save(OUT/'primary-coding-plan-patch.json', dict(encoding='base64', data=base64.b64encode(patch).decode(),
                                                 sha256=hashlib.sha256(patch).hexdigest()))
    save(OUT/'setup.json', dict(base_commit=BASE, source_commit=SOURCE,
        merge_commit=git('rev-parse', 'HEAD').decode().strip(), branch='codex/t02-integration',
        isolated_checkout=str(ROOT), primary_checkout=str(PRIMARY), primary_status=status,
        primary_coding_plan_identity=identity(PRIMARY/'coding-plan.md'), conflicts=[],
        merge_strategy='exact reviewed commit merged with --no-ff; original history retained',
        instructions='No applicable AGENTS.md in checkouts or inspected ancestors; coding-plan.md, interface.md and documented M06 diagnostic instructions read',
        review_attribution='Independent AI review supplied by user: no blocking findings, 152 checkpoint/training tests and exact fresh-process CPU reproduction including hashes. No human review claimed; no new independent reviewer or subagent commissioned.',
        scope='T02 integration only; stop before T03, T04, pilot, P selection, self-play or RL'))
    preserve('preservation-before.json')

def preserve(name):
    setup = json.loads((OUT/'setup.json').read_text())
    assert git('rev-parse', 'HEAD', cwd=PRIMARY).decode().strip() == BASE
    assert git('status', '--porcelain=v1', '-uall', cwd=PRIMARY).decode() == setup['primary_status']
    assert identity(PRIMARY/'coding-plan.md') == setup['primary_coding_plan_identity']
    patch = base64.b64decode(json.loads((OUT/'primary-coding-plan-patch.json').read_text())['data'], validate=True)
    assert git('diff', '--binary', '--', 'coding-plan.md', cwd=PRIMARY) == patch
    accepted, reviewed, docs = {}, {}, {}
    for commit, collection in [(BASE, accepted), (SOURCE, reviewed)]:
        for line in git('ls-tree', '-r', commit).decode().splitlines():
            metadata, path = line.split('\t')
            raw = (ROOT/path).read_bytes()
            if path in ('README.md', 'coding-plan.md'):
                original = git('show', f'{commit}:{path}')
                normalized = raw
                if path == 'coding-plan.md':
                    normalized = normalized.replace(b'| [x] | T02 /', b'| [ ] | T02 /')
                assert normalized.startswith(original), (commit, path)
                docs[f'{commit}:{path}'] = dict(exact_original_prefix_preserved=True, bytes=len(original), sha256=hashlib.sha256(original).hexdigest())
                continue
            oid = metadata.split()[-1]
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == oid, (commit, path)
            collection[path] = identity(ROOT/path)
    external = {}
    def verify(path, expected):
        path = Path(path)
        expected = {k:expected[k] for k in ('bytes', 'sha256')}
        actual = identity(path)
        assert actual == expected, str(path)
        if str(path) in external:
            assert external[str(path)] == actual
        external[str(path)] = actual
    old = json.loads((ROOT/'runs/d08-t01-integration/preservation-final.json').read_text())
    for path, expected in old['accepted_external_inputs'].items():
        verify(path, expected)
    production = json.loads((ROOT/'runs/d08/results.json').read_text())
    assert {p.name for p in Path(production['bulk_directory']).iterdir()} == set(production['artifacts'])
    archive = ROOT.parent/'external-artifacts/d08-bcb208d-production'
    for path, expected in production['artifacts'].items():
        verify(expected['path'], expected); verify(archive/path, expected)
    t02 = json.loads((ROOT/'runs/t02/external-artifacts.json').read_text())
    for path, expected in t02['files'].items():
        verify(path, expected)
    for path, expected in t02['source_inputs_rehashed_unchanged'].items():
        verify(Path(path) if Path(path).is_absolute() else ROOT/path, expected)
    for path, expected in old['original_d08_and_durable_copy_artifacts'].items():
        verify(archive/path, expected)
    assert git('merge-base', '--is-ancestor', SOURCE, 'HEAD') == b''
    assert git('merge-base', '--is-ancestor', BASE, 'HEAD') == b''
    save(OUT/name, dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE,
        source_commit=SOURCE, accepted_files=accepted, reviewed_files=reviewed, appended_documents=docs,
        external_files=external, protected_heldout_registries_preserved=True,
        primary_main_and_exact_unrelated_patch_preserved=True))
    print(f'{name}: {len(accepted)} accepted files, {len(reviewed)} reviewed files, {len(external)} external files preserved', flush=True)

def checks(kind):
    python = str(PRIMARY/('.venv' if kind == 'model' else '.venv-acceptance')/'bin/python')
    env_code = "import importlib.metadata as m,importlib.util,platform,json,sys,chess_rl; v={n:m.version(n) for n in ['numpy','python-chess','chess','pytest']}; assert platform.python_version()=='3.12.12'; assert v=={'numpy':'2.5.3','python-chess':'1.999','chess':'1.11.2','pytest':'9.1.1'}; print(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),executable=sys.executable,package=chess_rl.__file__,versions=v,torch_available=importlib.util.find_spec('torch') is not None),indent=2))"
    if kind == 'model':
        env_code += "; assert m.version('torch')=='2.8.0'; print('torch',m.version('torch'))"
    else:
        env_code += "; assert importlib.util.find_spec('torch') is None"
    run(f'environment-{kind}', [python, '-B', '-c', env_code])
    run(f'dependencies-{kind}', ['/Users/arvindsrinivass/.local/bin/uv', '--offline', '--cache-dir',
        '/private/tmp/rl-chess-t02-integration-uv-cache', 'pip', 'check', '--python', python])
    pytest = [python, '-B', '-m', 'pytest', '--import-mode=importlib', '-p', 'no:cacheprovider']
    plugin = ['-p', 'runs.m06.diagnostic_plugin'] if kind == 'model' else []
    run(f'regressions-{kind}', pytest + plugin + [f'--junitxml=runs/t02-integration/regressions-{kind}.xml'])
    probes = [f'runs/{n}/test_review_cases.py' for n in ['d02-s02-review', 'd03-s03-review', 'd04-s04-review', 'd05-s05-review']]
    run(f'accepted-review-probes-{kind}', pytest + probes + [f'--junitxml=runs/t02-integration/accepted-review-probes-{kind}.xml'])
    run(f'frozen-adapter-{kind}', [python, '-B', 'scripts/check_adapter.py'])
    if kind == 'model':
        run('checkpoint-training-tests', pytest + ['tests/test_checkpoints.py','tests/test_training.py',
            '--junitxml=runs/t02-integration/checkpoint-training-tests.xml'])
        run('m06-numerical', [python, '-B', 'scripts/check_m06.py'])
        run('fresh-process-continuation', [python, '-B', 'scripts/check_t02.py', '--directory', str(ARTIFACTS)])
    else:
        run('pinned-upstream-sources', [python, '-B', 'scripts/check_upstream.py', '--maia3-checkout',
            '/private/tmp/rl-chess-f03-maia3', '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'])

if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    elif sys.argv[1] == 'preserve': preserve(sys.argv[2])
    else: checks(sys.argv[1])
