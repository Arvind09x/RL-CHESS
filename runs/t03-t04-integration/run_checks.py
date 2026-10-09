"""Fresh, append-only T03/T04 integration checks in the two accepted environments."""
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
OUT = ROOT / 'runs/t03-t04-integration'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
CHECKPOINT = ROOT.parent / 'external-artifacts/t03-tiny-overfit-20261009/final.pt'
BASE = '4fa48d4afef91a3bea125fb5f9f85b4d5159fc28'
SOURCES = ['1a9b00dd799af7578476624e4471073eaa3450eb', '197bf821bfe13f2fab1dc5a8a54da2d8d13779e8']

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

def primary_snapshot():
    return dict(head=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip(),
        branch=git('branch','--show-current',cwd=PRIMARY).decode().strip(),
        status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),
        patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        index_patch_base64=base64.b64encode(git('diff','--cached','--binary',cwd=PRIMARY)).decode(),
        plan=identity(PRIMARY/'coding-plan.md'))

def external_inventory():
    indexed=json.loads((ROOT/'runs/t02-integration/preservation-after.json').read_text())['external_files']
    expected=json.loads((ROOT/'runs/t02-integration/workspace-external-after.json').read_text())['files']
    expected.update(json.loads((ROOT/'runs/t03/external-artifacts.json').read_text()))
    expected.update(indexed)
    expected.update(json.loads((ROOT/'runs/t03/frozen-plan.json').read_text())['inputs'])
    for name, target in expected.items():
        actual=identity(Path(name) if Path(name).is_absolute() else ROOT/name)
        assert actual == {k:target[k] for k in ('bytes','sha256')}, name
    paths={str(p) for p in (ROOT.parent/'external-artifacts').rglob('*') if p.is_file()}
    paths.update(n for n in expected if Path(n).is_absolute())
    return {n:identity(Path(n)) for n in sorted(paths)}

def check_trees():
    reports={}
    docs=['README.md','coding-plan.md']
    base_docs={n:git('show',BASE+':'+n) for n in docs}
    for commit in [BASE,*SOURCES]:
        count=0
        for row in git('ls-tree','-r',commit).decode().splitlines():
            metadata,name=row.split('\t'); raw=(ROOT/name).read_bytes()
            if name in docs: continue
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==metadata.split()[-1], (commit,name)
            count+=1
        reports[commit]=count
        assert not git('merge-base','--is-ancestor',commit,'HEAD')
    for n in docs:
        combined=base_docs[n]+b''.join(git('show',c+':'+n)[len(base_docs[n]):] for c in SOURCES)
        raw=(ROOT/n).read_bytes()
        if n=='coding-plan.md':
            raw=raw.replace(b'| [x] | T03 /',b'| [ ] | T03 /').replace(b'| [x] | T04 /',b'| [ ] | T04 /')
        assert raw.startswith(combined), n
    original=ROOT.parent/'t03-tiny-overfit'
    plan=json.loads((ROOT/'runs/t03/frozen-plan.json').read_text())
    for n,expected in plan['tested_sources'].items():
        assert identity(original/n)==expected, n
        assert identity(ROOT/n)==expected, n
    for directory,commit in [('t03-tiny-overfit',SOURCES[0]),('t04-real-evaluator',SOURCES[1])]:
        assert git('rev-parse','HEAD',cwd=ROOT.parent/directory).decode().strip()==commit
        assert git('status','--porcelain',cwd=ROOT.parent/directory)==b''
    return dict(exact_non_document_files_per_commit=reports,
        both_documentation_completion_records_preserved=True,
        original_frozen_source_set_and_bytes_unchanged=True,
        original_reviewed_checkouts_clean_and_unchanged=True,
        original_freeze_not_extended_to_t04_sources=True)

def preserve(name):
    setup=json.loads((OUT/'setup.json').read_text())
    assert primary_snapshot()==setup['primary']
    external=external_inventory()
    if name!='preservation-before.json':
        assert external==json.loads((OUT/'preservation-before.json').read_text())['external_files']
    save(OUT/name,dict(trees=check_trees(),external_files=external,
        primary_main_and_exact_uncommitted_patch_unchanged=True,
        verified_at_utc=datetime.now(timezone.utc).isoformat()))
    print(name, len(external), 'external files verified',flush=True)

def prepare():
    primary=primary_snapshot()
    assert primary['head']==BASE and primary['branch']=='main' and primary['status']==' M coding-plan.md\n'
    assert primary['index_patch_base64']==''
    save(OUT/'setup.json',dict(base_commit=BASE,reviewed_source_commits=SOURCES,
        merge_commit=git('rev-parse','HEAD').decode().strip(), primary=primary,
        branch='codex/t03-t04-integration',isolated_checkout=str(ROOT),
        conflicts=['README.md','coding-plan.md'],resolution='base prefix followed by exact T03 append and exact T04 append',
        review_type='Supplied independent AI review; no human review and no newly commissioned review',
        supplied_review=dict(no_blocking_findings_within_documented_cpu_scope=True,t03_tests=20,t04_tests=34,
            exact_actual_t03_model_hashes_and_metrics_verified=True,step_26_first_passing_verified=True,
            actual_checkpoint_exercised_through_t04_and_bounded_search=True),
        instructions='No applicable AGENTS.md found; interface.md, coding-plan.md, T02 accepted integration workflow and T03/T04 scope inspected',
        scope='Acceptance and integration only; no original overfit rerun, R01, GPU or training campaign'))
    preserve('preservation-before.json')

def checks(kind):
    python = str(PRIMARY/('.venv' if kind == 'model' else '.venv-acceptance')/'bin/python')
    env_code = "import importlib.metadata as m,importlib.util,platform,json,sys,chess_rl; v={n:m.version(n) for n in ['numpy','python-chess','chess','pytest']}; assert platform.python_version()=='3.12.12'; assert v=={'numpy':'2.5.3','python-chess':'1.999','chess':'1.11.2','pytest':'9.1.1'}; print(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),executable=sys.executable,package=chess_rl.__file__,versions=v,torch_available=importlib.util.find_spec('torch') is not None),indent=2))"
    if kind == 'model':
        env_code += "; assert m.version('torch')=='2.8.0'; print('torch',m.version('torch'))"
    else:
        env_code += "; assert importlib.util.find_spec('torch') is None"
    run(f'environment-{kind}', [python, '-B', '-c', env_code])
    run(f'dependencies-{kind}', ['/Users/arvindsrinivass/.local/bin/uv', '--offline', '--cache-dir',
        '/private/tmp/rl-chess-t03-t04-integration-uv-cache', 'pip', 'check', '--python', python])
    pytest = [python, '-B', '-m', 'pytest', '--import-mode=importlib', '-p', 'no:cacheprovider']
    plugin = ['-p', 'runs.m06.diagnostic_plugin'] if kind == 'model' else []
    run(f'regressions-{kind}', pytest + plugin + [f'--junitxml=runs/t03-t04-integration/regressions-{kind}.xml'])
    probes = [f'runs/{n}/test_review_cases.py' for n in ['d02-s02-review', 'd03-s03-review', 'd04-s04-review', 'd05-s05-review']]
    run(f'accepted-review-probes-{kind}', pytest + probes + [f'--junitxml=runs/t03-t04-integration/accepted-review-probes-{kind}.xml'])
    run(f'frozen-adapter-{kind}', [python, '-B', 'scripts/check_adapter.py'])
    if kind == 'model':
        run('checkpoint-training-tests', pytest + ['tests/test_checkpoints.py','tests/test_training.py','tests/test_tiny_overfit.py','tests/test_real_evaluator.py',
            '--junitxml=runs/t03-t04-integration/checkpoint-training-tests.xml'])
        run('m06-numerical', [python, '-B', 'scripts/check_m06.py'])
        run('t03-checkpoint-identity-metrics', [python, '-B', str(OUT/'verify_t03.py')])
        run('t03-through-t04', [python, '-B', 'scripts/check_t04.py', '--checkpoint', str(CHECKPOINT), '--output', str(OUT/'t03-through-t04.json')])
    else:
        run('pinned-upstream-sources', [python, '-B', 'scripts/check_upstream.py', '--maia3-checkout',
            '/private/tmp/rl-chess-f03-maia3', '--lczero-checkout', '/private/tmp/rl-chess-f03-lczero'])

if __name__ == '__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='preserve': preserve(sys.argv[2])
    else: checks(sys.argv[1])
