"""Independent review commands; all outputs live outside historical R02 evidence."""
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
ENV = dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(path, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(path), *args], env=ENV)

def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def preservation():
    historical = json.loads((ROOT / 'runs/r02/preservation-after.json').read_text())
    r01 = ROOT.parent / 'r01-self-play'
    return dict(
        review_head=git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
        review_tracked={p: sha(ROOT / p) for p in git(ROOT, 'ls-files').decode().splitlines()},
        r01_head=git(r01, 'rev-parse', 'HEAD').decode().strip(),
        r01_tracked={p: sha(r01 / p) for p in historical['r01_tracked_files']},
        external_files={p: dict(bytes=Path(p).stat().st_size, sha256=sha(p)) for p in historical['external_files']},
        primary=dict(head=git(PRIMARY, 'rev-parse', 'HEAD').decode().strip(),
            tracked_files={p: sha(PRIMARY / p) for p in historical['primary']['tracked_files']},
            patch_sha256=hashlib.sha256(git(PRIMARY, 'diff', '--binary')).hexdigest(),
            staged_patch_sha256=hashlib.sha256(git(PRIMARY, 'diff', '--cached', '--binary')).hexdigest(),
            index_sha256=sha(PRIMARY / '.git/index')))

def run(name, command):
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True)
    save(name + '.command.json', dict(argv=list(map(str, command)), cwd=str(ROOT),
        environment={k: ENV[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'GIT_OPTIONAL_LOCKS')},
        duration_seconds=time.monotonic()-start, exit_code=result.returncode))
    (OUT / (name + '.stdout.txt')).write_text(result.stdout)
    (OUT / (name + '.stderr.txt')).write_text(result.stderr)
    print(result.stdout[-4000:], flush=True)
    if result.returncode:
        print(result.stderr[-2000:], flush=True)
        raise SystemExit(result.returncode)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode in ('before', 'after'):
        actual = preservation()
        save('preservation-' + mode + '.json', actual)
        if mode == 'after':
            assert actual == json.loads((OUT / 'preservation-before.json').read_text())
        else:
            old = json.loads((ROOT / 'runs/r02/preservation-after.json').read_text())
            assert actual['r01_tracked'] == old['r01_tracked_files']
            assert actual['external_files'] == old['external_files']
            assert actual['primary']['tracked_files'] == old['primary']['tracked_files']
            assert actual['primary']['patch_sha256'] == old['primary']['patch_sha256']
            assert actual['primary']['index_sha256'] == old['primary']['index_sha256']
        print('All tracked, accepted checkout, primary patch/index and external artifacts preserved.')
    else:
        python = MODEL if mode in ('model', 'focused') or mode.startswith('probes') else ADAPTER
        command = [str(python), '-B', '-m', 'pytest', '--import-mode=importlib', '-p', 'no:cacheprovider',
                   '--junitxml=' + str(OUT / (mode + '.xml'))]
        if mode == 'model':
            command += ['-p', 'runs.m06.diagnostic_plugin']
        if mode == 'focused':
            command += ['tests/test_replay.py']
        if mode.startswith('probes'):
            command += [str(OUT / 'test_independent.py')]
            if mode != 'probes':
                artifacts = OUT / (mode + '-artifacts')
                artifacts.mkdir()
                ENV['REVIEW_OUTPUT'] = str(artifacts)
        run(mode, command)
