"""Preserve all reviewed bytes, raw evidence, external data and primary main."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '9565093abe40a7807e3472c1101f5f28c14b7afe'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
APPENDED = {'README.md', 'coding-plan.md'}


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


preserved, appended = {}, {}
for line in git('ls-tree', '-r', BASE).decode().splitlines():
    meta, name = line.split('\t', 1)
    mode, kind, oid = meta.split()
    assert kind == 'blob' and mode in ('100644', '100755')
    path = ROOT / name
    content = path.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if name in APPENDED:
        previous = git('show', BASE + ':' + name)
        assert content.startswith(previous) and len(content) > len(previous), name
        appended[name] = dict(accepted_prefix_bytes=len(previous), current=identity(path))
    else:
        assert actual == oid, name
        preserved[name] = dict(git_blob=oid, **identity(path))
assert git('rev-parse', 'main', cwd=PRIMARY).decode().strip() == BASE
assert git('symbolic-ref', '--short', 'HEAD', cwd=PRIMARY).decode().strip() == 'main'
assert git('status', '--porcelain', cwd=PRIMARY) == b''
expected = json.loads((ROOT/'runs/d06-d07-e01-integration/preservation.json').read_text())['original_external_identities']
external = {}
for name, accepted in expected.items():
    current = identity(Path(name))
    assert current == accepted, name
    external[name] = current
reports = sorted((ROOT/'runs/t01').glob('*.command.json'))
assert reports
commands = {}
for path in reports:
    record = json.loads(path.read_text())
    assert record['exit_code'] == 0 and record['tested_source_unchanged_during_command'], path
    for name, expected_hash in record['tested_source_sha256'].items():
        assert identity(ROOT/name)['sha256'] == expected_hash, name
    for value in record['outputs'].values():
        output_path = ROOT/'runs/t01'/value['path']
        if value.get('encoding') == 'utf8-json-string':
            raw = json.loads(output_path.read_text()).encode('utf-8')
            assert len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256']
            assert identity(output_path) == dict(bytes=value['container_bytes'], sha256=value['container_sha256'])
        else:
            assert identity(output_path) == {k: value[k] for k in ('bytes', 'sha256')}
    commands[path.name] = identity(path)
model_replay = json.loads((ROOT/'runs/t01/frozen-model.stdout.txt').read_text())
adapter_replay = json.loads((ROOT/'runs/t01/frozen-adapter.stdout.txt').read_text())
accepted_replay = json.loads((ROOT/'runs/d06-d07-e01-integration/frozen-adapter-model.stdout.txt').read_text())
assert model_replay == adapter_replay == accepted_replay and len(model_replay) == 13
report = dict(base_commit=BASE, accepted_files=len(preserved) + len(appended),
    byte_identical_files=preserved, appended_documents=appended, external_identities=external,
    primary_main_clean_and_unchanged=True, all_recorded_checks_pass=True,
    source_and_raw_output_identities_verified=True, accepted_13_fixture_replay_exact=True, command_identities=commands)
(ROOT/'runs/t01/preservation.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('byte_identical_files', 'external_identities', 'command_identities')}, indent=2))
