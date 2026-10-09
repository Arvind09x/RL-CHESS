"""Verify accepted Git bytes, historical evidence, external inputs and primary edits."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
BASE = '9565093abe40a7807e3472c1101f5f28c14b7afe'
COMMITS = {'D08': 'bcb208d73271451b05d1409d4bf340aaad82eabf',
           'T01': 'f669fc8b4de397a31768321b9105e255b0ff0e0e'}
OUT = ROOT / 'runs/d08-t01-integration'
ARCHIVE = ROOT.parent / 'external-artifacts/d08-bcb208d-production'


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def save(path, obj):
    with path.open('x') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def verify(destination):
    base_files, imported = {}, {}
    for line in git('ls-tree', '-r', BASE).decode().splitlines():
        metadata, name = line.split('\t')
        oid = metadata.split()[-1]
        raw = (ROOT / name).read_bytes()
        if name in ('README.md', 'coding-plan.md', 'src/chess_rl/stockfish_worker.py'):
            continue
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == oid, name
        base_files[name] = identity(ROOT / name)
    original_docs = {}
    for name in ('README.md', 'coding-plan.md'):
        original = git('show', f'{BASE}:{name}')
        current = (ROOT / name).read_bytes()
        # Accepted milestone checkbox changes are separate from the preserved suffixes.
        if name == 'coding-plan.md':
            for milestone in ('D08', 'T01'):
                current = current.replace(f'| [x] | {milestone} /'.encode(), f'| [ ] | {milestone} /'.encode())
        assert current.startswith(original), name
        for label, commit in COMMITS.items():
            source = git('show', f'{commit}:{name}')
            assert source.startswith(original)
            assert source[len(original):] in current, (name, label)
        original_docs[name] = {'base_prefix_preserved': True, 'both_exact_completion_suffixes_preserved': True}
    for label, commit in COMMITS.items():
        assert git('merge-base', '--is-ancestor', commit, 'HEAD') == b''
        changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()
        for name in changed:
            if name in ('README.md', 'coding-plan.md'):
                continue
            assert (ROOT / name).read_bytes() == git('show', f'{commit}:{name}'), (label, name)
            imported[name] = {'source': label, **identity(ROOT / name)}
    old = json.loads((ROOT / 'runs/d06-d07-e01-integration/preservation.json').read_text())
    external = {}
    for name, expected in old['original_external_identities'].items():
        assert identity(Path(name)) == expected, name
        external[name] = expected
    results = json.loads((ROOT / 'runs/d08/results.json').read_text())
    artifacts = {}
    assert {p.name for p in Path(results['bulk_directory']).iterdir()} == set(results['artifacts'])
    for name, expected in results['artifacts'].items():
        expected_identity = {k: expected[k] for k in ('bytes', 'sha256')}
        assert identity(Path(expected['path'])) == expected_identity, name
        assert identity(ARCHIVE / name) == expected_identity, ('durable copy', name)
        artifacts[name] = expected_identity
    assert sum(v['bytes'] for v in artifacts.values()) == results['full_production_storage_bytes']
    setup = json.loads((OUT / 'setup.json').read_text())
    assert git('rev-parse', 'HEAD', cwd=PRIMARY).decode().strip() == BASE
    assert git('status', '--porcelain=v1', '-uall', cwd=PRIMARY).decode() == setup['primary_status']
    assert identity(PRIMARY / 'coding-plan.md') == setup['primary_coding_plan_identity']
    patch = base64.b64decode(json.loads((OUT / 'primary-coding-plan-patch.json').read_text())['data'], validate=True)
    assert git('diff', '--binary', '--', 'coding-plan.md', cwd=PRIMARY) == patch
    report = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE,
                  source_commits=COMMITS, accepted_base_files=base_files,
                  exact_imported_files=imported, appended_documents=original_docs,
                  accepted_external_inputs=external, original_d08_and_durable_copy_artifacts=artifacts,
                  durable_copy=str(ARCHIVE), total_d08_bytes=sum(v['bytes'] for v in artifacts.values()),
                  primary_main_and_unrelated_edits_preserved=True)
    save(destination, report)
    print(json.dumps(dict(base_files=len(base_files), exact_imported_files=len(imported),
                         external_inputs=len(external), d08_artifacts=len(artifacts),
                         durable_copy=str(ARCHIVE), primary_edits_preserved=True), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    verify(p.parse_args().output)
