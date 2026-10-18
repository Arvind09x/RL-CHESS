"""Pin a clean checkout on the submission host; verify it without Git in a job."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        raise ValueError('prepare requires a clean committed isolated checkout')
    revision = git('rev-parse', 'HEAD').decode().strip()
    files = git('ls-files', '-z').decode().split('\0')
    from chess_rl.selfplay_execution import source_identity
    manifest = dict(schema='batching-source-v1', revision=revision,
                    execution_source=source_identity(root),
                    files={name:sha256(root / name) for name in files if name})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest


def verify(root, source_record):
    """Read-only content verification; never invoke Git or a model."""
    root = Path(root).resolve()
    manifest = json.loads(Path(source_record).read_text())
    revision = manifest.get('revision', '')
    if (manifest.get('schema') != 'batching-source-v1' or len(revision) != 40
            or any(c not in '0123456789abcdef' for c in revision)
            or not isinstance(manifest.get('files'), dict) or not manifest['files']):
        raise ValueError('invalid source identity record')
    for name, expected in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not path.parts or sha256(root / path) != expected:
            raise ValueError(f'frozen source changed: {name}')
    from chess_rl.selfplay_execution import source_identity
    if manifest.get("execution_source") != source_identity(root):
        raise ValueError("source inventory changed, including added/untracked execution files")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/batching-source.json')
    args = parser.parse_args()
    manifest = prepare(ROOT, args.output)
    print(json.dumps(dict(revision=manifest['revision'], record=str(args.output), files=len(manifest['files']))))


if __name__ == '__main__':
    main()
