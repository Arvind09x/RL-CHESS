"""Post-test development acceptance driver. No Git or scheduler invocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.prepare_batching_source import sha256, verify
from chess_rl.evaluation.benchmark_layouts import DEFAULT_LAYOUTS, parse_layout

ROOT = Path(__file__).resolve().parents[1]


def commands(root, output, left, right, *, device='cuda:0', layouts=DEFAULT_LAYOUTS, pairs=8):
    executions = [parse_layout(layout) for layout in layouts]
    batches = sorted({e.batch_size for e in executions if e.mode == 'batched'})
    common = ['--left', str(left), '--right', str(right), '--device', device]
    bench = str(Path(root) / 'scripts/benchmark_matches.py')
    calls = []
    for batch in batches:
        for stage, folder in (('accept', 'numerical'), ('check-games', 'games')):
            calls.append([sys.executable, bench, stage, *common, '--batch-size', str(batch),
                          '--output', str(Path(output) / f'{folder}-b{batch}')])
    calls.append([sys.executable, str(Path(root) / 'scripts/profile_match_evaluator.py'),
                  '--checkpoint', str(left), '--device', device, '--batch-size', str(max(batches, default=1)),
                  '--output', str(Path(output) / 'profile')])
    calls.append([sys.executable, bench, 'benchmark', *common, '--layouts', *layouts, '--pairs', str(pairs),
                  '--simulations', '128', '--ply-cap', '512', '--flush-ms', '20',
                  '--output', str(Path(output) / 'throughput')])
    return calls


def run(root, source_record, output, left, left_sha256, right, right_sha256, *, runner=subprocess.run):
    root, output = Path(root).resolve(), Path(output)
    manifest = verify(root, source_record)
    paths = [root / left, root / right]
    for path, expected in zip(paths, (left_sha256, right_sha256), strict=True):
        if len(expected) != 64 or sha256(path) != expected:
            raise ValueError(f'checkpoint checksum mismatch: {path}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / 'source-revision.txt').write_text(manifest['revision'] + '\n')
    (output / 'checkpoints.json').write_text(json.dumps(dict(left=dict(path=str(paths[0]), sha256=left_sha256),
                                                          right=dict(path=str(paths[1]), sha256=right_sha256)), indent=2))
    for command in commands(root, output, *paths):
        runner(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--left-sha256', required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--right-sha256', required=True)
    args = parser.parse_args()
    run(ROOT, args.source_record, args.output, args.left, args.left_sha256, args.right, args.right_sha256)


if __name__ == '__main__':
    main()
