"""Prepare D04 positions from accepted D03 bytes; no engine or labels."""
import argparse
import json
from pathlib import Path
from chess_rl.data_manifest import DatasetManifest
from chess_rl.position_sampling import prepare_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--games', type=Path, required=True)
    p.add_argument('--split-audit', type=Path, required=True)
    p.add_argument('--split-audit-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    report = prepare_positions(args.games, args.split_audit, args.split_audit_sha256,
                               DatasetManifest.load(args.manifest), args.output_dir,
                               workers=args.workers,
                               progress=lambda n: print(n, flush=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
