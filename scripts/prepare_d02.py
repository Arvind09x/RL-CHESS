"""Prepare all source mainlines from the selected archive; bulk output outside Git."""
import argparse
import json
from pathlib import Path

from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import prepare_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_archive(args.archive, DatasetManifest.load(args.manifest), args.output_dir,
                             progress=lambda member, ordinal, counts: print(member, ordinal, counts, flush=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
