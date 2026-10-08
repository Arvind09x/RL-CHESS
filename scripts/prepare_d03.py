"""Consolidate verified D02 source games and assign stable splits before sampling."""
import argparse
import json
from pathlib import Path

from chess_rl.canonical_games import prepare_games
from chess_rl.data_manifest import DatasetManifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_games(args.replays, args.summary, args.summary_sha256,
                           DatasetManifest.load(args.manifest), args.output_dir,
                           progress=lambda n: print(f"Consolidated {n} source records", flush=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
