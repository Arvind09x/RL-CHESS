"""Append training candidates while verifying a frozen D04 registry."""
import argparse
import json
from pathlib import Path
from chess_rl.data_manifest import DatasetManifest
from chess_rl.position_expansion import expand_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','games','split-audit','previous-dir','previous-games','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split-audit-sha256',required=True)
    p.add_argument('--previous-audit-sha256',required=True)
    a = p.parse_args()
    print(json.dumps(expand_positions(a.games,a.split_audit,a.split_audit_sha256,
        DatasetManifest.load(a.manifest),a.previous_dir,a.previous_games,
        a.previous_audit_sha256,a.output_dir),indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
