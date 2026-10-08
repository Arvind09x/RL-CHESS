"""Run unchanged historical M05 checker through explicit constructor disabling."""
from functools import partial
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chess_rl.model import ChessNetwork
import tests.network_checks as checks
checks.ChessNetwork = partial(ChessNetwork, relative_enabled=False)
if __name__ == '__main__':
    script = sys.argv.pop(1)
    sys.argv[0] = script
    runpy.run_path(script, run_name='__main__')
