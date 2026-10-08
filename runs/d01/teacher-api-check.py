"""Check accepted python-chess UCI transport without any analysis/search call."""
import json
from pathlib import Path
import tempfile

import chess.engine
from chess_rl.data_manifest import DatasetManifest

ROOT = Path(__file__).resolve().parents[2]
manifest = DatasetManifest.load(ROOT / 'data/manifests/elite-2025-01-v1.json')
with tempfile.TemporaryDirectory(prefix='d01-python-chess-') as cwd:
    with chess.engine.SimpleEngine.popen_uci('/private/tmp/Stockfish-sf_17.1/src/stockfish', cwd=cwd) as engine:
        assert engine.id['name'] == manifest.engine.uci_name
        assert engine.id['author'] == manifest.engine.uci_author
        assert engine.options['MultiPV'].default == 1
        for name, value in {'Threads': 1, 'Hash': 128, 'UCI_ShowWDL': True}.items():
            assert engine.options[name].parse(value) == value
        # MultiPV is managed by python-chess's eventual analyse(multipv=1), not
        # configure(). The default and supported range are verified here/D01.
        engine.configure({'Threads': 1, 'Hash': 128, 'UCI_ShowWDL': True,
                          'Clear Hash': None, 'SyzygyPath': ''})
        engine.ping()
        print(json.dumps({'id': engine.id, 'configured': {'Threads': 1, 'Hash': 128,
              'UCI_ShowWDL': True, 'Clear Hash': None, 'SyzygyPath': ''},
              'MultiPV_default': engine.options['MultiPV'].default,
              'transport': 'python-chess 1.999 / chess 1.11.2',
              'analysis_or_play_calls': 0}, indent=2))
        engine.quit()
