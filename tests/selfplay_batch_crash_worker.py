"""Hard-stop the real batched Session after a move file, before its boundary pointer."""
from pathlib import Path
import signal
import sys
from chess_rl import selfplay as S, selfplay_campaign as C
from chess_rl.runtime import configure
from chess_rl.selfplay_batching import play_group
from .test_selfplay_batching import tasks

if __name__ == '__main__':
    mode, raw = sys.argv[1:]
    path = Path(raw)
    configure('cpu')
    if mode == 'crash':
        original = S._write
        def paused(target, value, **kwargs):
            if Path(target) == path / 'games/0000/boundary.json' and value.get('journal_positions') == 2:
                print('DURABLE_MOVE_READY', flush=True)
                signal.pause()
            return original(target, value, **kwargs)
        S._write = paused
    plan = C.load(path)
    result = play_group(tasks(path, plan), plan, path)
    assert all(g['status'] in C.FINAL for g in result['game_results']), result
