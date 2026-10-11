"""Fresh-process real-checkpoint journal worker for the SIGKILL regression."""
import json
import signal
import sys
from pathlib import Path

import torch

from chess_rl import selfplay as impl
from chess_rl.evaluation.matches import digest
from chess_rl.replay import finalize
from chess_rl.selfplay import Phase, Session, Settings
from .test_selfplay import controlled, opening


def main():
    mode, directory, checkpoint, output = sys.argv[1:]
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    session = (Session.load(directory) if mode == "recover" else
        Session.create(directory, checkpoint, opening(), settings=Settings(1, 8,
            phases=(Phase(0, 0., 0.),)), seed=971, opening_selection_seed=73, journal=True))
    # The real checkpoint loader/identity/runtime paths execute in every process.
    # Force a known completed outcome through the actual search/adapter/journal.
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    def evaluator(states):
        return controlled(moves[len(states[0].board.move_stack)])(states)
    session.evaluator = evaluator
    if mode == "crash":
        original = impl._write
        def paused_pointer(path, value, **kwargs):
            if Path(path).name == "boundary.json" and value.get("journal_positions") == 3:
                assert (directory / "moves/000002.json").is_file()
                print("DURABLE_MOVE_READY", flush=True)
                signal.pause()  # parent kills us before pointer publication
                raise AssertionError("worker should have been killed")
            return original(path, value, **kwargs)
        impl._write = paused_pointer
    snapshot = session.run()
    result = finalize(snapshot)
    Path(output).write_text(json.dumps(dict(snapshot=snapshot, source_id=result["source_id"],
        game_id=result["game_id"], sample_ids=[s["sample_id"] for s in result["samples"]],
        record_sha256=digest(snapshot["game"])), sort_keys=True))


if __name__ == "__main__":
    main()
