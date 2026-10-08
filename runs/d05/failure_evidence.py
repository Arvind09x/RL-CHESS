"""Save controlled fixture attempts separately from real Stockfish labels."""
import asyncio
import chess
import json
from pathlib import Path

from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import json_line
from chess_rl.stockfish_worker import AnnotationWorker
from tests.test_stockfish_worker import fixture_game, response

ROOT = Path(__file__).resolve().parents[2]
manifest = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-d04-v1.json")
out = ROOT / "runs/d05/failure-fixtures"
out.mkdir(exist_ok=False)
results = []
with (out / "attempts.jsonl").open("x") as handle:
    for scenario in ("missing_wdl", "illegal_pv", "missing_bestmove", "timeout_retry",
                     "timeout_exhaustion", "keyboard_interrupt", "cancelled"):
        game, ref, _ = fixture_game(manifest, 1)
        saved = []
        def sink(record):
            handle.write(json_line(dict(scenario=scenario, fixture_only=True, **record)))
            handle.flush()
            saved.append(record)
        async def attempt(state, trace):
            trace.update(hash_reset_completed=True, completed=True)
            if saved and scenario != "timeout_exhaustion":
                return response(state)
            if scenario.startswith("timeout"):
                trace["completed"] = False
                raise TimeoutError("controlled deadline")
            if scenario == "keyboard_interrupt":
                raise KeyboardInterrupt("controlled interruption")
            if scenario == "cancelled":
                raise asyncio.CancelledError("controlled cancellation")
            info, best = response(state)
            if scenario == "missing_wdl":
                info.pop("wdl")
            if scenario == "illegal_pv":
                info["pv"] = [chess.Move.from_uci("e2e5")]
            if scenario == "missing_bestmove":
                best = None
            return info, best
        worker = AnnotationWorker(manifest, {"fixture_only": True}, attempt, sink, max_attempts=2)
        label, propagated = None, None
        try:
            label = asyncio.run(worker.annotate(ref, game))
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            propagated = type(error).__name__
        results.append(dict(scenario=scenario, attempts=len(saved), statuses=[r["status"] for r in saved],
            successful_labels=int(label is not None), explicitly_excluded=label is None and propagated is None,
            propagated=propagated))
(out / "summary.json").write_text(json.dumps(dict(fixture_only=True, results=results), indent=2)+"\n")
print(json.dumps(results, indent=2))
