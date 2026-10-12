"""Prepare, verify, run or report the fixed CPU P02 development strength check.

Only the explicit `run` subcommand launches games. No training/data-label APIs.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_rl.evaluation.pilot import (atomic_json, load_plan, load_players, prepare,
                                      report, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/p02-cpu-strength-dev.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/p02-cpu-001-strength-dev")
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare(ROOT, args.plan)
        print(json.dumps(dict(plan=str(args.plan.resolve()), sha256=plan["sha256"],
                              games=plan["planned_games"], pairs=plan["planned_pairs"])))
        return
    plan = load_plan(args.plan, ROOT)
    if args.action == "verify":
        from chess_rl.runtime import configure
        from chess_rl.adapter import make_state
        import numpy as np
        configure("cpu", threads=1)
        players = load_players(plan, ROOT)
        # Checkpoint forwards only: no search, move selection or game execution.
        state = make_state()
        outputs = [p.evaluator([state])[0] for p in players]
        if np.array_equal(outputs[0].priors, outputs[1].priors):
            raise ValueError("Candidate/baseline unexpectedly have identical policies")
        for p, result in zip(players, outputs, strict=True):
            again = p.evaluator([state])[0]
            if not np.array_equal(result.priors, again.priors) or not np.array_equal(result.wdl, again.wdl):
                raise ValueError("Repeated deterministic forward differs")
        evidence = dict(action="checkpoint-forward-verification; no games or searches",
            plan_sha256=plan["sha256"], players=[p.metadata() for p in players],
            distinct_policies=True, repeated_forwards_identical=True)
        atomic_json(ROOT / "artifacts/p02-cpu-001-preserved/strength-routing-verification.json", evidence)
        print(json.dumps(evidence))
    elif args.action == "report":
        summary = report(args.output / "results.sqlite3", plan)
        atomic_json(args.output / "summary.json", summary)
        print(json.dumps(summary))
    else:
        run(plan, ROOT, args.output)


if __name__ == "__main__":
    main()
