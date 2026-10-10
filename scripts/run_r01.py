"""Create or resume one frozen R01 CPU game. No training or replay buffer."""
import argparse
import json
from pathlib import Path

import torch

from chess_rl.evaluation.matches import Opening
from chess_rl.selfplay import Session, Settings, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Required for create; frozen JSON settings/opening/checkpoint/seeds")
    parser.add_argument("--stop-after-moves", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == "create":
        if args.plan is None:
            parser.error("create requires --plan")
        plan = json.loads(args.plan.read_text())
        opening = Opening(**plan["opening"])
        session = Session.create(args.directory, plan["checkpoint_path"], opening,
            settings=Settings.from_dict(plan["settings"]), seed=plan["seed"],
            opening_selection_seed=plan["opening_selection_seed"])
    else:
        if args.plan is not None:
            parser.error("resume uses recorded settings; --plan is not accepted")
        session = Session.load(args.directory)
    result = session.run(stop_after_moves=args.stop_after_moves)
    print(json.dumps(summarize([result]), indent=2))


if __name__ == "__main__":
    main()
