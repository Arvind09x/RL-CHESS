"""Create or resume one frozen R01 CPU/CUDA game. No training or replay buffer."""
import argparse
import json
from pathlib import Path

from chess_rl.runtime import configure

from chess_rl.evaluation.matches import Opening
from chess_rl.selfplay import Session, Settings, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Required for create; frozen JSON settings/opening/checkpoint/seeds")
    parser.add_argument("--stop-after-moves", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Relocated checkpoint for resume; bytes must match")
    args = parser.parse_args()
    configure(args.device)
    if args.mode == "create":
        if args.checkpoint is not None:
            parser.error("create uses the checkpoint in --plan; --checkpoint is for resume")
        if args.plan is None:
            parser.error("create requires --plan")
        plan = json.loads(args.plan.read_text())
        opening = Opening(**plan["opening"])
        session = Session.create(args.directory, (args.plan.parent / plan["checkpoint_path"]).resolve(), opening,
            settings=Settings.from_dict(plan["settings"]), seed=plan["seed"],
            opening_selection_seed=plan["opening_selection_seed"], device=args.device, journal=True)
    else:
        if args.plan is not None:
            parser.error("resume uses recorded settings; --plan is not accepted")
        session = Session.load(args.directory, checkpoint_path=args.checkpoint)
    result = session.run(stop_after_moves=args.stop_after_moves)
    print(json.dumps(summarize([result]), indent=2))


if __name__ == "__main__":
    main()
