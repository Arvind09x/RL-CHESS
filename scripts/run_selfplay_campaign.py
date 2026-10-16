"""Create, run, inspect and archive a resumable self-play campaign.

create freezes the plan and plays nothing. run plays or resumes every
unfinished game; rerun it after any stop. status reports progress, game
lengths, colour balance, per-move timing and a finish projection. archive
audits every finished game into one JSONL file for scripts/run_r02.py. choose
records, once, which of two campaign halves form a round under a deadline.
See chess_rl.selfplay_campaign for the guarantees.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from chess_rl import selfplay_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def epoch(text):
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError("deadline needs a UTC offset, e.g. 2026-10-17T10:00:00+05:30")
    return moment.timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="freeze a new campaign plan; plays nothing")
    create.add_argument("--directory", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--games", type=int, required=True)
    create.add_argument("--simulations", type=int, required=True, help="search simulations per move")
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    create.add_argument("--purpose", required=True, help="e.g. 'self-play round 1 from P (1M step 90000)'")
    run = commands.add_parser("run", help="play or resume every unfinished game")
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--heavy-lock", type=Path, default=ROOT / "runs/.heavy-job.lock",
                     help="shared lock so self-play and matches never overlap on one machine")
    run.add_argument("--deadline", type=epoch, help="ISO time with offset; stop starting and running games then")
    for name in ("status", "archive"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
    commands.choices["archive"].add_argument("--output", type=Path, help="default: <directory>/games.jsonl")
    choose = commands.add_parser("choose", help="record once which halves form the round")
    for name in ("output", "first", "second", "first-buffer", "both-buffer"):
        choose.add_argument(f"--{name}", type=Path, required=True)
    choose.add_argument("--deadline", type=epoch, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = campaign.create(args.directory, args.checkpoint, games=args.games, simulations=args.simulations,
                                 seed=args.seed, device=args.device, purpose=args.purpose)
        result = {k: v for k, v in result.items() if k != "games"} | {"games": len(result["games"])}
    elif args.command == "run":
        result = campaign.run(args.directory, workers=args.workers, heavy_lock=args.heavy_lock,
                              deadline=args.deadline)
    elif args.command == "status":
        result = campaign.status(args.directory)
    elif args.command == "archive":
        result = campaign.archive(args.directory, args.output)
    else:
        result = campaign.choose_round(args.output, first=args.first, second=args.second,
            deadline=args.deadline, first_buffer=args.first_buffer, both_buffer=args.both_buffer)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
