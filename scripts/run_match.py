"""Freeze and play the smoke, validation, rung and test matches; report and select.

openings    freeze the 64 validation + 64 test openings from the protected registry (once)
candidates  freeze P, the validation candidates and their eligibility, and the selection rules (once)
plan        freeze one match (players, openings, seeds, settings, runtime, sources); plays nothing
run         play or resume every game without a stored record
status      counts only, no scores
report      smoke: statuses, lengths and timing; other matches: scores of a finished schedule only
select      apply the frozen selection rule to the finished validation matches (once)
See chess_rl.evaluation.match_campaign for the guarantees.
"""
import argparse
import json
from pathlib import Path

from chess_rl.evaluation import match_campaign as M
from chess_rl.evaluation.pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (checkpoint paths are relative to it)")
    parser.add_argument("--matches", type=Path, help="default: <root>/runs/matches")
    commands = parser.add_subparsers(dest="command", required=True)
    openings = commands.add_parser("openings")
    openings.add_argument("--data-config", type=Path, required=True, help="e.g. configs/data-d09-1m.json")
    commands.add_parser("candidates")
    plan = commands.add_parser("plan")
    plan.add_argument("--kind", choices=M.KINDS, required=True)
    plan.add_argument("--candidate", help="smoke/validation: a frozen candidate; test/rung: defaults to the selection")
    plan.add_argument("--opponent", choices=M.RUNG_OPPONENTS, help="rung opponent")
    plan.add_argument("--stockfish", type=Path, help="Stockfish rungs: the pinned executable (full sha256 is checked)")
    plan.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    plan.add_argument("--workers", type=int, required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("--match-id", required=True)
    commands.add_parser("select")
    args = parser.parse_args()
    root = args.root.resolve()
    matches = (args.matches or root / "runs/matches").resolve()
    openings_path, candidates_path = matches / "openings.json", matches / "candidates.json"
    if args.command == "openings":
        from chess_rl.runtime import data_paths
        paths = data_paths(args.data_config)
        result = M.freeze_openings(openings_path, registry_dir=paths["prepared"], games_path=paths["games_path"],
                                   manifest_path=paths["manifest_path"])
        result = dict(path=result["path"], file=result["file"], set_sha256=result["record"]["sets"]["sha256"])
    elif args.command == "candidates":
        result = M.freeze_candidates(candidates_path, root=root)["record"]
    elif args.command == "plan":
        result = M.create_plan(matches, root=root, kind=args.kind, candidate=args.candidate, opponent=args.opponent,
                               openings_path=openings_path, candidates_path=candidates_path, device=args.device,
                               workers=args.workers, selection_path=matches / "selection.json",
                               stockfish=args.stockfish)
        result = {k: v for k, v in result.items() if k not in ("schedule", "openings", "sources")}
    elif args.command == "run":
        result = M.run(matches / args.match_id, root)
    elif args.command == "status":
        result = M.status(matches / args.match_id, root)
    elif args.command == "report":
        result = M.match_report(matches / args.match_id, root)
        atomic_json(matches / args.match_id / "report.json", result)
    else:
        result = M.select(matches, root, candidates_path=candidates_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "select" and not result["final"]:
        raise SystemExit(f"selection not final: {result['status']}")


if __name__ == "__main__":
    main()
