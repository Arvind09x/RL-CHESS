"""Replay JSONL E01 match artifacts without a model or engine."""
import argparse
import json
from pathlib import Path
from chess_rl.evaluation.replay import audit_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audits = [audit_match(json.loads(line)) for line in args.matches.read_text().splitlines() if line.strip()]
    report = dict(audits=audits, matches=len(audits), games=sum(a["games"] for a in audits),
                  generated_plies=sum(a["generated_plies"] for a in audits),
                  completed_search_simulations=sum(a["completed_search_simulations"] for a in audits),
                  measured_engine_nodes=sum(a["measured_engine_nodes"] for a in audits), passed=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
