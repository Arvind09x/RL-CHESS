"""Bounded protocol fixtures; absent P/R checkpoints remain explicitly null."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import chess
import numpy as np
from chess_rl.adapter import Evaluation, make_state, move_to_id, terminal_result
from chess_rl.evaluation import (Decision, Opening, OpeningSets, SearchPlayer, SearchSettings,
    RandomLegalPlayer, GreedyPlayer, StockfishPin, StockfishPlayer, run_match)
from chess_rl.search.evaluator import evaluate

ROOT = Path(__file__).resolve().parents[2]
# Accepted D01 binary pin, independent of any annotation settings.
PIN = StockfishPin("/private/tmp/Stockfish-sf_17.1/src/stockfish",
    "3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4",
    "Stockfish 17.1", "17.1", "03e27488f3d21d8ff4dbf3065603afa21dbd0ef3")


def player(role, evaluator=evaluate, settings=SearchSettings(2), evaluator_id="s01-uniform-dummy-v1"):
    return SearchPlayer("protocol-fixture-" + role, role, evaluator_id, evaluator, settings)


def opening(name, state):
    return Opening.from_state(name, state, source="e01 hand-authored controlled protocol fixture; no production book")


def book(state=None):
    return OpeningSets("e01-smoke-openings-v1", (opening("validation-e4e5" if state is None else "validation-setup",
        state or make_state(moves_uci=["e2e4", "e7e5"])),),
        (opening("reserved-d4d5", make_state(moves_uci=["d2d4", "d7d5"])),))


def one_hot(uci):
    def evaluator(states):
        results = []
        for state in states:
            terminal = terminal_result(state)
            if terminal is not None:
                results.append(terminal)
            elif chess.Move.from_uci(uci) not in state.board.legal_moves:
                results.extend(evaluate([state]))
            else:
                priors = np.zeros(4352, dtype=np.float32)
                priors[move_to_id(state, chess.Move.from_uci(uci))] = 1
                results.append(Evaluation(priors, [0, 1, 0], 0))
        return results
    return evaluator


class Controlled:
    def __init__(self, uci): self.uci = uci
    def metadata(self):
        return dict(player_id="controlled-actual-move", kind="controlled", version="e01-smoke-v1", settings=dict(uci=self.uci))
    def choose(self, state, *, seed):
        return Decision(move_to_id(state, chess.Move.from_uci(self.uci)), dict(controlled=True))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    plans = []
    def plan(name, left, right, openings, cap, comparison="fixed-opponent", split="validation"):
        plans.append(dict(name=name, left=left, right=right, openings=openings, cap=cap, comparison=comparison, split=split))
    plan("dummy-pr-validation", player("P"), player("R"), book(), 4, "P-versus-R")
    plan("dummy-pr-reserved-fixture", player("P"), player("R"), book(), 4, "P-versus-R", "test")
    plan("dummy-random", player("P"), RandomLegalPlayer(), book(), 4)
    plan("dummy-greedy", player("P"), GreedyPlayer(), book(), 4)
    for nodes in (32, 128):
        plan(f"dummy-stockfish-{nodes}", player("P"), StockfishPlayer(PIN, nodes=nodes), book(), 4)
    plan("mate-last-ply", player("P", one_hot("g6g7"), evaluator_id="controlled-mate-v1"),
         player("R", one_hot("g6g7"), evaluator_id="controlled-mate-v1"),
         book(make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")), 1, "P-versus-R")
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    plan("reached-repetition", player("P", one_hot("f6g8"), evaluator_id="controlled-cycle-v1"),
         player("R", one_hot("f6g8"), evaluator_id="controlled-cycle-v1"),
         book(make_state(moves_uci=cycle + cycle[:3])), 8, "P-versus-R")
    for status in ("failed", "interrupted"):
        def fault(states, status=status):
            raise RuntimeError("controlled fault") if status == "failed" else KeyboardInterrupt("controlled interrupt")
        plan("fault-" + status, player("P", fault, evaluator_id="controlled-" + status), player("R"), book(), 3, "P-versus-R")
    specials = [("r3k2r/8/8/8/8/8/8/R3K2R " + c + " KQkq - 0 1", origin + dest)
        for c, origin, dests in [("w", "e1", ("g1", "c1")), ("b", "e8", ("g8", "c8"))] for dest in dests]
    specials += [("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6"),
                 ("7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1", "e4d3")]
    specials += [(fen, origin + p) for fen, origin in [
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8"),
        ("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1")] for p in "qrbn"]
    for index, (fen, uci) in enumerate(specials):
        plan(f"special-{index:02d}-{uci}", player("P", one_hot(uci), SearchSettings(1), "controlled-special-v1"),
             Controlled(uci), book(make_state(fen)), 1)
    # Freeze plans/identities before any game. Test fixtures are not E02 results.
    config = [dict(match_id=p["name"], left=p["left"].metadata(), right=p["right"].metadata(),
        openings=p["openings"].to_dict(), cap=p["cap"], comparison=p["comparison"], split=p["split"],
        protocol_only=True, seed=20261008) for p in plans]
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    with (out / "matches.jsonl").open("w") as handle:
        for p in plans:
            record = run_match(p["left"], p["right"], p["openings"], split=p["split"], match_id=p["name"],
                comparison=p["comparison"], seed=20261008, ply_cap=p["cap"], protocol_only=True).to_dict()
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(matches=len(plans), planned_games=len(plans) * 2, real_checkpoints_available=False)))


if __name__ == "__main__": main()
