"""Deterministic fake Stockfish for labeller tests: legal PVs, fixed WDL, UCI option surface.

Set FAKE_ENGINE_SLEEP=seconds to slow each search (interruption tests).
"""
import os
import sys
import time

import chess

NAME = "Stockfish 17.1"
AUTHOR = "the Stockfish developers (see AUTHORS file)"
NETS = sys.argv[1:3]


def options():
    return [
        "option name Debug Log File type string default <empty>",
        "option name Threads type spin default 1 min 1 max 1024",
        "option name Hash type spin default 16 min 1 max 33554432",
        "option name Clear Hash type button",
        "option name Ponder type check default false",
        "option name MultiPV type spin default 1 min 1 max 256",
        "option name Skill Level type spin default 20 min 0 max 20",
        "option name Move Overhead type spin default 10 min 0 max 5000",
        "option name nodestime type spin default 0 min 0 max 10000",
        "option name UCI_Chess960 type check default false",
        "option name UCI_LimitStrength type check default false",
        "option name UCI_Elo type spin default 1320 min 1320 max 3190",
        "option name UCI_ShowWDL type check default false",
        "option name SyzygyPath type string default <empty>",
        "option name SyzygyProbeDepth type spin default 1 min 1 max 100",
        "option name Syzygy50MoveRule type check default true",
        "option name SyzygyProbeLimit type spin default 7 min 0 max 7",
        f"option name EvalFile type string default {NETS[0]}",
        f"option name EvalFileSmall type string default {NETS[1]}",
    ]


def main():
    board = chess.Board()
    delay = float(os.environ.get("FAKE_ENGINE_SLEEP", "0"))
    out = sys.stdout
    for raw in sys.stdin:
        line = raw.strip()
        if line == "uci":
            print(f"id name {NAME}", file=out)
            print(f"id author {AUTHOR}", file=out)
            print(file=out)
            for option in options():
                print(option, file=out)
            print("uciok", file=out)
        elif line == "isready":
            print("readyok", file=out)
        elif line.startswith("position"):
            tokens = line.split()
            if tokens[1] == "startpos":
                board = chess.Board()
                moves = tokens[3:] if len(tokens) > 2 else []
            else:
                end = tokens.index("moves") if "moves" in tokens else len(tokens)
                board = chess.Board(" ".join(tokens[2:end]))
                moves = tokens[end + 1:] if "moves" in tokens else []
            for uci in moves:
                board.push_uci(uci)
        elif line.startswith("go"):
            if delay:
                time.sleep(delay)
            nodes = int(line.split()[-1])
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            pv = [legal[0]]
            after = board.copy()
            after.push(pv[0])
            reply = sorted(after.legal_moves, key=lambda m: m.uci())
            if reply:
                pv.append(reply[0])
            pv_text = " ".join(m.uci() for m in pv)
            for depth in (1, 2):
                print(f"info depth {depth} seldepth {depth + 1} multipv 1 score cp {17 * depth} wdl 420 380 200 "
                      f"nodes {nodes // (3 - depth)} nps 900000 hashfull 1 tbhits 0 time {depth * 11} pv {pv_text}", file=out)
            print(f"bestmove {pv[0].uci()}", file=out)
        elif line == "quit":
            break
        # setoption, ucinewgame, compiler: silently accepted
        out.flush()


if __name__ == "__main__":
    main()
