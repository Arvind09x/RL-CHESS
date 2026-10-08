"""D06 selection and timing, separate from the accepted D05 worker."""
from collections import Counter
import hashlib
import heapq
import logging
import math
import time

from chess_rl.canonical_games import canonical_json, load_json

SELECTION_VERSION = "d06-rule-sha256-v1"
SEED = 20261010


def key(ref):
    return ref["game_id"], ref["sample_ply"]


def rank(position):
    payload = dict(selection_version=SELECTION_VERSION, seed=SEED,
                   rule_position=position)
    return int(hashlib.sha256(canonical_json(payload).encode()).hexdigest(), 16)


def select_references(rows, excluded, count=1000):
    """Lowest hash ranks of distinct train rule positions; stable representative.

    For each retained identity choose the lexicographically first (game_id, ply)
    audited reference. D05 references are removed before ranking. A bounded heap
    avoids holding the corpus in memory. No outcome or engine result is read.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be positive")
    heap, chosen = [], {}
    candidates = Counter()
    for ref in rows:
        if ref["split"] != "train" or key(ref) in excluded:
            continue
        bucket = ref["phase"] + ":" + ref["rule_position"].split()[1]
        if ref["phase"] not in ("opening", "middlegame", "endgame") or bucket[-1] not in "wb":
            raise ValueError("Invalid phase/color")
        candidates[bucket] += 1
        position = ref["rule_position"]
        if position in chosen:
            if key(ref) < key(chosen[position]):
                chosen[position] = ref
            continue
        score = rank(position)
        if len(heap) == count and (score, position) >= (-heap[0][0], heap[0][1]):
            continue
        if len(heap) == count:
            _, removed = heapq.heappop(heap)
            del chosen[removed]
        heapq.heappush(heap, (-score, position))
        chosen[position] = ref
    if len(chosen) != count:
        raise ValueError("Insufficient distinct eligible training positions")
    return sorted(chosen.values(), key=lambda r: (rank(r["rule_position"]), r["rule_position"])), dict(candidates)


def read_rows(path):
    with path.open() as handle:
        for raw in handle:
            yield load_json(raw)


def distribution(values):
    """Nearest-rank percentiles; no interpolation or empty-denominator fiction."""
    values = sorted(values)
    if not values:
        return dict(count=0, sum=0, mean=None, min=None, max=None, p50=None, p90=None, p95=None, p99=None)
    result = dict(count=len(values), sum=sum(values), mean=sum(values)/len(values), min=values[0], max=values[-1])
    result.update({f"p{p}": values[max(0, math.ceil(p/100*len(values))-1)] for p in (50, 90, 95, 99)})
    return result


class TransportClock(logging.Handler):
    """Observe existing transport events without changing worker or UCI behavior."""
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.started = time.perf_counter()
        self.events = []

    def emit(self, record):
        if record.msg in ("%s: >> %s", "%s: << %s") and len(record.args) == 2:
            direction = "received" if ">>" in record.msg else "sent"
            line = record.args[1]
            if line in ("uci", "uciok", "isready", "readyok", "setoption name Clear Hash", "ucinewgame", "quit") or line.startswith(("go ", "bestmove ")):
                self.events.append(dict(seconds=time.perf_counter()-self.started, direction=direction, line=line))

    def metrics(self, elapsed):
        def first(prefix):
            return next((e["seconds"] for e in self.events if e["line"].startswith(prefix)), None)
        uciok, clear, go, best, quit_at = (first(s) for s in ("uciok", "setoption name Clear Hash", "go ", "bestmove ", "quit"))
        ready = next((e["seconds"] for e in self.events if e["line"] == "readyok" and clear is not None and e["seconds"] > clear), None)
        delta = lambda end, start: None if end is None or start is None else end-start
        return dict(engine_attempt_seconds=elapsed, startup_to_uciok_seconds=uciok,
                    explicit_reset_to_ready_seconds=delta(ready, clear),
                    ready_to_go_seconds=delta(go, ready),
                    go_to_bestmove_seconds=delta(best, go),
                    bestmove_to_return_seconds=None if best is None else elapsed-best,
                    quit_to_return_seconds=None if quit_at is None else elapsed-quit_at,
                    events=self.events)
