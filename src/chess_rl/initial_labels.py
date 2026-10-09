"""D08 fixed production membership and explicit three-split root replay.

No targets, game outcomes, replacement references or training enter selection.
"""
from collections import Counter
import hashlib
import heapq

from chess_rl.adapter import make_state, terminal_result
from chess_rl.canonical_games import canonical_json
from chess_rl.position_sampling import RECORD_VERSION, phase, rule_position, validate_game

SEED = 20261011
SELECTION_VERSION = "d08-reference-sha256-v1"
QUOTAS = {"train": 9000, "validation": 500, "test": 500}
REFERENCE_FIELDS = {"record_version", "game_id", "sample_ply", "split", "phase", "rule_position"}


def key(ref):
    return ref["game_id"], ref["sample_ply"]


def selection_order(ref):
    payload = dict(selection_version=SELECTION_VERSION, seed=SEED,
                   split=ref["split"], game_id=ref["game_id"], sample_ply=ref["sample_ply"])
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest(), *key(ref)


class _Worst:
    def __init__(self, ref):
        self.ref, self.order = ref, selection_order(ref)

    def __lt__(self, other):
        return self.order > other.order


def select_references(rows, excluded, *, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    if set(quotas) != set(QUOTAS) or any(type(n) is not int or n <= 0 for n in quotas.values()):
        raise ValueError("Positive integer quota required for each explicit split")
    heaps = {split: [] for split in quotas}
    eligible = Counter()
    for ref in rows:
        if set(ref) != REFERENCE_FIELDS or ref["record_version"] != RECORD_VERSION:
            raise ValueError("Invalid audited reference schema")
        split = ref["split"]
        if split not in quotas or ref["phase"] not in ("opening", "middlegame", "endgame"):
            raise ValueError("Invalid split/phase")
        if ref["rule_position"].split()[1] not in ("w", "b"):
            raise ValueError("Invalid actual color")
        if key(ref) in excluded:
            continue
        eligible[split] += 1
        item, heap = _Worst(ref), heaps[split]
        if len(heap) < quotas[split]:
            heapq.heappush(heap, item)
        elif item.order < heap[0].order:
            heapq.heapreplace(heap, item)
    if any(len(heaps[s]) != quotas[s] for s in quotas):
        raise ValueError("Insufficient audited capacity; no quota reduction or refill")
    refs = [item.ref for split in QUOTAS for item in sorted(heaps[split], key=lambda x: x.order)]
    if len({key(r) for r in refs}) != len(refs):
        raise ValueError("Duplicate selected reference")
    if max(Counter(r["game_id"] for r in refs).values()) > 8:
        raise ValueError("More than eight selected references per game")
    positions = {s: {r["rule_position"] for r in refs if r["split"] == s} for s in QUOTAS}
    if any(positions[a] & positions[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Cross-split rule-position overlap")
    return refs, dict(eligible)


def replay_production_root(ref, game, manifest):
    """Replay real assignment, never relabel a held-out game as training."""
    validate_game(game, manifest)
    if (set(ref) != REFERENCE_FIELDS or ref["record_version"] != RECORD_VERSION
            or ref["game_id"] != game["game_id"] or ref["split"] != game["split"]
            or ref["split"] not in QUOTAS or type(ref["sample_ply"]) is not int
            or not 0 <= ref["sample_ply"] <= len(game["moves_uci"])):
        raise ValueError("Production reference/game mismatch")
    state = make_state(game["start_fen"], game["moves_uci"][:ref["sample_ply"]], history_complete=True)
    if terminal_result(state) is not None:
        raise ValueError("Terminal production root")
    if rule_position(state.board) != ref["rule_position"] or phase(state.board, ref["sample_ply"]) != ref["phase"]:
        raise ValueError("Production root identity/phase mismatch")
    return state


def accounting(refs, labels, attempts, exclusions):
    finished = {key(r) for r in labels} | {key(r["reference"]) for r in exclusions}
    successes = dict(Counter(r["split"] for r in labels))
    pending = [r for r in refs if key(r) not in finished]
    return dict(successful_by_split={s: successes.get(s, 0) for s in QUOTAS},
                shortfall_by_split={s: QUOTAS[s] - successes.get(s, 0) for s in QUOTAS},
                pending_references=pending,
                failed_attempts=sum(a["status"] == "failed" for a in attempts),
                interrupted_attempts=sum(a["status"] == "interrupted" for a in attempts),
                retries=sum(a["attempt"] > 1 for a in attempts), excluded=len(exclusions))
