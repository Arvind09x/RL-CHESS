"""D09 label expansion: deterministic nested membership, compact attempt ledger, equivalence.

Membership order: every accepted D08 reference first (its order), then the
remaining validation quota, then test, then train, each in rank order with at
most `per_game_cap` references per game counting D08's. Any train prefix is the
deterministic selection for that smaller size.

Attempt records keep what the loader verifies (final principal row, sent UCI
history, engine id, effective options, policy evidence) and summarize the rest,
because the full D08 ledger is ~14 KB per attempt and the loader reads it whole.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

from chess_rl.canonical_games import load_json
from chess_rl.initial_labels import key
from chess_rl.position_sampling import RECORD_VERSION

SELECTION_VERSION = "d09-expansion-sha256-v1"
LEDGER_VERSION = "compact-attempt-ledger-v1"
SPLITS = ("validation", "test", "train")
REFERENCE_FIELDS = {"game_id", "phase", "record_version", "rule_position", "sample_ply", "split"}
EQUIVALENCE_EXCLUDED_LABEL_FIELDS = ("successful_attempt", "target_provenance")
EQUIVALENCE_EXCLUDED_SEARCH_FIELDS = ("time", "nps")


def read_rows(path):
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.strip():
                yield load_json(line)


def rank(ref, seed):
    text = json.dumps(dict(selection_version=SELECTION_VERSION, seed=seed, split=ref["split"],
                           game_id=ref["game_id"], sample_ply=ref["sample_ply"]),
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def select_membership(samples_path, prior_refs, quotas, *, seed, per_game_cap):
    """Return (ordered refs, capacity per split after the cap, eligible counts).

    prior_refs come first unchanged. Remaining references are ranked by sha256
    and walked in rank order; a reference is skipped when its game already holds
    per_game_cap selected references. Capacity counts the whole walk so a quota
    above capacity is refused before any engine work.
    """
    if set(quotas) != set(SPLITS) or any(type(n) is not int or n < 0 for n in quotas.values()):
        raise ValueError("quotas must name validation, test and train with nonnegative integers")
    if type(per_game_cap) is not int or per_game_cap < 1:
        raise ValueError("per_game_cap must be a positive integer")
    prior_keys = {key(r) for r in prior_refs}
    if len(prior_keys) != len(prior_refs):
        raise ValueError("duplicate prior reference")
    per_game = Counter(r["game_id"] for r in prior_refs)
    prior_by_split = Counter(r["split"] for r in prior_refs)
    for split in SPLITS:
        if prior_by_split[split] > quotas[split]:
            raise ValueError(f"{split} quota below the prior membership it must contain")
    ranked = {split: [] for split in SPLITS}
    eligible = Counter()
    for ref in read_rows(samples_path):
        if set(ref) != REFERENCE_FIELDS or ref["record_version"] != RECORD_VERSION or ref["split"] not in SPLITS:
            raise ValueError("invalid audited reference schema")
        if key(ref) in prior_keys:
            continue
        eligible[ref["split"]] += 1
        ranked[ref["split"]].append((rank(ref, seed), ref["game_id"], ref["sample_ply"]))
    selected_keys, capacity = {split: [] for split in SPLITS}, {}
    for split in SPLITS:
        ranked[split].sort()
        # The full walk is computed so capacity is exact and every quota is a
        # prefix of the same walk (nested selections for any smaller size).
        walk = []
        for _, game_id, ply in ranked[split]:
            if per_game[game_id] >= per_game_cap:
                continue
            per_game[game_id] += 1
            walk.append((game_id, ply))
        capacity[split] = prior_by_split[split] + len(walk)
        kept = walk[:quotas[split] - prior_by_split[split]]
        if len(kept) < quotas[split] - prior_by_split[split]:
            raise ValueError(f"{split} quota {quotas[split]} exceeds capacity {capacity[split]} at cap {per_game_cap}")
        selected_keys[split] = kept
    wanted = {k for split in SPLITS for k in selected_keys[split]}
    rows = {}
    for ref in read_rows(samples_path):
        k = key(ref)
        if k in wanted:
            if k in rows:
                raise ValueError("duplicate audited sample")
            rows[k] = ref
    if set(rows) != wanted:
        raise ValueError("selected reference missing from the audited table")
    refs = list(prior_refs)
    for split in SPLITS:
        refs.extend(rows[k] for k in selected_keys[split])
    positions = {split: {r["rule_position"] for r in refs if r["split"] == split} for split in SPLITS}
    if (positions["train"] & positions["validation"]) or (positions["train"] & positions["test"]) \
            or (positions["validation"] & positions["test"]):
        raise ValueError("cross-split rule-position overlap")
    if max(Counter(r["game_id"] for r in refs).values()) > per_game_cap:
        raise ValueError("per-game cap violated")
    return refs, capacity, dict(eligible)


def compact_attempt(record):
    """Keep the fields the loader verifies; summarize bulky transport captures."""
    returned = record.get("returned")
    if not isinstance(returned, dict):
        return record
    compact = dict(returned)
    rows = compact.get("returned_rows")
    if isinstance(rows, list):
        principal = [r for r in rows if isinstance(r, dict) and r.get("multipv") == 1]
        compact["returned_rows"] = principal[-1:] if principal else []
        compact["returned_rows_total"] = len(rows)
    for name in ("uci_received", "option_capabilities"):
        if name in compact:
            text = json.dumps(compact.pop(name), sort_keys=True, separators=(",", ":"))
            compact[f"{name}_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            compact[f"{name}_bytes"] = len(text)
    out = dict(record, returned=compact)
    out["ledger"] = LEDGER_VERSION
    return out


def label_differences(left, right):
    """Field paths on which two labels of one reference differ under the frozen criterion."""
    paths = []
    keys = (set(left) | set(right)) - set(EQUIVALENCE_EXCLUDED_LABEL_FIELDS)
    for name in sorted(keys):
        if name == "returned_search":
            a, b = left.get(name) or {}, right.get(name) or {}
            for field in sorted((set(a) | set(b)) - set(EQUIVALENCE_EXCLUDED_SEARCH_FIELDS)):
                if a.get(field) != b.get(field):
                    paths.append(f"returned_search.{field}")
        elif left.get(name) != right.get(name):
            paths.append(name)
    return paths


def compare_labels(left_rows, right_rows):
    """Summary of equivalence over references labelled by both sides."""
    left = {key(r): r for r in left_rows}
    right = {key(r): r for r in right_rows}
    common = sorted(set(left) & set(right))
    mismatched, fields = [], Counter()
    for k in common:
        paths = label_differences(left[k], right[k])
        if paths:
            mismatched.append(dict(game_id=k[0], sample_ply=k[1], fields=paths))
            fields.update(paths)
    return dict(criterion=dict(excluded_label_fields=list(EQUIVALENCE_EXCLUDED_LABEL_FIELDS),
                               excluded_search_fields=list(EQUIVALENCE_EXCLUDED_SEARCH_FIELDS)),
                compared=len(common), matched=len(common) - len(mismatched),
                mismatched=len(mismatched), mismatched_fields=dict(fields),
                examples=mismatched[:20], left_only=len(set(left) - set(right)),
                right_only=len(set(right) - set(left)))


def completing_records(segments):
    """Per reference, the attempt/label/exclusion records of the segment that completed it.

    segments: ordered list of dicts with 'attempts', 'labels', 'exclusions'
    row lists. Records from earlier segments for a reference completed later
    are discarded (they describe interrupted work) and counted.
    """
    labels, exclusions, attempts = {}, {}, defaultdict(list)
    discarded = 0
    for segment in segments:
        done_here = set()
        for row in segment["labels"]:
            k = key(row)
            if k in labels or k in exclusions:
                raise ValueError(f"reference completed twice: {k}")
            labels[k] = row
            done_here.add(k)
        for row in segment["exclusions"]:
            k = key(row["reference"])
            if k in labels or k in exclusions:
                raise ValueError(f"reference completed twice: {k}")
            exclusions[k] = row
            done_here.add(k)
        for row in segment["attempts"]:
            k = key(row["reference"])
            if k in done_here:
                attempts[k].append(row)
            elif k in labels or k in exclusions:
                raise ValueError("attempt recorded after completion")
            else:
                discarded += 1
    for k, rows in attempts.items():
        if [r["attempt"] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"attempt sequence broken for {k}")
    return labels, exclusions, attempts, discarded
