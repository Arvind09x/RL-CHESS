"""D04 rule-position sampling; source results never enter this module's targets.

The accepted adapter owns legality/history and actual-occurrence terminal rules.
Bulk candidate pools are compressed per-game SQLite records outside Git.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import chess

from chess_rl.adapter import make_state, terminal_result
from chess_rl.adapter.state import require_legal_move
from chess_rl.canonical_games import (SPLITS, assign_split, canonical_json,
                                      canonical_payload, game_id, load_json)
from chess_rl.pgn_replay import file_identity, json_line

SAMPLING_VERSION = "phase-spread-sha256-v1"
PHASE_VERSION = "material-ply-v1"
RECORD_VERSION = "position-reference-v1"
SEED = 20261009
PHASES = ("opening", "middlegame", "endgame")
PRIORITY = ("test", "validation", "train")
PHASE_DESCRIPTION = (
    "material-ply-v1: endgame first when combined nonpawn/nonking material "
    "(N=3,B=3,R=5,Q=9) <=14; otherwise opening at sample_ply<20 from declared "
    "start; otherwise middlegame; counters and source results do not set phase"
)
SAMPLING_DESCRIPTION = (
    "phase-spread-sha256-v1: seed=20261009; eligible roots 0..mainline length "
    "strictly before first actual-occurrence-v1 terminal; earliest occurrence "
    "per four-field legal-EP rule position per game; missing phases in "
    "opening/middlegame/endgame order choose nearest phase ply midpoint; then "
    "maximize minimum ply distance from retained roots; ties by SHA256 compact "
    "sorted ASCII JSON {game_id,sample_ply,sampling_version,seed}, then ply; "
    "limit eight; test>validation>train ownership of selected positions; remove "
    "lower-split overlaps then refill using same selector with survivors locked; "
    "freeze heldout game assignments and sample registry; expansion preserves "
    "all existing samples and adds training only, never protected rule positions"
)


def rule_position(board):
    """Exact identity, not a digest: actual pieces/turn/castling/legal EP.

    Counters, repetitions and network history deliberately do not distinguish it.
    No claim is made that these histories have identical terminal/teacher values.
    """
    return " ".join(board.fen(en_passant="legal").split()[:4])


def phase(board, ply):
    material = sum(value * len(board.pieces(piece, color))
                   for piece, value in ((chess.KNIGHT, 3), (chess.BISHOP, 3),
                                        (chess.ROOK, 5), (chess.QUEEN, 9))
                   for color in chess.COLORS)
    return "endgame" if material <= 14 else "opening" if ply < 20 else "middlegame"


def tie(identity, ply):
    return hashlib.sha256(canonical_json(dict(game_id=identity, sample_ply=ply,
        sampling_version=SAMPLING_VERSION, seed=SEED)).encode()).digest(), ply


def select(identity, candidates, blocked=frozenset(), locked=()):
    """Keep survivors unchanged, cover available phases, then farthest progress.

    Candidate triples are (ply, phase, exact rule identity), earliest per game.
    Same-split sharing is allowed; within-game sharing is not.
    """
    chosen = list(locked)
    if len(chosen) > 8 or len({c[2] for c in chosen}) != len(chosen):
        raise ValueError("Invalid locked samples")
    if any(c not in candidates or c[2] in blocked for c in chosen):
        raise ValueError("Locked sample missing or protected overlap")
    remaining = [c for c in candidates if c not in chosen and c[2] not in blocked]
    ties = {c[0]: tie(identity, c[0]) for c in remaining}
    for name in PHASES:
        if len(chosen) == 8:
            break
        group = [c for c in remaining if c[1] == name]
        if group and not any(c[1] == name for c in chosen):
            center2 = min(c[0] for c in group) + max(c[0] for c in group)
            item = min(group, key=lambda c: (abs(2*c[0] - center2), ties[c[0]]))
            chosen.append(item)
            remaining.remove(item)
    while remaining and len(chosen) < 8:
        item = min(remaining, key=lambda c: (
            -min(abs(c[0] - s[0]) for s in chosen) if chosen else 0, ties[c[0]]))
        chosen.append(item)
        remaining.remove(item)
    return sorted(chosen)


def candidates_for_game(game):
    """Replay every move legally with the shared adapter, retaining full stack.

    Historical moves after policy termination are still legally validated, but
    their roots are excluded permanently; source results do not adjudicate.
    """
    state = make_state(game["start_fen"], history_complete=True)
    seen, candidates, counts = set(), [], Counter(roots=0, terminal_roots=0,
                                                  post_terminal_roots=0, repeated_roots=0)
    ended = False
    for ply in range(len(game["moves_uci"]) + 1):
        counts["roots"] += 1
        if ended:
            counts["post_terminal_roots"] += 1
        elif terminal_result(state) is not None:
            ended = True
            counts["terminal_roots"] += 1
        else:
            key = rule_position(state.board)
            if key in seen:
                counts["repeated_roots"] += 1
            else:
                seen.add(key)
                candidates.append((ply, phase(state.board, ply), key))
        if ply < len(game["moves_uci"]):
            move = chess.Move.from_uci(game["moves_uci"][ply])
            require_legal_move(state.board, move)
            state.board.push(move)
    return candidates, dict(counts)


def validate_game(game, manifest):
    for key in ("dataset_id", "encoding_version", "move_vocab_version", "ending_policy"):
        if game.get(key) != getattr(manifest, key):
            raise ValueError(f"Canonical game {key} mismatch")
    if (game.get("history_complete") is not True
            or game.get("history_origin") != "declared-pgn-start"
            or game.get("record_version") != "replay-jsonl-v1"
            or game.get("wdl_order") != list(manifest.wdl_order)):
        raise ValueError("Canonical game history/version mismatch")
    payload = canonical_payload(game["start_fen"], game["moves_uci"])
    if game_id(payload) != game["game_id"] or assign_split(game["game_id"], manifest.split_seed) != game["split"]:
        raise ValueError("Canonical game identity/assignment mismatch")


def _worker(game):
    candidates, counts = candidates_for_game(game)
    return game["game_id"], game["split"], zlib.compress(json_line(candidates).encode()), counts


def sample_record(identity, split, candidate):
    ply, name, key = candidate
    return dict(record_version=RECORD_VERSION, game_id=identity, split=split,
                sample_ply=ply, phase=name, rule_position=key)


def verify_intake(games, audit_path, expected_sha256, manifest):
    audit_identity = file_identity(audit_path)
    if audit_identity["sha256"] != expected_sha256 or expected_sha256 != manifest.preparation.split_audit_sha256:
        raise ValueError("Accepted split audit SHA-256 mismatch")
    audit = load_json(Path(audit_path).read_bytes())
    if (audit.get("complete") is not True or audit.get("audit_version") != "canonical-split-audit-v1"
            or audit.get("dataset_id") != manifest.dataset_id
            or audit.get("split_seed") != manifest.split_seed
            or audit.get("split_algorithm") != manifest.preparation.split_algorithm
            or audit.get("archive") != dict(sha256=manifest.archive.downloaded_sha256,
                                           bytes=manifest.archive.downloaded_bytes)):
        raise ValueError("Accepted split audit contract mismatch")
    identity = file_identity(games)
    if identity != audit["outputs"]["games.jsonl"]:
        raise ValueError("Accepted canonical games identity mismatch")
    return audit, {"games.jsonl": identity, "split-audit.json": audit_identity}


def prepare_positions(games, audit_path, expected_sha256, manifest, output_dir, *, workers=1, progress=None):
    if manifest.sampling_seed != SEED or manifest.max_distinct_nonterminal_positions_per_game != 8:
        raise ValueError("D04 seed/limit mismatch")
    for value, expected in ((manifest.preparation.sampling_algorithm, SAMPLING_DESCRIPTION),
                            (manifest.preparation.phase_boundaries, PHASE_DESCRIPTION)):
        if value not in (None, expected):
            raise ValueError("Unsupported D04 decision")
    audit, inputs = verify_intake(games, audit_path, expected_sha256, manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(out / "candidate-pools.sqlite")
    db.execute("CREATE TABLE pools (game_id TEXT PRIMARY KEY, split TEXT, candidates BLOB)")
    counts = Counter()
    game_counts, ply_counts, source_counts = Counter(), Counter(), Counter()
    observed = hashlib.sha256()
    observed_bytes = 0

    def records():
        nonlocal observed_bytes
        with Path(games).open("rb") as handle:
            for raw in handle:
                observed.update(raw)
                observed_bytes += len(raw)
                game = load_json(raw)
                validate_game(game, manifest)
                game_counts[game["split"]] += 1
                ply_counts[game["split"]] += len(game["moves_uci"])
                source_counts[game["split"]] += len(game["sources"])
                yield game

    # Bounded batches avoid executor.map eagerly loading the complete corpus.
    from itertools import islice
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = records()
        processed = 0
        while batch := list(islice(iterator, 512)):
            for identity, split, blob, root_counts in executor.map(_worker, batch, chunksize=8):
                db.execute("INSERT INTO pools VALUES (?,?,?)", (identity, split, blob))
                counts.update(root_counts)
                processed += 1
            db.commit()
            if progress and processed % 5120 == 0:
                progress(processed)
    if dict(sha256=observed.hexdigest(), bytes=observed_bytes) != inputs["games.jsonl"]:
        raise ValueError("Canonical input changed during replay")
    if any(game_counts[s] != audit["actual_split_counts"][s]["games"] or
           ply_counts[s] != audit["actual_split_counts"][s]["plies"] or
           source_counts[s] != audit["actual_split_counts"][s]["sources"] for s in SPLITS):
        raise ValueError("Canonical split counts changed")
    assignments = hashlib.sha256()
    for identity, split in db.execute("SELECT game_id,split FROM pools ORDER BY game_id"):
        assignments.update(json_line(dict(game_id=identity, split=split)).encode())
    if assignments.hexdigest() != audit["assignments_sha256"]:
        raise ValueError("Canonical assignment checksum mismatch")
    db.execute("CREATE INDEX by_split ON pools(split,game_id)")
    db.commit()
    report = select_corpus(db, out, progress=progress)
    if file_identity(audit_path) != inputs["split-audit.json"] or file_identity(games) != inputs["games.jsonl"]:
        raise ValueError("Input artifacts changed before completion")
    report.update(audit_version="position-overlap-audit-v1", complete=True,
                  dataset_id=manifest.dataset_id, sampling_version=SAMPLING_VERSION,
                  sampling_seed=SEED, sampling_algorithm=SAMPLING_DESCRIPTION,
                  phase_boundaries=PHASE_DESCRIPTION, overlap_identity="rule-position-disjoint-v1",
                  inputs=inputs, assignments_sha256=assignments.hexdigest(),
                  replay_counts=dict(counts), assigned_games=dict(game_counts),
                  outputs={name: file_identity(out / name) for name in
                           ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl")})
    (out / "overlap-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    db.close()
    return report


def select_corpus(db, out, *, progress=None):
    """Priority passes reserve selected heldouts before lower splits refill."""
    db.execute("CREATE TABLE selected (game_id TEXT, split TEXT, sample_ply INTEGER, phase TEXT, rule_position TEXT, PRIMARY KEY(game_id,sample_ply))")
    stats = {}
    protected = set()
    with (out / "exclusions.jsonl").open("w") as exclusions:
        for split in PRIORITY:
            counters = Counter(candidate_positions=0, initial_samples=0, removed=0,
                               refilled=0, retained=0, games_covered=0, games_empty=0,
                               blocked_candidate_positions=0)
            before, after, pool_phases, owners = Counter(), Counter(), Counter(), Counter()
            own_keys = set()
            for identity, blob in db.execute("SELECT game_id,candidates FROM pools WHERE split=? ORDER BY game_id", (split,)):
                pool = [tuple(c) for c in json.loads(zlib.decompress(blob))]
                initial = select(identity, pool)
                survivors = [c for c in initial if c[2] not in protected]
                retained = select(identity, pool, protected, survivors)
                removed = [c for c in initial if c[2] in protected]
                counters.update(candidate_positions=len(pool), initial_samples=len(initial),
                                removed=len(removed), refilled=len(retained)-len(survivors),
                                retained=len(retained), games_covered=bool(retained), games_empty=not retained,
                                blocked_candidate_positions=sum(c[2] in protected for c in pool))
                before.update(c[1] for c in initial)
                after.update(c[1] for c in retained)
                pool_phases.update(c[1] for c in pool)
                for c in removed:
                    owner = db.execute("SELECT split FROM selected WHERE rule_position=? LIMIT 1", (c[2],)).fetchone()[0]
                    owners[owner] += 1
                    exclusions.write(json_line(dict(**sample_record(identity, split, c), owner=owner)))
                db.executemany("INSERT INTO selected VALUES (?,?,?,?,?)",
                               [(identity, split, *c) for c in retained])
                own_keys.update(c[2] for c in retained)
            db.commit()
            # Index is needed for exclusions lookup; selected grows only by split.
            db.execute("CREATE INDEX IF NOT EXISTS selected_key ON selected(rule_position)")
            protected.update(own_keys) if split != "train" else None
            stats[split] = dict(counters, phase_candidates={p: pool_phases[p] for p in PHASES},
                               phase_before={p: before[p] for p in PHASES},
                               phase_after={p: after[p] for p in PHASES},
                               exclusions_by_owner=dict(owners), distinct_rule_positions=len(own_keys))
            if progress:
                progress(f"selected {split}: {counters['retained']}")
    with (out / "samples.jsonl").open("w") as samples, (out / "protected-heldout.jsonl").open("w") as heldouts:
        for identity, split, ply, name, key in db.execute("SELECT * FROM selected ORDER BY game_id,sample_ply"):
            raw = json_line(sample_record(identity, split, (ply, name, key)))
            samples.write(raw)
            if split != "train":
                heldouts.write(raw)
    with (out / "heldout-games.jsonl").open("w") as handle:
        for identity, split in db.execute("SELECT game_id,split FROM pools WHERE split!='train' ORDER BY game_id"):
            handle.write(json_line(dict(game_id=identity, split=split)))
    overlaps = db.execute("SELECT count(*) FROM (SELECT rule_position FROM selected GROUP BY rule_position HAVING count(DISTINCT split)>1)").fetchone()[0]
    if overlaps:
        raise ValueError("Cross-split overlap survived")
    return dict(splits=stats, cross_split_overlaps_after=overlaps,
                capacity_records=sum(s["retained"] for s in stats.values()),
                protected_rule_positions=len(protected),
                initial_10000_quota={"train": 9000, "validation": 500, "test": 500},
                labels_produced=0, frozen_registry_version="protected-heldout-v1")


def expand_training(candidates_by_game, existing_samples, heldout_registry, assignments):
    """Append only training records; never replace any prior retained reference.

    Inputs must be joined/replayed and audited by the caller. This deterministic
    primitive intentionally does not accept new heldout samples. Persisted
    assignment mapping must include every existing game, including zero samples.
    """
    heldout_registry = list(heldout_registry)
    protected = {r["rule_position"] for r in heldout_registry}
    existing = list(existing_samples)
    if any(r["split"] != assignments.get(r["game_id"]) for r in existing + list(heldout_registry)):
        raise ValueError("Existing assignment changed")
    if {canonical_json(r) for r in existing if r["split"] != "train"} != {canonical_json(r) for r in heldout_registry}:
        raise ValueError("Protected heldout samples changed")
    additions = []
    for identity in sorted(candidates_by_game):
        if identity not in assignments:
            raise ValueError("Missing frozen or new canonical assignment")
        if assignments[identity] != "train":
            continue
        pool = [tuple(c) for c in candidates_by_game[identity]]
        prior = [r for r in existing if r["game_id"] == identity]
        locked = [(r["sample_ply"], r["phase"], r["rule_position"]) for r in prior]
        selected = select(identity, pool, protected, locked)
        additions.extend(sample_record(identity, "train", c) for c in selected if c not in locked)
    result = sorted(existing + additions, key=lambda r: (r["game_id"], r["sample_ply"]))
    if any(r["split"] == "train" and r["rule_position"] in protected for r in result):
        raise ValueError("Training collides with protected heldout")
    return result
