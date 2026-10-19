"""One resumable match driver for the smoke, validation, rung and test matches.

A match is defined by its players, opening split, pair count and search settings.
Everything a game depends on is frozen before the first game:

- ``openings.json`` (written once): 64 validation and 64 test roots taken only from
  the protected held-out registry, plus the two checked smoke openings.
- ``candidates.json`` (written once): P, the round-1 checkpoints that may play
  validation (each marked selectable or report-only), the excluded runs and the
  selection rules.
- one ``plan.json`` per match (written once, digest protected) with players,
  checkpoint identities, openings, seeds, settings, ply cap, workers, runtime and
  evaluation source hashes, and the full schedule.

Results are one immutable SQLite row per game, committed by the worker before it
reports, and every stored game is re-validated by replay on resume and report.
Every score used for selection, the demo rule or a claim is R's score (the
plan's ``score_player_id``). Reports of P-vs-R and rung matches refuse
unfinished schedules, so no observed score can be used to stop a match early;
the smoke report shows only statuses, lengths and timing.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import sqlite3
import statistics
import tempfile
import time

import chess

from chess_rl.adapter import apply_move, id_to_move, make_state, terminal_result
from chess_rl.position_sampling import rule_position
from .matches import Opening, OpeningSets, digest, play_game, snapshot, state_trace, summarize, turn_seed
from .players import (Checkpoint, GreedyPlayer, RandomLegalPlayer, SearchPlayer, SearchSettings,
                      StockfishPin, StockfishPlayer)
from . import pilot

OPENINGS_SCHEMA = "match-openings-v1"
CANDIDATES_SCHEMA = "match-candidates-v1"
PLAN_SCHEMA = "match-plan-v1"
SELECTION_SCHEMA = "match-selection-v1"
# "policy": the no-search diagnostic (1 simulation = the raw policy's top move); never selection or claims.
# "head-to-head": two frozen candidates against each other (report-only, never selection or claims).
KINDS = ("smoke", "validation", "rung", "test", "policy", "head-to-head")
FIELDS = ("opening_id", "initial_fen", "moves_uci", "history_complete", "source")

# Frozen evaluation protocol (match-driver plan, decisions 1-8 approved 17 Oct 2026).
SIMULATIONS = 128
POLICY_SIMULATIONS = 1
PLY_CAP = 512
WORKERS = 5
OPENINGS_PER_SPLIT = 64
PLY_BAND = (6, 12)
SMOKE_OPENING_IDS = ("italian-1", "italian-2")
RUNG_PAIRS = 16
PROVISIONAL_MIN_PAIRS = 48
FINAL_PAIRS = 64
SEEDS = dict(smoke=2026101801, validation=2026101802, rung=2026101803, test=2026101804, policy=2026101806,
             **{"head-to-head": 2026101807})
BOOTSTRAP = dict(samples=10000, seed=2026101805)
HEAD_TO_HEAD_RULE = ("head-to-head diagnostic: two frozen round candidates play each other on the 64 validation "
                     "openings with paired colours and identical settings (128 simulations, or 1 = the raw policy's "
                     "top move); the score is the first-named candidate's; report-only: never read by selection, "
                     "never used for a claim, never on test openings")
POLICY_RULE = ("no-search diagnostic: 1 simulation per move; with no visits every move scores c_puct x prior "
               "(unvisited value 0), so the single visit and the move played are the highest-prior legal move "
               "(ties: smallest move id), i.e. the raw policy's top move; the value head never chooses a move; "
               "both players identical settings; validation openings only; report-only: never read by "
               "selection, never used for a claim")
OPENING_RANK = ("rank = sha256 of compact sorted JSON [\"match-openings-v1\", split, game_id, sample_ply]; "
                "ascending; first row per distinct rule position kept")
OPENING_RULE = (
    "roots only from protected-heldout.jsonl rows of the split (validation roots from validation rows, "
    "test roots from test rows), sample_ply in [6, 12]; the row's game must be in heldout-games.jsonl and "
    "games.jsonl under the same split, start from the standard position with history_complete true; the "
    "opening is the game's start FEN plus every move before sample_ply (history_complete true); the "
    "replayed root must equal the row's rule_position, be nonterminal, appear in the registry under this "
    "split only, and differ from every smoke opening; at most one root per game; 64 per split in rank order")
INCOMPLETE_RULE = (
    "a failed or interrupted game in a selectable candidate's match blocks selection until diagnosed; no game "
    "is ever replaced or replayed under "
    "a different plan; a truncation at the ply cap is an unscored outcome; candidates are compared only on "
    "the openings whose pairs are complete (both legs scored) in every selectable candidate's validation "
    "match (a report-only match never blocks, shrinks or changes the selection); at least 48 common "
    "complete pairs are needed for a provisional report and all 64 for the final selection; matches are "
    "never stopped on observed scores")
SELECTION_RULE = (
    "R = the selectable candidate with the higher validation score (R's score against P on the common "
    "complete pairs); report-only candidates are reported and never selected; a tie goes to the first "
    "candidate in tie order; validation never supports an improvement claim (selection biases the "
    "selected interval upward)")
DEMO_RULE = "the demo uses R if R's final validation score > 0.5, else P (a selection rule, not a claim)"
CLAIM_RULE = ("R improves over P only if the lower bound of R's test pair-bootstrap interval is above 0.5; an "
              "interval entirely below 0.5 indicates regression; an interval containing 0.5 supports neither")
EVALUATION_SOURCES = ("src/chess_rl/adapter", "src/chess_rl/search", "src/chess_rl/evaluation",
                      "src/chess_rl/model", "src/chess_rl/training/checkpoints.py", "src/chess_rl/contracts.py",
                      "src/chess_rl/tensor_ops.py", "src/chess_rl/runtime.py",
                      "src/chess_rl/position_sampling.py", "scripts/run_match.py")
RUNG_OPPONENTS = ("random-legal", "material-greedy", "stockfish-1k", "stockfish-10k")
STOCKFISH_NODES = {"stockfish-1k": 1000, "stockfish-10k": 10000}
# The engine that labelled the data (teacher-evidence.json); rungs must use these exact bytes.
STOCKFISH_PIN = dict(sha256="3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4",
                     uci_name="Stockfish 17.1", release="sf_17.1",
                     source_commit="03e27488f3d21d8ff4dbf3065603afa21dbd0ef3")

# Round-1 candidates as decided by Arvind on 17 Oct 2026 (23:53 IST), before any validation game.
ROUND1_CANDIDATES = dict(
    P=dict(path="artifacts/p-1m/step-090000.pt",
           sha256="cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2"),
    candidates=(
        dict(name="R1-mix", path="runs/train/r1-mix/update-002303.pt",
             sha256="32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2",
             summary="runs/train/r1-mix/summary.json", recipe="mix", eligibility="selectable",
             note="one pass, original forgetting rule kept (flag fired at +0.181, below the +0.25 stop)"),
        dict(name="R1-pure-full", path="runs/train/r1-pure-full/update-002303.pt",
             sha256="989da5b16be1f3afaba921e38520670a0f19d84d8bbeaf010fcb9bb3947d560d",
             summary="runs/train/r1-pure-full/summary.json", recipe="pure", eligibility="report-only",
             note=("exploratory under amendment A6 (hard stop not enforced); the original forgetting rule is "
                   "kept, under which R1-pure's candidate is P; plays validation for the report only"))),
    excluded=(
        dict(name="R1-mix-2pass", path="runs/train/r1-mix-2pass", reason="diagnostic second pass (amendment A7)"),
        dict(name="R1-pure-full-2pass", path="runs/train/r1-pure-full-2pass",
             reason="diagnostic second pass (amendment A7)")),
    tie_order=("R1-mix",),
    decided="Arvind, 17 Oct 2026 23:53 IST, before any validation game")


# Files.

def sha_file(path):
    return pilot.sha_file(path)


def publish_once(path, value):
    """Atomic no-clobber JSON publication: refuses an existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"{path} already exists; frozen records are never rewritten")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise ValueError(f"{path} already exists; frozen records are never rewritten") from exc
    finally:
        os.unlink(temporary)


def sealed(content):
    return dict(**content, sha256=digest(content))


def read_sealed(path, schema):
    value = json.loads(Path(path).read_text())
    content = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != digest(content) or value.get("schema") != schema:
        raise ValueError(f"{path}: digest or schema mismatch")
    return value


def identity(path):
    path = Path(path)
    return dict(sha256=sha_file(path), bytes=path.stat().st_size)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def opening(raw):
    return Opening(**{key: (tuple(raw[key]) if key == "moves_uci" else raw[key]) for key in FIELDS})


# Openings.

def opening_rank(split, game_id, sample_ply):
    return digest([OPENINGS_SCHEMA, split, game_id, sample_ply])


def smoke_openings():
    """The two checked smoke openings: authored development lines that are not registry positions."""
    authored = {o["opening_id"]: o for o in pilot.authored_openings()}
    return [authored[name] for name in SMOKE_OPENING_IDS]


def freeze_openings(output, *, registry_dir, games_path, manifest_path, count=OPENINGS_PER_SPLIT,
                    ply_band=PLY_BAND, log=print):
    """Select, verify and publish the validation and test openings once."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} already exists; the opening freeze is never rewritten")
    registry_dir = Path(registry_dir)
    registry_path, games_registry_path = registry_dir / "protected-heldout.jsonl", registry_dir / "heldout-games.jsonl"
    manifest = json.loads(Path(manifest_path).read_text())
    registry_identity = identity(registry_path)
    if registry_identity["sha256"] != manifest["preparation"]["protected_heldout_registry_sha256"]:
        raise ValueError("protected-heldout.jsonl differs from the dataset manifest")
    held_games = {}
    for row in _rows(games_registry_path):
        held_games[row["game_id"]] = row["split"]
    splits_of = {}
    candidates = {"validation": [], "test": []}
    for row in _rows(registry_path):
        splits_of.setdefault(row["rule_position"], set()).add(row["split"])
        if row["split"] in candidates and ply_band[0] <= row["sample_ply"] <= ply_band[1]:
            candidates[row["split"]].append(row)
    smoke = smoke_openings()
    smoke_keys = {opening(o).rule_key() for o in smoke}
    if smoke_keys & set(splits_of):
        raise ValueError("a smoke opening is a protected held-out position")
    for split in candidates:
        candidates[split].sort(key=lambda r: opening_rank(split, r["game_id"], r["sample_ply"]))
    # Read only the games that can be needed: every candidate up to a generous rank horizon.
    horizon = max(20 * count, 200)
    wanted = {r["game_id"] for split in candidates for r in candidates[split][:horizon]}
    games = {}
    for game in _rows(games_path):
        if game["game_id"] in wanted:
            if game["game_id"] in games:
                raise ValueError("duplicate canonical game")
            games[game["game_id"]] = game
    chosen = {}
    rejected = {}
    for split, rows in candidates.items():
        picked, keys, used_games, reasons = [], set(), set(), {}

        def reject(reason):
            reasons[reason] = reasons.get(reason, 0) + 1

        for row in rows[:horizon]:
            if len(picked) == count:
                break
            game = games.get(row["game_id"])
            if game is None or held_games.get(row["game_id"]) != split or game.get("split") != split:
                reject("game not held out under this split")
                continue
            if game["start_fen"] != chess.STARTING_FEN or game["history_complete"] is not True:
                reject("nonstandard start or incomplete history")
                continue
            if splits_of[row["rule_position"]] != {split}:
                reject("rule position registered under another split")
                continue
            moves = game["moves_uci"][:row["sample_ply"]]
            if len(moves) != row["sample_ply"]:
                reject("game shorter than sample ply")
                continue
            state = make_state(game["start_fen"], moves, history_complete=True)
            if rule_position(state.board) != row["rule_position"]:
                raise ValueError(f"registry row does not replay to its rule position: {row['game_id']}")
            if terminal_result(state) is not None:
                reject("terminal root")
                continue
            if row["rule_position"] in keys:
                reject("duplicate root")
                continue
            if row["rule_position"] in smoke_keys:
                reject("smoke root")
                continue
            if row["game_id"] in used_games:
                # Two roots of one game would be nested lines of one history: correlated pairs.
                reject("game already supplied a root")
                continue
            ident = f"{split}-{len(picked) + 1:02d}"
            o = Opening(ident, game["start_fen"], tuple(moves), True,
                        f"protected-heldout:{split}:{row['game_id']}:{row['sample_ply']}")
            keys.add(row["rule_position"])
            used_games.add(row["game_id"])
            picked.append((o, dict(row, rank=opening_rank(split, row["game_id"], row["sample_ply"]))))
        if len(picked) != count:
            raise ValueError(f"only {len(picked)} {split} openings satisfy the rule")
        chosen[split] = picked
        rejected[split] = reasons
    sets = OpeningSets("protected-heldout-plies-6-12-v1", tuple(o for o, _ in chosen["validation"]),
                       tuple(o for o, _ in chosen["test"]))
    content = dict(schema=OPENINGS_SCHEMA, rule=OPENING_RULE, rank=OPENING_RANK, ply_band=list(ply_band),
                   per_split=count, sets=sets.to_dict(),
                   registry_rows={split: [row for _, row in chosen[split]] for split in chosen},
                   rejected_before_selection=rejected,
                   smoke=dict(openings=smoke, rule=("authored development openings whose roots are not in the "
                              "protected registry under any split and differ from every validation/test root")),
                   inputs={"protected-heldout.jsonl": registry_identity,
                           "heldout-games.jsonl": identity(games_registry_path),
                           "games.jsonl": identity(games_path), "manifest.json": identity(manifest_path)},
                   created_utc=_now())
    publish_once(output, sealed(content))
    log(json.dumps(dict(event="openings-frozen", path=str(output), validation=count, test=count,
                        set_sha256=content["sets"]["sha256"], rejected=rejected)))
    return load_openings(output)


def load_openings(path):
    """Verify the frozen opening record and rebuild its sets."""
    value = read_sealed(path, OPENINGS_SCHEMA)
    raw = value["sets"]
    sets = OpeningSets(raw["set_id"], tuple(opening(o) for o in raw["validation"]),
                       tuple(opening(o) for o in raw["test"]))
    if snapshot(sets.to_dict()) != snapshot(raw):
        raise ValueError("opening sets do not rebuild to the frozen record")
    smoke = [opening(o) for o in value["smoke"]["openings"]]
    keys = {o.rule_key() for o in sets.validation + sets.test}
    if any(o.rule_key() in keys for o in smoke) or len({o.rule_key() for o in smoke}) != len(smoke):
        raise ValueError("smoke openings must differ from each other and from every validation/test root")
    for split in ("validation", "test"):
        for o, row in zip(getattr(sets, split), value["registry_rows"][split], strict=True):
            if row["split"] != split or o.rule_key() != row["rule_position"]:
                raise ValueError("opening does not match its registry row")
    return dict(record=value, sets=sets, smoke=smoke, path=str(Path(path).resolve()),
                file=identity(path))


# Candidates.

def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def freeze_candidates(output, *, root, spec=ROUND1_CANDIDATES, log=print):
    """Record P, the validation candidates (selectable or report-only) and the selection rules once."""
    from chess_rl.training.checkpoints import load_inference
    output, root = Path(output), Path(root).resolve()
    if output.exists():
        raise ValueError(f"{output} already exists; candidate eligibility is never rewritten")

    def checked(path, sha256):
        if sha_file(root / path) != sha256:
            raise ValueError(f"{path}: checkpoint bytes differ from the recorded sha256")
        loaded = load_inference(root / path)
        if sha_file(root / path) != sha256:
            raise ValueError(f"{path}: checkpoint changed while loading")
        return loaded

    p = checked(spec["P"]["path"], spec["P"]["sha256"])
    entries = []
    names = [c["name"] for c in spec["candidates"]]
    if len(set(names)) != len(names) or "P" in names:
        raise ValueError("candidate names must be unique and differ from P")
    for c in spec["candidates"]:
        if c["eligibility"] not in ("selectable", "report-only"):
            raise ValueError("eligibility must be selectable or report-only")
        loaded = checked(c["path"], c["sha256"])
        prov = loaded.provenance
        if prov.get("initial_checkpoint", {}).get("sha256") != spec["P"]["sha256"]:
            raise ValueError(f"{c['name']}: not trained from P")
        if prov.get("recipe") != c["recipe"]:
            raise ValueError(f"{c['name']}: recipe {prov.get('recipe')!r} differs from {c['recipe']!r}")
        exploratory = "exploratory" in prov or prov.get("thresholds", {}).get("hard_stop_enforced") is False
        if c["eligibility"] == "selectable" and (exploratory or "passes_over_selfplay" in prov):
            raise ValueError(f"{c['name']}: an exploratory or multi-pass run cannot be selectable")
        if c["eligibility"] == "report-only" and not exploratory:
            raise ValueError(f"{c['name']}: report-only is reserved for the exploratory run")
        summary = json.loads((root / c["summary"]).read_text())
        if (summary["candidate"]["sha256"] != c["sha256"]
                or summary["candidate"]["checkpoint"] != Path(c["path"]).name
                or bool(summary.get("exploratory", False)) != exploratory):
            raise ValueError(f"{c['name']}: run summary names another candidate")
        entries.append(dict(c, checkpoint_id=loaded.checkpoint_id, kind="rl-R",
                            provenance_sha256=digest(prov), exploratory=exploratory,
                            summary_identity=identity(root / c["summary"]),
                            summary_selection=summary["selection"]))
    if not any(e["eligibility"] == "selectable" for e in entries):
        raise ValueError("at least one candidate must be selectable")
    tie_order = list(spec["tie_order"])
    if [n for n in tie_order if n not in names] or any(
            dict(zip(names, (c["eligibility"] for c in spec["candidates"])))[n] != "selectable" for n in tie_order):
        raise ValueError("tie order may name only selectable candidates")
    content = dict(schema=CANDIDATES_SCHEMA,
                   P=dict(spec["P"], name="P", checkpoint_id=p.checkpoint_id, kind="pretrained-P",
                          provenance_sha256=digest(p.provenance)),
                   candidates=entries, excluded=[dict(e) for e in spec["excluded"]], tie_order=tie_order,
                   rules=dict(selection=SELECTION_RULE, incomplete=INCOMPLETE_RULE, demo=DEMO_RULE,
                              claim=CLAIM_RULE, provisional_min_pairs=PROVISIONAL_MIN_PAIRS,
                              final_pairs=FINAL_PAIRS, score="R's score against P (win 1, draw 0.5, loss 0)"),
                   decided=spec["decided"], created_utc=_now())
    publish_once(output, sealed(content))
    log(json.dumps(dict(event="candidates-frozen", path=str(output),
                        candidates={e["name"]: e["eligibility"] for e in entries})))
    return load_candidates(output)


def load_candidates(path):
    value = read_sealed(path, CANDIDATES_SCHEMA)
    return dict(record=value, path=str(Path(path).resolve()), file=identity(path),
                by_name={c["name"]: c for c in value["candidates"]})


# Round 2.
#
# Round 2 adds one candidate record and one selection record beside the round-1 ones; nothing of round 1 is
# rewritten. R1-mix keeps its sealed validation match (never replayed); R2-mix plays a new one. The two plans
# were frozen at different code revisions, so the round-2 selection compares their evaluation sources under
# an audited exception (CROSS_REVISION_EXCEPTION below): every file must hash identically except the three
# named files, and only when their hashes in the older plan equal the recorded round-1 values; the exception
# is admitted only with an evidence record produced by scripts/check_evaluation_equivalence.py showing that
# both revisions give byte-identical network outputs and search decisions for P and R1-mix.

ROUND_FILES = {1: dict(candidates="candidates.json", selection="selection.json", status="selection-status.json"),
               2: dict(candidates="candidates-round2.json", selection="selection-round2.json",
                       status="selection-round2-status.json")}
EQUIVALENCE_SCHEMA = "evaluation-equivalence-v1"
EQUIVALENCE_FILE = "evaluation-equivalence-round2.json"

# Round-2 candidates (round-2 plan, approved decisions 3 and 4 of 18 Oct 2026; training amendment of
# 19 Oct 2026 ~12:35 IST: 20/80 round-1/round-2 self-play sampling for both recipes, one pass over round 2).
ROUND2_CANDIDATES = dict(
    round=2,
    previous_round=dict(candidates="runs/matches/candidates.json", selection="runs/matches/selection.json"),
    P=ROUND1_CANDIDATES["P"],
    start=dict(name="R1-mix", path="runs/train/r1-mix/update-002303.pt",
               sha256="32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2"),
    candidates=(
        dict(name="R1-mix", carried_from_round=1, eligibility="selectable",
             validation_match="validation-P-vs-R1-mix",
             note="the round-1 selection; its sealed validation match is reused, never replayed"),
        dict(name="R2-mix", path="runs/train/r2-mix-617f26a9327a21e6/update-002610.pt",
             sha256="5e0a026bd688ce137ed11110c9729dac91e17477cc8504f3a208e893339db915",
             summary="runs/train/r2-mix-617f26a9327a21e6/summary.json", recipe="mix", eligibility="selectable",
             note=("round 2 from R1-mix on round-1 + round-2 self-play (20/80 sampling, 2610 updates = one pass "
                   "over round 2), original forgetting rule kept (flag fired, final delta +0.134 below the "
                   "+0.25 stop)")),
        dict(name="R1-pure-full", carried_from_round=1, eligibility="report-only",
             validation_match="validation-P-vs-R1-pure-full",
             note="round-1 exploratory run (amendment A6); its finished validation match is reported beside round 2"),
        dict(name="R2-pure", path="runs/train/r2-pure-617f26a9327a21e6/update-002610.pt",
             sha256="9cce66e2c23905a6ba0068d617a9e7d4dc2333624a7188a36dfd09a4fae47c14",
             summary="runs/train/r2-pure-617f26a9327a21e6/summary.json", recipe="pure", eligibility="report-only",
             note=("round 2 pure recipe from R1-mix, hard stop not enforced (approved decision 3, as A6): every "
                   "checkpoint exceeded +0.25 (final delta +0.765); the original rule's candidate is R1-mix; "
                   "plays validation for the report only"))),
    excluded=(
        dict(name="dryrun-r2-mix-617f26a9327a21e6", path="runs/train/dryrun-r2-mix-617f26a9327a21e6",
             reason="4-update dry run (resume and inference gate)"),
        dict(name="dryrun-r2-pure-617f26a9327a21e6", path="runs/train/dryrun-r2-pure-617f26a9327a21e6",
             reason="4-update dry run (resume and inference gate)"),
        *ROUND1_CANDIDATES["excluded"]),
    tie_order=("R1-mix", "R2-mix"),
    decided="Arvind, 18 Oct 2026 ~13:00 IST (round-2 plan, decisions 3-4) and 19 Oct 2026 ~15:50 IST, before any "
            "round-2 validation game")

# The audited cross-revision exception. Only these three evaluation sources may hash differently between a
# round-1 validation plan and the round-2 plans, and only when the round-1 plan pins exactly these hashes; the
# six listed files may appear in a round-2 plan and not in the round-1 plan (new modules under the evaluation
# source directories, none imported by the play path), at exactly these hashes; no pinned file may be missing
# from a round-2 plan. Anything else ends the reuse: R1-mix is rerun under the new revision.
CROSS_REVISION_EXCEPTION = dict(
    files={"scripts/run_match.py": "e2a6fee211e2e9e56a9aee3b482ebf193fd9febdf033f9f591292585e3e36273",
           "src/chess_rl/evaluation/match_campaign.py":
               "8d81888d2cdfa25b8f56aeb808b408b976c8547dcf208c2b25ff303750568488",
           "src/chess_rl/training/checkpoints.py":
               "f5dd6ff0bb9f5199c76fa43cfb8d7946453c717befe239eebae711ec82227af9"},
    added_files={"src/chess_rl/evaluation/batching_acceptance.py":
                     "410f9f342221be6faf054aa2018592933a70a790ac1df0eb7caf0bfe57d757df",
                 "src/chess_rl/evaluation/benchmark_layouts.py":
                     "404af82eee423c40dd584e7a55b7b5e940b89ab1e7f46bce18bf01a2183e2d53",
                 "src/chess_rl/evaluation/concurrent_matches.py":
                     "9414d58ba39b628af768db8aafe6de9db0962cd2550d6d4e3478d5447fa97fb1",
                 "src/chess_rl/evaluation/development_matches.py":
                     "ef80166b3fca081703c5059f3c0092e2ffdbb02fce31db197118217f5d33f6f9",
                 "src/chess_rl/evaluation/execution.py":
                     "20d335be98e73609747c81580ada881a16fa43454ff138f89c27b36a5612e5b0",
                 "src/chess_rl/search/batched_evaluator.py":
                     "2092ed24cab8bb1df4cf95688b68599659296ab527c9b1c7cc94daa088d544c3"},
    audit="docs/round2-match-audit.md",
    rationale=("scripts/run_match.py: command-line parsing and dispatch only. match_campaign.py: new match kinds "
               "(policy, head-to-head), round-2 records and this selection rule; the functions that schedule, "
               "play, publish and re-validate games are unchanged (git diff 1bacbfe..HEAD). training/checkpoints.py "
               "(5bb1e10): the mixed-checkpoint validator also accepts the round-2 sampler state; for round-1 and P "
               "checkpoints it dispatches to the unchanged MixedState/DataOrder path and loads the same tensors. "
               "The added files are standalone batching/development tools; nothing on the play path imports them. "
               "Every other evaluation source (adapter, search, model, matches, players, replay, runtime, contracts, "
               "tensor_ops, position_sampling) hashes identically; git shows no commit touching them since 1bacbfe."),
    evidence=("scripts/check_evaluation_equivalence.py: supporting evidence for the audit, not proof of B200 "
              "equivalence. P and R1-mix evaluated on CPU under both revisions on all 64 validation roots and the "
              "positions reached by alternating 128-simulation decisions from them; network outputs and search "
              "decisions byte-identical; the record (" + EQUIVALENCE_FILE + ") is bound to the exact old and new "
              "source hashes, checkpoints, reference plan, openings and numerical environment."))

# What the evidence record must cover before the selection admits the exception.
EQUIVALENCE_REQUIRED = dict(roots=OPENINGS_PER_SPLIT, min_plies=2, simulations=SIMULATIONS, white="P", black="R1-mix")


def round_paths(matches_root, round_number):
    names = ROUND_FILES.get(round_number)
    if names is None:
        raise ValueError(f"unknown round {round_number!r}")
    matches_root = Path(matches_root)
    return {k: matches_root / v for k, v in names.items()}


def latest_round(matches_root):
    """The highest round whose candidate record exists (1 when none does)."""
    present = [r for r, names in ROUND_FILES.items() if (Path(matches_root) / names["candidates"]).exists()]
    return max(present, default=1)


def plan_candidates(plan, matches_root):
    """The candidate record a plan was frozen with, found by identity among the rounds' records."""
    for names in ROUND_FILES.values():
        path = Path(matches_root) / names["candidates"]
        if path.exists() and identity(path) == plan["candidates_file"]:
            return load_candidates(path)
    raise ValueError(f"{plan['match_id']}: the candidate record this plan was frozen with is missing or changed")


def freeze_round2_candidates(output, *, root, spec=ROUND2_CANDIDATES, log=print):
    """Record the round-2 candidates once: P (unchanged), the round-1 entries carried over from the sealed
    round-1 record, and the round-2 runs trained from the round-1 selection."""
    from chess_rl.training.checkpoints import load_inference
    output, root = Path(output), Path(root).resolve()
    if output.exists():
        raise ValueError(f"{output} already exists; candidate eligibility is never rewritten")
    previous = load_candidates(root / spec["previous_round"]["candidates"])
    previous_selection = load_selection(root / spec["previous_round"]["selection"])
    if previous["record"]["P"]["sha256"] != spec["P"]["sha256"] or previous_selection["candidates_file"] != previous["file"]:
        raise ValueError("round-1 records do not belong together or name another P")
    if (not previous_selection["final"] or previous_selection["selected"]["name"] != spec["start"]["name"]
            or previous_selection["selected"]["sha256"] != spec["start"]["sha256"]):
        raise ValueError("round 2 must start from the final round-1 selection")

    def checked(path, sha256):
        if sha_file(root / path) != sha256:
            raise ValueError(f"{path}: checkpoint bytes differ from the recorded sha256")
        loaded = load_inference(root / path)
        if sha_file(root / path) != sha256:
            raise ValueError(f"{path}: checkpoint changed while loading")
        return loaded

    p = checked(spec["P"]["path"], spec["P"]["sha256"])
    if p.checkpoint_id != previous["record"]["P"]["checkpoint_id"]:
        raise ValueError("P differs from the round-1 record")
    names = [c["name"] for c in spec["candidates"]]
    if len(set(names)) != len(names) or "P" in names:
        raise ValueError("candidate names must be unique and differ from P")
    entries = []
    for c in spec["candidates"]:
        if c["eligibility"] not in ("selectable", "report-only"):
            raise ValueError("eligibility must be selectable or report-only")
        if c.get("carried_from_round") == 1:
            old = previous["by_name"].get(c["name"])
            if old is None or old["eligibility"] != c["eligibility"]:
                raise ValueError(f"{c['name']}: not a round-1 candidate with this eligibility")
            checked(old["path"], old["sha256"])
            plan_path = (root / spec["previous_round"]["candidates"]).parent / c["validation_match"] / "plan.json"
            plan = read_sealed(plan_path, PLAN_SCHEMA)
            if (plan["kind"] != "validation" or plan["candidates_file"] != previous["file"]
                    or plan["score_player_id"] != c["name"] or plan["players"][1]["sha256"] != old["sha256"]):
                raise ValueError(f"{c['name']}: {c['validation_match']} is not its round-1 validation match")
            entries.append(dict(old, carried_from_round=1, validation_match=c["validation_match"],
                                validation_plan_sha256=plan["sha256"], note=c["note"]))
            continue
        loaded = checked(c["path"], c["sha256"])
        prov = loaded.provenance
        if prov.get("initial_checkpoint", {}).get("sha256") != spec["start"]["sha256"]:
            raise ValueError(f"{c['name']}: not trained from {spec['start']['name']}")
        if prov.get("recipe") != c["recipe"]:
            raise ValueError(f"{c['name']}: recipe {prov.get('recipe')!r} differs from {c['recipe']!r}")
        exploratory = "exploratory" in prov or prov.get("thresholds", {}).get("hard_stop_enforced") is False
        if c["eligibility"] == "selectable" and (exploratory or prov.get("passes_over_selfplay") not in (None, 1)):
            raise ValueError(f"{c['name']}: an exploratory or multi-pass run cannot be selectable")
        if c["eligibility"] == "report-only" and not exploratory:
            raise ValueError(f"{c['name']}: report-only is reserved for the exploratory run")
        summary = json.loads((root / c["summary"]).read_text())
        role = "selectable" if c["eligibility"] == "selectable" else "report-only"
        if (summary["candidate"]["sha256"] != c["sha256"] or Path(summary["candidate"]["path"]).name != Path(c["path"]).name
                or summary.get("role") != role or bool(summary.get("exploratory", False)) != exploratory
                or summary.get("completed") is not True
                or summary.get("start_checkpoint", {}).get("sha256") != spec["start"]["sha256"]):
            raise ValueError(f"{c['name']}: run summary names another candidate, role or start")
        sampling = summary.get("round_sampling")
        if c["eligibility"] == "selectable" and sampling is not None:
            reused = [r for r in sampling["by_round"] if r["cycles_completed"] > 1 or r["effective_passes"] > 1.01]
            if reused:
                raise ValueError(f"{c['name']}: a run that re-sampled a round beyond one pass cannot be selectable")
        entries.append(dict(c, checkpoint_id=loaded.checkpoint_id, kind="rl-R", round=2,
                            provenance_sha256=digest(prov), exploratory=exploratory,
                            summary_identity=identity(root / c["summary"]),
                            summary_selection=summary["selection"], final_delta_total=summary.get("final_delta_total"),
                            round_sampling=None if sampling is None else sampling.get("by_round")))
    if not any(e["eligibility"] == "selectable" for e in entries):
        raise ValueError("at least one candidate must be selectable")
    tie_order = list(spec["tie_order"])
    eligibility = {e["name"]: e["eligibility"] for e in entries}
    if [n for n in tie_order if n not in names] or any(eligibility[n] != "selectable" for n in tie_order):
        raise ValueError("tie order may name only selectable candidates")
    content = dict(schema=CANDIDATES_SCHEMA, round=2,
                   previous_round=dict(candidates_file=previous["file"], candidates_sha256=previous["record"]["sha256"],
                                       selection_file=identity(root / spec["previous_round"]["selection"]),
                                       selection_sha256=previous_selection["sha256"],
                                       selected=previous_selection["selected"]["name"]),
                   start=dict(spec["start"], role="start weights of every round-2 run"),
                   P=dict(previous["record"]["P"]),
                   candidates=entries, excluded=[dict(e) for e in spec["excluded"]], tie_order=tie_order,
                   rules=dict(previous["record"]["rules"], cross_revision=CROSS_REVISION_EXCEPTION),
                   decided=spec["decided"], created_utc=_now())
    publish_once(output, sealed(content))
    log(json.dumps(dict(event="candidates-frozen", round=2, path=str(output), candidates=eligibility)))
    return load_candidates(output)


def source_difference(plan_sources, reference_sources):
    """Files whose hashes differ between two source maps (missing on either side counts as a difference)."""
    names = set(plan_sources) | set(reference_sources)
    return {n: dict(plan=plan_sources.get(n), reference=reference_sources.get(n))
            for n in sorted(names) if plan_sources.get(n) != reference_sources.get(n)}


def load_equivalence(path, *, old_plan, new_plan, candidates):
    """The evidence record for the cross-revision exception, bound to exactly these two plans: the old plan's
    complete pinned sources, the new plan's hashes of the same files, the three excepted files on both sides,
    P and R1-mix by sha256, R1-mix's sealed plan, the frozen openings, and full coverage (all 64 validation
    roots, 128 simulations, at least two decisions per root, both networks at every position, no smoke run)."""
    record = read_sealed(path, EQUIVALENCE_SCHEMA)
    files = CROSS_REVISION_EXCEPTION["files"]
    old_sources, new_sources = old_plan["sources"], new_plan["sources"]
    if (record["old_sources"] != {n: old_sources[n] for n in files}
            or record["new_sources"] != {n: new_sources[n] for n in files}
            or record["old_pinned_sources"] != old_sources
            or record["new_pinned_sources"] != {n: new_sources.get(n) for n in old_sources}):
        raise ValueError(f"{path}: evidence is for another pair of revisions")
    r1 = candidates["by_name"].get("R1-mix")
    checkpoints = record["checkpoints"]
    if (r1 is None or set(checkpoints) != {"P", "R1-mix"}
            or checkpoints["R1-mix"].get("sha256") != r1["sha256"]
            or checkpoints["P"].get("sha256") != candidates["record"]["P"]["sha256"]
            or record["reference_plan"]["sha256"] != r1.get("validation_plan_sha256")
            or record["reference_plan"]["sha256"] != old_plan["sha256"]
            or record["openings_file"] != new_plan["openings_file"]
            or record["openings_set_sha256"] != new_plan["openings_set_sha256"]):
        raise ValueError(f"{path}: evidence is for other checkpoints, another reference plan or other openings")
    s, need = record["settings"], EQUIVALENCE_REQUIRED
    if (record.get("smoke") or s["roots"] != need["roots"] or s["plies"] < need["min_plies"]
            or s["simulations"] != need["simulations"] or s["white"] != need["white"] or s["black"] != need["black"]
            or record["comparisons"] < need["roots"] or record["search_decisions"] < need["roots"]
            or record["network_evaluations"] != 2 * record["comparisons"]):
        raise ValueError(f"{path}: evidence coverage is incomplete (all 64 validation roots, >= 2 decisions per root at "
                         "128 simulations, both networks at every position; no smoke run)")
    if not record["passed"] or record["comparisons"] == 0 or record["mismatches"] != 0:
        raise ValueError(f"{path}: evidence does not establish equivalence")
    return record


def _check_round2_sources(candidates, plans, matches_root):
    """Round-2 source rule: every validation plan's sources equal the newest plan's, except the three files of
    the cross-revision exception at their recorded round-1 hashes and the listed added files at their exact
    hashes; a pinned file missing from the newest plan is never admitted. Any admitted difference needs the
    evidence record. Returns the audit written into the selection record."""
    exception = CROSS_REVISION_EXCEPTION
    newest = max(plans.values(), key=lambda p: p["created_utc"])
    audit = dict(reference_plan=newest["match_id"], exception=exception, plans={}, evidence=None)
    older = []
    for n, plan in plans.items():
        diff = source_difference(plan["sources"], newest["sources"])
        audit["plans"][n] = dict(match_id=plan["match_id"], differing_files=diff)
        if not diff:
            continue
        for name, pair in diff.items():
            if pair["reference"] is None:
                raise ValueError(f"{n}: evaluation source {name} is pinned by this plan but missing from the round-2 "
                                 "plans; not covered by the cross-revision exception")
            if pair["plan"] is None:
                if exception["added_files"].get(name) != pair["reference"]:
                    raise ValueError(f"{n}: evaluation source {name} appears only in the round-2 plans and is not in "
                                     "the audited list of added files at its exact hash")
                continue
            if name not in exception["files"] or pair["plan"] != exception["files"][name]:
                raise ValueError(f"{n}: evaluation source {name} differs from the round-2 plans and is not covered by "
                                 "the cross-revision exception; rerun this validation match under the current revision")
        older.append(plan)
    if older:
        reference = next((p for p in older if p["score_player_id"] == "R1-mix"), None)
        if reference is None or any(p["sources"] != reference["sources"] for p in older):
            raise ValueError("the cross-revision exception covers only round-1 plans at R1-mix's exact revision")
        path = Path(matches_root) / EQUIVALENCE_FILE
        if not path.exists():
            raise ValueError(f"{path} is missing: the cross-revision exception needs the equivalence evidence")
        record = load_equivalence(path, old_plan=reference, new_plan=newest, candidates=candidates)
        audit["evidence"] = dict(file=identity(path), sha256=record["sha256"], comparisons=record["comparisons"],
                                 search_decisions=record["search_decisions"], settings=record["settings"],
                                 old_revision=record.get("old_revision"), runtime=record.get("runtime"))
    return audit


def _check_validation_plans_round2(candidates, plans, matches_root):
    """Round-2 variant of _check_validation_plans: a carried-over match may pin the round-1 candidates file;
    sources follow the audited exception."""
    record, first = candidates["record"], next(iter(plans.values()))
    allowed_files = (candidates["file"], record["previous_round"]["candidates_file"])
    for n, plan in plans.items():
        spec = candidates["by_name"][n]
        expected_file = allowed_files[1] if spec.get("carried_from_round") == 1 else allowed_files[0]
        if (plan["kind"] != "validation" or plan["candidates_file"] != expected_file
                or plan["opening_ids"] != first["opening_ids"] or plan["openings_set_sha256"] != first["openings_set_sha256"]
                or plan["settings"] != first["settings"] or plan["ply_cap"] != first["ply_cap"]
                or plan["protocol"] != first["protocol"] or plan["seed"] != first["seed"]
                or plan["runtime"] != first["runtime"]
                or plan["device"] != first["device"] or plan["workers"] != first["workers"]
                or plan["score_player_id"] != n or plan["players"][1]["sha256"] != spec["sha256"]
                or plan["players"][0]["sha256"] != record["P"]["sha256"]
                or (spec.get("carried_from_round") == 1 and plan["sha256"] != spec["validation_plan_sha256"])):
            raise ValueError(f"{n}: validation plan differs from the frozen protocol")
    return first, _check_round2_sources(candidates, plans, matches_root)


# Runtime and sources.

def _runtime_in_process(device):
    from chess_rl.runtime import configure
    configure(device, threads=1)
    return runtime_identity(device)


def runtime_identity(device):
    """Identity of a configured worker process (call after runtime.configure)."""
    import torch
    # No kernel version or device memory size: a resume on another node of the same image stays valid.
    value = dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                 torch=str(torch.__version__), numpy=importlib.metadata.version("numpy"), chess=chess.__version__,
                 python_chess=importlib.metadata.version("python-chess"), device=device, dtype="float32",
                 threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled(),
                 matmul_precision=torch.get_float32_matmul_precision())
    if device == "cuda:0":
        from chess_rl.training.checkpoints import cuda_runtime_config
        value["cuda"] = {k: v for k, v in cuda_runtime_config().items() if k != "total_memory"}
    return value


def worker_runtime(device):
    """The worker runtime, measured in a fresh spawned process so the caller's torch state is untouched."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_runtime_in_process, device).result()


def source_files(root):
    """The listed evaluation sources plus every chess_rl module imported by the evaluation path."""
    import sys
    import chess_rl
    import chess_rl.runtime, chess_rl.search.real_evaluator, chess_rl.training.checkpoints  # noqa: E401,F401
    root = Path(root)
    names = set()
    for name in EVALUATION_SOURCES:
        path = root / name
        names.update(str(p.relative_to(root)) for p in (path.rglob("*.py") if path.is_dir() else [path]))
    package = Path(chess_rl.__file__).resolve().parent.parent
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if getattr(module, "__name__", "").split(".")[0] == "chess_rl" and file:
            names.add(str(Path("src") / Path(file).resolve().relative_to(package)))
    return [root / n for n in sorted(names)]


def source_hashes(root, names=None):
    root = Path(root).resolve()
    files = source_files(root) if names is None else [root / n for n in names]
    return {str(p.relative_to(root)): sha_file(p) for p in files}


# Players.

def _unavailable(states):
    raise RuntimeError("plan-time player identity only; no evaluation")


def search_player(spec, root, settings, evaluator=_unavailable):
    checkpoint = Checkpoint(spec["checkpoint_id"], str((Path(root) / spec["path"]).resolve()),
                            spec["sha256"], spec["kind"])
    role = "P" if spec["kind"] == "pretrained-P" else "R"
    return SearchPlayer(spec["name"], role, f"real-evaluator:{spec['sha256']}", evaluator, settings, checkpoint)


def opponent_player(name, pin=None):
    if name == "random-legal":
        return RandomLegalPlayer()
    if name == "material-greedy":
        return GreedyPlayer()
    if name in STOCKFISH_NODES:
        if pin is None:
            raise ValueError("a Stockfish rung needs the pinned engine")
        return StockfishPlayer(StockfishPin(**pin), nodes=STOCKFISH_NODES[name], player_id=name)
    raise ValueError(f"unknown rung opponent {name!r}")


# Plans.

def jobs(plan):
    for raw in plan["openings"]:
        pair_id = f"{plan['match_id']}:{raw['opening_id']}"
        for leg in (0, 1):
            yield dict(game_id=f"{raw['opening_id']}:leg{leg}", pair_id=pair_id, leg=leg,
                       game_seed=int(digest([plan["match_id"], plan["seed"], raw["opening_id"], leg])[:16], 16),
                       opening=raw)


def _check_protocol(protocol, kind, simulations, ply_cap, workers, count):
    if protocol not in ("production", "fixture"):
        raise ValueError("protocol must be production or fixture")
    if protocol == "production":
        if kind == "head-to-head":
            if simulations not in (SIMULATIONS, POLICY_SIMULATIONS) or ply_cap != PLY_CAP:
                raise ValueError("production head-to-head matches use 128 or 1 simulations and the 512-ply cap")
        elif kind == "policy":
            if simulations != POLICY_SIMULATIONS or ply_cap != PLY_CAP:
                raise ValueError("production no-search matches use 1 simulation and the 512-ply cap")
        elif simulations != SIMULATIONS or ply_cap != PLY_CAP:
            raise ValueError("production matches use 128 simulations and the 512-ply cap")
        if kind not in ("rung", "policy", "head-to-head") and workers != WORKERS:
            raise ValueError("production B200 matches use 5 workers")
        if count is not None:
            raise ValueError("production matches play their frozen opening count")


def load_selection(path):
    return read_sealed(path, SELECTION_SCHEMA)


def create_plan(matches_root, *, root, kind, candidate=None, opponent=None, openings_path, candidates_path,
                device, workers, simulations=SIMULATIONS, ply_cap=PLY_CAP, protocol="production",
                selection_path=None, stockfish=None, count=None, runtime=None, log=print):
    """Freeze one match. Validation: P vs a frozen candidate on all validation openings. Smoke: P vs a
    candidate on the two smoke openings. Test and rung: the selected R only (final selection required)."""
    matches_root, root = Path(matches_root), Path(root).resolve()
    if kind not in KINDS:
        raise ValueError(f"unknown match kind {kind!r}")
    _check_protocol(protocol, kind, simulations, ply_cap, workers, count)
    if device not in ("cpu", "cuda:0") or type(workers) is not int or workers < 1:
        raise ValueError("device must be cpu or cuda:0 and workers a positive integer")
    openings = load_openings(openings_path)
    candidates = load_candidates(candidates_path)
    settings = SearchSettings(simulations)
    selection = None
    if kind in ("test", "rung"):
        latest = latest_round(matches_root)
        if candidates["record"].get("round", 1) != latest:
            raise ValueError(f"a round-{latest} candidate record exists; a {kind} match binds only to the latest "
                             "round's final selection")
        if selection_path is None or not Path(selection_path).exists():
            raise ValueError(f"a {kind} match needs the recorded selection")
        selection = load_selection(selection_path)
        if not selection["final"] or selection["selected"] is None:
            raise ValueError(f"the selection is not final; no {kind} match can be planned")
        if selection["protocol"] != protocol or selection["candidates_file"] != candidates["file"]:
            raise ValueError("selection belongs to another protocol or candidate record")
        if candidate is not None and candidate != selection["selected"]["name"]:
            raise ValueError(f"a {kind} match plays only the selected candidate {selection['selected']['name']}")
        candidate = selection["selected"]["name"]
    spec = candidates["by_name"].get(candidate)
    if spec is None:
        raise ValueError(f"{candidate!r} is not a frozen validation candidate")
    if selection is not None and spec["sha256"] != selection["selected"]["sha256"]:
        raise ValueError("selected checkpoint differs from the frozen candidate")
    p_spec = candidates["record"]["P"]
    if kind == "smoke":
        selected = openings["smoke"]
        split = "smoke"
    elif kind == "test":
        selected, split = list(openings["sets"].test), "test"
    else:
        selected, split = list(openings["sets"].validation), "validation"
        if kind == "rung":
            selected = selected[:RUNG_PAIRS]
    if count is not None:
        selected = selected[:count]
    pin = None
    if kind == "rung":
        if opponent not in RUNG_OPPONENTS:
            raise ValueError(f"rung opponent must be one of {RUNG_OPPONENTS}")
        if opponent in STOCKFISH_NODES:
            if stockfish is None:
                raise ValueError("a Stockfish rung needs --stockfish")
            pin = dict(STOCKFISH_PIN, executable=str(Path(stockfish).resolve()))
            if protocol == "fixture":
                pin = dict(stockfish)
            StockfishPin(**pin).verify()
        left, right = search_player(spec, root, settings), opponent_player(opponent, pin)
        match_id = f"rung-{candidate}-vs-{opponent}"
        comparison = "fixed-opponent"
    elif kind == "head-to-head":
        other = candidates["by_name"].get(opponent)
        if other is None or opponent == candidate:
            raise ValueError("a head-to-head match needs a second, different frozen candidate as --opponent")
        left, right = search_player(spec, root, settings), search_player(other, root, settings)
        match_id = f"head-to-head-{candidate}-vs-{opponent}-sims{simulations}"
        comparison = "R-versus-R"
    else:
        if opponent is not None:
            raise ValueError("only rungs and head-to-head matches take an opponent")
        left, right = search_player(p_spec, root, settings), search_player(spec, root, settings)
        match_id = f"{kind}-P-vs-{candidate}"
        comparison = "P-versus-R"
    for player in (left, right):
        if isinstance(player, SearchPlayer):
            player.checkpoint.verify()
    if kind == "test":
        if (matches_root / "test-plan-claim.json").exists() or any(matches_root.glob("test-*/plan.json")):
            raise ValueError("a test plan already exists; the test openings are played once")
    directory = matches_root / match_id
    if (directory / "plan.json").exists():
        raise ValueError(f"{directory}/plan.json already exists; plans are never rewritten")
    players = []
    for side, player in (("left", left), ("right", right)):
        entry = dict(side=side, metadata=snapshot(player.metadata()))
        if isinstance(player, SearchPlayer):
            s = p_spec if player.role == "P" else candidates["by_name"][player.player_id]
            entry.update(type="search", name=s["name"], path=s["path"], sha256=s["sha256"],
                         checkpoint_id=s["checkpoint_id"], kind=s["kind"],
                         eligibility=s.get("eligibility", "reference"), recipe=s.get("recipe"))
        else:
            entry.update(type="opponent", name=opponent, pin=pin)
        players.append(entry)
    plan = dict(schema=PLAN_SCHEMA, protocol=protocol, kind=kind, match_id=match_id, comparison=comparison,
                split=split, planned_pairs=len(selected), planned_games=2 * len(selected),
                opening_ids=[o.opening_id for o in selected], openings=[o.to_dict() for o in selected],
                openings_file=openings["file"], openings_set_sha256=openings["record"]["sets"]["sha256"],
                candidates_file=candidates["file"], players=players,
                score_player_id=right.player_id if comparison == "P-versus-R" else left.player_id,
                score_orientation={"P-versus-R": "R's score against P",
                                   "fixed-opponent": "R's score against the fixed opponent",
                                   "R-versus-R": f"{left.player_id}'s score against {right.player_id}"}[comparison],
                settings=settings.to_dict(), seed=SEEDS[kind], seed_algorithm="sha256(match_id,seed,opening_id,leg)-v1",
                pairing="identical opening and history; left White in leg 0, Black in leg 1",
                ply_cap=ply_cap, workers=workers, device=device,
                runtime=runtime if runtime is not None else worker_runtime(device),
                sources=source_hashes(root), statistics=dict(BOOTSTRAP, confidence=0.95, unit="opening-pair"),
                rules=dict(incomplete=INCOMPLETE_RULE, claim=CLAIM_RULE if kind == "test" else None,
                           policy=POLICY_RULE if kind == "policy" else None,
                           head_to_head=HEAD_TO_HEAD_RULE if kind == "head-to-head" else None),
                report_policy=("statuses, lengths and timing only" if kind == "smoke"
                               else "scores only for a finished schedule"),
                selection_file=None if selection is None else identity(selection_path),
                created_utc=_now())
    plan["schedule"] = list(jobs(plan))
    publish_once(directory / "plan.json", sealed(plan))
    if kind == "test":
        publish_once(matches_root / "test-plan-claim.json",
                     sealed(dict(schema="match-test-claim-v1", match_id=match_id, created_utc=_now())))
    log(json.dumps(dict(event="plan-frozen", match_id=match_id, kind=kind, pairs=len(selected),
                        directory=str(directory))))
    return load_plan(directory, root)


def load_plan(directory, root, *, check_sources=True):
    """check_sources is enforced for playing and resuming; reading a finished store re-validates every game
    by replay with the current code and reports whether the sources still match."""
    plan = read_sealed(Path(directory) / "plan.json", PLAN_SCHEMA)
    if plan["schedule"] != snapshot(list(jobs(plan))):
        raise ValueError("frozen schedule differs from the plan's openings and seeds")
    if [o["opening_id"] for o in plan["openings"]] != plan["opening_ids"]:
        raise ValueError("opening ids differ from the frozen openings")
    if check_sources and plan["sources"] != source_hashes(root, sorted(plan["sources"])):
        raise ValueError("evaluation source differs from the frozen plan; do not mix results across revisions")
    return plan


def verify_inputs(plan, root, matches_root):
    """Checkpoint bytes and the frozen opening, candidate and selection files still match."""
    root, matches_root = Path(root).resolve(), Path(matches_root)
    frozen_openings = plan["openings_file"]
    if frozen_openings is not None and (not (matches_root / "openings.json").exists()
                                        or identity(matches_root / "openings.json") != frozen_openings):
        raise ValueError("openings.json differs from the file this plan was frozen with")
    # A plan pins the candidate (and selection) record of its own round; the file must still exist unchanged.
    for key, frozen in (("candidates", plan["candidates_file"]), ("selection", plan["selection_file"])):
        if frozen is None:
            continue
        files = [matches_root / names[key] for names in ROUND_FILES.values()]
        if not any(f.exists() and identity(f) == frozen for f in files):
            raise ValueError(f"the {key} record this plan was frozen with is missing or changed")
    for entry in plan["players"]:
        if entry["type"] == "search" and sha_file(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"{entry['path']}: checkpoint bytes changed since the plan was frozen")
        if entry["type"] == "opponent" and entry["pin"] is not None:
            StockfishPin(**entry["pin"]).verify()


# Results store.

def connect(path):
    return pilot.connect(path)


def initialize_store(path, plan):
    pilot.initialize_store(path, plan)


def validate_game(record, job, plan):
    """Replay one stored game against its scheduled pair, seeds, routing, budget and outcome."""
    cap = plan["ply_cap"]
    budget = plan["settings"]["simulation_budget"]
    if (record["pair_id"] != job["pair_id"] or record["leg"] != job["leg"]
            or record["game_seed"] != job["game_seed"] or record["ply_cap"] != cap
            or record["opening"] != snapshot(opening(job["opening"]).to_dict())):
        raise ValueError("stored game differs from its scheduled pair, seed, cap or opening")
    left, right = plan["players"]
    white, black = (left, right) if job["leg"] == 0 else (right, left)
    for color, entry in (("white", white), ("black", black)):
        if record["players"][color] != entry["metadata"]:
            raise ValueError("stored game checkpoint/colour routing mismatch")
    by_id = {e["metadata"]["player_id"]: e for e in plan["players"]}
    state = opening(job["opening"]).state()
    for i, attempt in enumerate(record["attempts"]):
        ply = len(state.board.move_stack)
        color = "white" if state.board.turn else "black"
        player_id = record["players"][color]["player_id"]
        if (terminal_result(state) is not None or ply >= cap or attempt["ply"] != ply
                or attempt["color"] != color or attempt["player_id"] != player_id
                or attempt["seed"] != turn_seed(job["game_seed"], ply, player_id)):
            raise ValueError("stored move routing/history mismatch")
        if attempt["applied"]:
            if attempt["status"] != "completed" or id_to_move(state, attempt["move_id"]).uci() != attempt["move_uci"]:
                raise ValueError("stored applied move mismatch")
            entry = by_id[player_id]
            if entry["type"] == "search":
                details = attempt["details"]
                edges = details["visited_edges"]
                if (details["requested_simulations"] != budget or details["completed_simulations"] != budget
                        or sum(edge[1] for edge in edges) != budget
                        or max(edges, key=lambda e: (e[1], -e[0]))[0] != attempt["move_id"]):
                    raise ValueError("stored search budget or maximum-visit move mismatch")
            elif entry["name"] in ("random-legal", "material-greedy"):
                if opponent_player(entry["name"]).choose(state.copy(), seed=attempt["seed"]).move_id != attempt["move_id"]:
                    raise ValueError("stored fixed-opponent move does not replay")
            apply_move(state, attempt["move_id"])
        elif (i != len(record["attempts"]) - 1 or attempt["status"] != record["status"]
              or record["status"] not in ("failed", "interrupted")):
            raise ValueError("an unapplied move must end a failed/interrupted game")
    if any(record[k] != v for k, v in state_trace(state).items()):
        raise ValueError("stored final history mismatch")
    terminal = terminal_result(state)
    if record["status"] == "completed":
        if terminal is None:
            raise ValueError("only actual terminal outcomes can be scored")
        winner = None if terminal.winner is None else "white" if terminal.winner else "black"
        if (record["winner"] != winner or record["ending_reason"] != terminal.ending_reason
                or record["outcome"] != {None: "1/2-1/2", "white": "1-0", "black": "0-1"}[winner]):
            raise ValueError("stored rule outcome mismatch")
    else:
        if record["winner"] is not None or record["outcome"] is not None:
            raise ValueError("truncated/failed/interrupted games have no score")
        if record["status"] == "truncated":
            if terminal is not None or len(state.board.move_stack) < cap or record["ending_reason"] != "ply_limit":
                raise ValueError("invalid truncation")
        elif record["status"] not in ("failed", "interrupted") or record["error"] is None:
            raise ValueError("invalid failure status")


def publish(path, job, plan, record, timing, runtime):
    validate_game(record, job, plan)
    value = dict(game_id=job["game_id"], plan_sha256=plan["sha256"], timing=timing, runtime=runtime, record=record)
    payload = json.dumps(value, sort_keys=True, allow_nan=False)
    with connect(path) as db:
        # No replacement of completed, truncated, failed or interrupted records.
        db.execute("INSERT INTO games VALUES (?, ?, ?)", (job["game_id"], payload, digest(value)))


def read_results(path, plan):
    path = Path(path)
    if not path.exists():
        return {}
    with connect(path) as db:
        saved = db.execute("SELECT value FROM metadata WHERE key='plan'").fetchone()
        if saved is None or json.loads(saved[0]) != snapshot(plan):
            raise ValueError("result store belongs to a different frozen plan")
        rows = db.execute("SELECT game_id, payload, sha256 FROM games ORDER BY game_id").fetchall()
    scheduled = {j["game_id"]: j for j in plan["schedule"]}
    values = {}
    for game_id, payload, checksum in rows:
        value = json.loads(payload)
        if (game_id not in scheduled or value["game_id"] != game_id or value["plan_sha256"] != plan["sha256"]
                or checksum != digest(value) or value["runtime"] != plan["runtime"]):
            raise ValueError("stored result identity/digest/runtime mismatch")
        validate_game(value["record"], scheduled[game_id], plan)
        values[game_id] = value
    return values


# Workers.

@contextmanager
def run_lock(output):
    with pilot.run_lock(output):
        yield


_WORKER = None
_WORKER_LOCK = None


def load_players(plan, root):
    """Bind each search player to its verified checkpoint bytes; both networks load once per worker."""
    from chess_rl.search.real_evaluator import RealEvaluator
    settings = SearchSettings(plan["settings"]["simulation_budget"])
    players = []
    for entry in plan["players"]:
        if entry["type"] == "search":
            path = Path(root) / entry["path"]
            if sha_file(path) != entry["sha256"]:
                raise ValueError(f"{entry['path']}: checkpoint bytes differ before loading")
            evaluator = (RealEvaluator(path) if plan["device"] == "cpu"
                         else RealEvaluator(path, device=plan["device"]))
            if evaluator.checkpoint.checkpoint_id != entry["checkpoint_id"]:
                raise ValueError("loaded checkpoint id mismatch")
            if sha_file(path) != entry["sha256"]:
                raise ValueError(f"{entry['path']}: checkpoint changed while loading")
            spec = dict(name=entry["name"], path=entry["path"], sha256=entry["sha256"],
                        checkpoint_id=entry["checkpoint_id"], kind=entry["kind"])
            player = search_player(spec, root, settings, evaluator)
        else:
            player = opponent_player(entry["name"], entry["pin"])
        if snapshot(player.metadata()) != entry["metadata"]:
            raise ValueError("worker player identity differs from the frozen plan")
        players.append(player)
    if players[0].metadata()["player_id"] == players[1].metadata()["player_id"]:
        raise ValueError("player identities must differ")
    return tuple(players)


def initialize_worker(plan, root, database):
    global _WORKER, _WORKER_LOCK
    # The coordinator owns stopping: a Ctrl-C must not turn running games into interrupted records.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Keep the output protected if the coordinator dies while a worker lives.
    _WORKER_LOCK = Path(database).with_name("workers.lock").open("a+")
    fcntl.flock(_WORKER_LOCK, fcntl.LOCK_SH)
    from chess_rl.runtime import configure
    configure(plan["device"], threads=1)
    runtime = runtime_identity(plan["device"])
    if runtime != plan["runtime"]:
        raise ValueError("worker runtime differs from the frozen plan")
    _WORKER = (plan, database, load_players(plan, root), runtime)


def execute_job(job):
    plan, database, players, runtime = _WORKER
    left, right = players
    white, black = (left, right) if job["leg"] == 0 else (right, left)
    started_utc, started = _now(), time.monotonic()
    record = play_game(opening(job["opening"]), white, black, pair_id=job["pair_id"],
                       leg=job["leg"], game_seed=job["game_seed"], ply_cap=plan["ply_cap"])
    timing = dict(elapsed_seconds=time.monotonic() - started, started_utc=started_utc, finished_utc=_now(),
                  plies_played=len(record["moves_uci"]) - len(job["opening"]["moves_uci"]))
    publish(database, job, plan, record, timing, runtime)
    return job["game_id"], record["status"]


def pending_jobs(plan, values):
    return [j for j in plan["schedule"] if j["game_id"] not in values]


def run(directory, root, *, log=print):
    """Play every game without a stored row; resumable after any kill."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    plan = load_plan(directory, root)
    verify_inputs(plan, root, directory.parent)
    database = directory / "results.sqlite3"
    with run_lock(directory):
        initialize_store(database, plan)
        values = read_results(database, plan)
        pending = pending_jobs(plan, values)
        log(json.dumps(dict(event="run-start", match_id=plan["match_id"], recorded=len(values),
                            pending=len(pending), workers=plan["workers"], device=plan["device"],
                            host=platform.node(), utc=_now())))
        try:
            if pending:
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    os.environ[name] = "1"
                pool = ProcessPoolExecutor(max_workers=min(plan["workers"], len(pending)),
                                           mp_context=multiprocessing.get_context("spawn"),
                                           initializer=initialize_worker,
                                           initargs=(plan, str(root), str(database)))
                clean = False
                try:
                    futures = [pool.submit(execute_job, job) for job in pending]
                    for future in as_completed(futures):
                        game_id, outcome_status = future.result()
                        log(json.dumps(dict(event="game", game_id=game_id, status=outcome_status, durable=True,
                                            utc=_now())))
                    clean = True
                finally:
                    if not clean:
                        # Stop at once: a killed worker loses only its uncommitted game, replayed on resume.
                        for process in list((getattr(pool, "_processes", None) or {}).values()):
                            if process.is_alive():
                                process.kill()
                    pool.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:
            with connect(database) as db:
                db.execute("INSERT INTO events(payload) VALUES (?)", (json.dumps(dict(
                    event="coordinator_stopped", type=type(exc).__name__, message=str(exc), utc=_now())),))
            raise
        result = status(directory, root)
        log(json.dumps(dict(event="run-end", **result)))
        return result


# Status, reports and selection.

def status(directory, root):
    """Counts only; no scores."""
    plan = load_plan(directory, root, check_sources=False)
    values = read_results(Path(directory) / "results.sqlite3", plan)
    counts = {k: 0 for k in ("completed", "truncated", "failed", "interrupted")}
    for value in values.values():
        counts[value["record"]["status"]] += 1
    return dict(match_id=plan["match_id"], kind=plan["kind"], planned_games=plan["planned_games"],
                recorded_games=len(values), pending_games=plan["planned_games"] - len(values),
                status_counts=counts, finished=len(values) == plan["planned_games"], plan_sha256=plan["sha256"])


def _quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
    return dict(n=len(values), mean=statistics.fmean(values), min=ordered[0], median=statistics.median(ordered),
                p90=pick(0.9), max=ordered[-1])


def timing_report(plan, values):
    games = [v for v in values.values()]
    elapsed = [v["timing"]["elapsed_seconds"] for v in games]
    plies = [v["timing"]["plies_played"] for v in games]
    total_plies = [len(v["record"]["moves_uci"]) for v in games]
    per_ply = [e / p for e, p in zip(elapsed, plies, strict=True) if p > 0]
    span = None
    if games:
        start = min(datetime.fromisoformat(v["timing"]["started_utc"]) for v in games)
        end = max(datetime.fromisoformat(v["timing"]["finished_utc"]) for v in games)
        span = (end - start).total_seconds()
    return dict(games=len(games), plies_played=_quantiles(plies), game_length_total_plies=_quantiles(total_plies),
                seconds_per_game=_quantiles(elapsed), seconds_per_ply=_quantiles(per_ply),
                pooled_seconds_per_ply=None if not sum(plies) else sum(elapsed) / sum(plies),
                wall_seconds=span, workers=plan["workers"],
                concurrent_games=min(plan["workers"], plan["planned_games"]))


def project_hours(seconds_per_ply, plies_per_game, games, workers):
    """Wall hours for a schedule if every worker is busy: games x (s/ply x plies) / workers."""
    return games * seconds_per_ply * plies_per_game / workers / 3600


def smoke_report(plan, values):
    """Statuses, lengths and timing only. Nothing links a length or status to a game, leg or colour:
    a completed game's final side to move (ply parity) plus its leg would reveal who won."""
    counts = {k: 0 for k in ("completed", "truncated", "failed", "interrupted")}
    for value in values.values():
        counts[value["record"]["status"]] += 1
    lengths = sorted(v["timing"]["plies_played"] for v in values.values())
    timing = timing_report(plan, values)
    timing.pop("game_length_total_plies")
    report = dict(match_id=plan["match_id"], kind="smoke", plan_sha256=plan["sha256"], status_counts=counts,
                  recorded_games=len(values), planned_games=plan["planned_games"],
                  plies_played_sorted=lengths,
                  seconds_per_game_sorted=sorted(v["timing"]["elapsed_seconds"] for v in values.values()),
                  timing=timing, report_policy=plan["report_policy"],
                  note=("outcomes are stored but not reported, and no length is tied to a game; four games give a "
                        "rough cost estimate whose main uncertainty is the average game length"))
    s = timing["pooled_seconds_per_ply"]
    if s:
        mean_length = statistics.fmean(lengths)
        report["projection_hours_at_5_workers"] = {
            f"L={L}": dict(validation_two_matches_256_games=project_hours(s, L, 256, WORKERS),
                           test_128_games=project_hours(s, L, 128, WORKERS))
            for L in sorted({int(round(mean_length)), 100, 150, 200})}
        report["projection_basis"] = (f"pooled {s:.3f} s per ply per game with {timing['concurrent_games']} games "
                                      "running at once; plies counted from the opening root")
    return report


def colour_wdl(games, player_id):
    out = {c: dict(win=0, draw=0, loss=0) for c in ("white", "black")}
    for g in games:
        if g["status"] != "completed":
            continue
        color = next(c for c in ("white", "black") if g["players"][c]["player_id"] == player_id)
        label = "draw" if g["winner"] is None else "win" if g["winner"] == color else "loss"
        out[color][label] += 1
    return out


def test_claim(summary, planned_pairs, protocol, failures):
    """The claim rule on R's test pair-bootstrap interval, with the validation attrition thresholds."""
    needed = PROVISIONAL_MIN_PAIRS if protocol == "production" else math.ceil(0.75 * planned_pairs)
    complete = summary["complete_pairs"]
    interval = None if summary["paired_uncertainty"] is None else summary["paired_uncertainty"]["interval"]
    if failures:
        return dict(result="none", reason="failed or interrupted test games must be diagnosed; no claim")
    if complete < needed or interval is None:
        return dict(result="none", reason=f"{complete} complete pairs, below the {needed} needed; no claim")
    result = ("R improves over P" if interval[0] > 0.5 else "regression" if interval[1] < 0.5
              else "neither improvement nor regression supported")
    claim = dict(result=result, interval=interval, complete_pairs=complete, planned_pairs=planned_pairs,
                 rule=CLAIM_RULE)
    if complete < planned_pairs:
        claim.update(provisional=True, reason=(f"{planned_pairs - complete} pairs incomplete (truncation); the "
                     "interval covers complete pairs only; Arvind confirms or rejects the claim"))
    return claim


def match_report(directory, root):
    """Full report of a finished schedule (smoke: statuses, lengths and timing only). A validation report
    also waits until every frozen candidate's validation match is finished."""
    directory = Path(directory)
    plan = load_plan(directory, root, check_sources=False)
    values = read_results(directory / "results.sqlite3", plan)
    if plan["kind"] == "smoke":
        return smoke_report(plan, values)
    if len(values) != plan["planned_games"]:
        raise ValueError(f"{plan['match_id']}: {plan['planned_games'] - len(values)} games pending; scores are "
                         "reported only for a finished schedule (use status)")
    comparison = None
    if plan["kind"] == "validation":
        candidates = plan_candidates(plan, directory.parent)
        selectable = [n for n, c in candidates["by_name"].items() if c["eligibility"] == "selectable"]
        matches = {}
        for name in selectable:
            sibling = directory.parent / candidates["by_name"][name].get("validation_match", f"validation-P-vs-{name}")
            if not (sibling / "plan.json").exists() or not status(sibling, root)["finished"]:
                raise ValueError(f"validation-P-vs-{name} is not finished; validation scores are reported only "
                                 "when every selectable candidate's validation match is finished")
            sibling_plan = load_plan(sibling, root, check_sources=False)
            stored = read_results(sibling / "results.sqlite3", sibling_plan)
            matches[name] = (sibling_plan, [stored[j["game_id"]]["record"] for j in sibling_plan["schedule"]])
        own_name = plan["score_player_id"]
        if own_name not in matches:
            matches[own_name] = (plan, [values[j["game_id"]]["record"] for j in plan["schedule"]])
            comparison = compare_on_common_pairs(selectable + [own_name], matches)
    games = [values[j["game_id"]]["record"] for j in plan["schedule"]]
    summary = summarize(games, plan["score_player_id"], seed=plan["statistics"]["seed"],
                        bootstrap_samples=plan["statistics"]["samples"])
    report = dict(match_id=plan["match_id"], kind=plan["kind"], plan_sha256=plan["sha256"],
                  score_player_id=plan["score_player_id"], score_orientation=plan["score_orientation"],
                  summary=summary, wdl_by_colour=colour_wdl(games, plan["score_player_id"]),
                  failures=summary["status_counts"]["failed"] + summary["status_counts"]["interrupted"],
                  timing=timing_report(plan, values),
                  sources_match_plan=plan["sources"] == source_hashes(root, sorted(plan["sources"])))
    if plan["kind"] == "validation":
        report["eligibility"] = candidates["by_name"][plan["score_player_id"]]["eligibility"]
        if comparison is not None:
            report["comparison_with_selectable_on_common_pairs"] = comparison
        report["note"] = ("validation picks among candidates; the selection biases the selected candidate's "
                          "interval upward, so validation never supports an improvement claim")
    if plan["kind"] == "policy":
        report["note"] = POLICY_RULE
    if plan["kind"] == "head-to-head":
        report["note"] = HEAD_TO_HEAD_RULE
    if plan["kind"] == "test":
        report["claim"] = test_claim(summary, plan["planned_pairs"], plan["protocol"], report["failures"])
    return report


def common_pairs(per_match):
    """Opening ids whose pairs are complete (both legs completed) in every given match."""
    sets = []
    for games in per_match:
        legs = {}
        for g in games:
            legs.setdefault(g["pair_id"].rsplit(":", 1)[1], []).append(g)
        sets.append({o for o, pair in legs.items() if len(pair) == 2 and all(g["status"] == "completed" for g in pair)})
    return set.intersection(*sets) if sets else set()


def restricted_score(games, player_id, openings, statistics_):
    chosen = [g for g in games if g["pair_id"].rsplit(":", 1)[1] in openings]
    if not chosen:
        return None
    return summarize(chosen, player_id, seed=statistics_["seed"], bootstrap_samples=statistics_["samples"])


def _check_validation_plans(candidates, plans):
    """Every validation plan present must follow one frozen protocol."""
    first = next(iter(plans.values()))
    for n, plan in plans.items():
        if (plan["kind"] != "validation" or plan["candidates_file"] != candidates["file"]
                or plan["opening_ids"] != first["opening_ids"] or plan["openings_set_sha256"] != first["openings_set_sha256"]
                or plan["settings"] != first["settings"] or plan["ply_cap"] != first["ply_cap"]
                or plan["protocol"] != first["protocol"] or plan["seed"] != first["seed"]
                or plan["runtime"] != first["runtime"] or plan["sources"] != first["sources"]
                or plan["device"] != first["device"] or plan["workers"] != first["workers"]
                or plan["score_player_id"] != n or plan["players"][1]["sha256"] != candidates["by_name"][n]["sha256"]):
            raise ValueError(f"{n}: validation plan differs from the frozen protocol")
    return first


def _thresholds(plan):
    planned = plan["planned_pairs"]
    if plan["protocol"] == "production":
        return FINAL_PAIRS, PROVISIONAL_MIN_PAIRS
    return planned, math.ceil(0.75 * planned)


def _own(plan, games, name):
    return summarize(games, name, seed=plan["statistics"]["seed"], bootstrap_samples=plan["statistics"]["samples"],
                     allow_partial=len(games) != plan["planned_games"])


def compare_on_common_pairs(names, matches):
    """Each named candidate's score on the openings complete in all of their matches (the same pairs)."""
    common = common_pairs([matches[n][1] for n in names])
    first = matches[names[0]][0]
    _, provisional_min = _thresholds(first)
    out = dict(candidates=list(names), common_count=len(common),
               common_complete_pairs=sorted(common, key=first["opening_ids"].index), scores=None)
    if len(common) >= provisional_min:
        out["scores"] = {n: restricted_score(matches[n][1], n, common, matches[n][0]["statistics"]) for n in names}
    else:
        out["note"] = f"fewer than {provisional_min} common complete pairs; no comparison"
    return out


def decide(candidates, matches, *, matches_root=None):
    """Pure selection rule. ``matches`` maps candidate name to (plan, recorded games in schedule order).

    Only the selectable candidates' validation matches decide: they must be present and finished, their
    failures block, and their common complete pairs set the thresholds. A report-only match never blocks,
    shrinks or changes the selection; it is reported beside it (finished or not). A round-2 record (its
    ``round`` is 2) uses the round-2 plan check: carried-over round-1 matches and the audited source
    exception, whose audit is written into the result."""
    record, by_name = candidates["record"], candidates["by_name"]
    if set(matches) - set(by_name):
        raise ValueError("a validation match names a candidate that is not frozen")
    selectable = [c["name"] for c in record["candidates"] if c["eligibility"] == "selectable"]
    report_only = [c["name"] for c in record["candidates"] if c["eligibility"] == "report-only"]
    if [n for n in selectable if n not in matches]:
        raise ValueError("selection needs every selectable candidate's validation match")
    plans = {n: matches[n][0] for n in matches}
    audit = None
    if record.get("round") == 2:
        if matches_root is None:
            raise ValueError("a round-2 selection needs the matches root for the equivalence evidence")
        first, audit = _check_validation_plans_round2(candidates, plans, matches_root)
    else:
        first = _check_validation_plans(candidates, plans)
    for n in selectable:
        if len(matches[n][1]) != plans[n]["planned_games"]:
            raise ValueError(f"{n}: validation schedule unfinished; selection refuses")
    failures = {n: sum(g["status"] in ("failed", "interrupted") for g in matches[n][1]) for n in selectable}
    common = common_pairs([matches[n][1] for n in selectable])
    planned = first["planned_pairs"]
    final_needed, provisional_min = _thresholds(first)
    own = {n: _own(plans[n], matches[n][1], n) for n in selectable}
    attrition = {n: dict(status_counts=own[n]["status_counts"], complete_pairs=own[n]["complete_pairs"])
                 for n in selectable}
    out = dict(schema=SELECTION_SCHEMA, protocol=first["protocol"], candidates_file=candidates["file"],
               validation_plans={n: dict(match_id=plans[n]["match_id"], plan_sha256=plans[n]["sha256"]) for n in plans},
               decided_by=selectable, failures=failures,
               common_complete_pairs=sorted(common, key=first["opening_ids"].index),
               common_count=len(common), planned_pairs=planned, final_pairs_required=final_needed,
               provisional_min_pairs=provisional_min, attrition=attrition, own_match=None,
               eligibility={n: by_name[n]["eligibility"] for n in by_name}, tie_order=record["tie_order"],
               rules=record["rules"], final=False, selected=None, demo=None, scores_on_common_set=None,
               report_only=None)
    if audit is not None:
        out.update(round=2, previous_round=record["previous_round"], cross_revision=audit,
                   reused_matches={n: by_name[n]["validation_match"] for n in by_name
                                   if by_name[n].get("carried_from_round") == 1 and n in plans})
    if any(failures.values()):
        out["status"] = "blocked: failed or interrupted games must be diagnosed; nothing is replaced"
        return out
    if len(common) < provisional_min:
        out["status"] = (f"insufficient: {len(common)} common complete pairs, below the {provisional_min} needed "
                         "for a provisional report; no comparison; Arvind decides")
        return out
    scores = {n: restricted_score(matches[n][1], n, common, plans[n]["statistics"]) for n in selectable}
    out["scores_on_common_set"] = scores
    out["own_match"] = own
    # Report-only candidates: reported beside the selection, never part of it.
    side = {}
    for n in report_only:
        if n not in matches or len(matches[n][1]) != plans[n]["planned_games"]:
            side[n] = dict(status="pending: match not finished; it does not affect the selection")
            continue
        games = matches[n][1]
        side[n] = dict(status="finished", failures=sum(g["status"] in ("failed", "interrupted") for g in games),
                       own_match=_own(plans[n], games, n),
                       comparison=compare_on_common_pairs(selectable + [n], matches))
    out["report_only"] = side
    if len(common) < final_needed:
        out["status"] = (f"provisional: {len(common)} of {planned} common complete pairs (final selection needs "
                         f"{final_needed}); no automatic selection; Arvind decides")
        return out
    best = max(scores[n]["complete_pair_score"] for n in selectable)
    tied = [n for n in selectable if scores[n]["complete_pair_score"] == best]
    order = record["tie_order"] + [n for n in selectable if n not in record["tie_order"]]
    chosen = next(n for n in order if n in tied)
    spec = by_name[chosen]
    out.update(final=True, status="final", selected=dict(name=chosen, path=spec["path"], sha256=spec["sha256"],
               checkpoint_id=spec["checkpoint_id"], validation_score=best, tied_with=[n for n in tied if n != chosen]),
               demo=dict(player=chosen if best > 0.5 else "P", rule=DEMO_RULE),
               note=("the selected candidate's validation interval is biased upward by the selection; only the "
                     "test match supports a claim"))
    return out


def select(matches_root, root, *, candidates_path, round_number=1, log=print):
    """Apply the frozen rule to every finished validation store. A final selection is written once to
    selection.json; any other outcome (blocked, insufficient, provisional) goes to selection-status.json,
    which is derived and can be rewritten, and needs Arvind's decision; the stores never change.
    Round 2 writes selection-round2.json / selection-round2-status.json from candidates-round2.json."""
    matches_root = Path(matches_root)
    paths = round_paths(matches_root, round_number)
    output = paths["selection"]
    if output.exists():
        raise ValueError(f"{output} already exists; the selection is recorded once")
    candidates = load_candidates(candidates_path)
    if candidates["record"].get("round", 1) != round_number:
        raise ValueError(f"{candidates_path} is not the round-{round_number} candidate record")
    matches = {}
    for n, spec in candidates["by_name"].items():
        directory = matches_root / spec.get("validation_match", f"validation-P-vs-{n}")
        if not (directory / "plan.json").exists():
            if spec["eligibility"] == "selectable":
                raise ValueError(f"{n}: no validation match planned; selection refuses")
            continue  # a report-only match never holds up the selection
        plan = load_plan(directory, root, check_sources=False)
        values = read_results(directory / "results.sqlite3", plan)
        matches[n] = (plan, [values[j["game_id"]]["record"] for j in plan["schedule"] if j["game_id"] in values])
    current = {n: m[0]["sources"] == source_hashes(root, sorted(m[0]["sources"])) for n, m in matches.items()}
    record = sealed(dict(decide(candidates, matches, matches_root=matches_root), sources_match_plans=current,
                         created_utc=_now()))
    if record["final"]:
        publish_once(output, record)
    else:
        pilot.atomic_json(paths["status"], record)
    log(json.dumps(dict(event="selection", status=record["status"],
                        selected=None if record["selected"] is None else record["selected"]["name"])))
    return load_selection(output) if record["final"] else record
