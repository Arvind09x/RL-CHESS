"""Frozen development strength check; preparation/reporting never plays games."""
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import time

import chess

from chess_rl.adapter import make_state, terminal_result
from .matches import Opening, digest, play_game, snapshot, state_trace, summarize, turn_seed
from .players import Checkpoint, SearchPlayer, SearchSettings

COMPARISON = "supervised-candidate-versus-untrained"
MATCH_ID = "p02-cpu-001-strength-dev-v1"
SEED = 2026101202
FIELDS = ("opening_id", "initial_fen", "moves_uci", "history_complete", "source")
SETTINGS = SearchSettings(32).to_dict()

# Independently authored development lines; no dataset, labels or reserved test
# openings are read. Four branches per family give 64 distinct replay roots.
FAMILIES = (
    ("italian", "e4 e5 Nf3 Nc6 Bc4 Bc5", ("c3", "d3", "O-O", "b4")),
    ("ruy-lopez", "e4 e5 Nf3 Nc6 Bb5 a6", ("Ba4", "Bxc6", "Bc4", "Be2")),
    ("scotch", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6", ("Nc3", "Nxc6", "Be3", "Nb3")),
    ("petrov", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4", ("d4", "Nc3", "Qe2", "Bd3")),
    ("sicilian", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6", ("Be3", "Bg5", "Be2", "Bc4")),
    ("french", "e4 e6 d4 d5", ("Nc3", "Nd2", "e5", "exd5")),
    ("caro-kann", "e4 c6 d4 d5", ("Nc3", "Nd2", "e5", "exd5")),
    ("pirc", "e4 d6 d4 Nf6 Nc3 g6", ("f4", "Nf3", "Be3", "Bg5")),
    ("queens-gambit-declined", "d4 d5 c4 e6 Nc3 Nf6", ("Bg5", "Nf3", "cxd5", "Bf4")),
    ("slav", "d4 d5 c4 c6 Nf3 Nf6", ("Nc3", "e3", "cxd5", "Qc2")),
    ("queens-gambit-accepted", "d4 d5 c4 dxc4 Nf3 Nf6", ("e3", "Nc3", "Qa4+", "g3")),
    ("nimzo-indian", "d4 Nf6 c4 e6 Nc3 Bb4", ("e3", "Qc2", "a3", "f3")),
    ("english", "c4 e5 Nc3 Nf6", ("g3", "Nf3", "e3", "d3")),
    ("reti", "Nf3 d5 g3 Nf6 Bg2 e6", ("O-O", "c4", "d3", "b3")),
    ("kings-indian", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6", ("Nf3", "f3", "f4", "Be2")),
    ("dutch", "d4 f5 c4 Nf6", ("g3", "Nc3", "Nf3", "e3")),
)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Publish a complete, fsynced JSON snapshot, including the directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_identity():
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), torch=importlib.metadata.version("torch"),
                numpy=importlib.metadata.version("numpy"), chess=chess.__version__,
                device="cpu", dtype="float32", threads=1, interop_threads=1,
                deterministic=True, matmul_precision="highest")


def authored_openings():
    openings = []
    for family, prefix, branches in FAMILIES:
        for index, branch in enumerate(branches):
            board = chess.Board()
            for san in (prefix + " " + branch).split():
                board.push_san(san)
            o = Opening.from_state(f"{family}-{index + 1}", make_state(
                moves_uci=[m.uci() for m in board.move_stack]),
                source=f"independently-authored-development-v1:{family}; no corpus/labels")
            if terminal_result(o.state()) is not None:
                raise ValueError("Development opening must be nonterminal")
            openings.append(dict(**o.to_dict(), family=family, san=prefix + " " + branch))
    if len(openings) != 64 or len({opening(o).rule_key() for o in openings}) != 64:
        raise ValueError("Expected 64 distinct opening roots")
    return snapshot(openings)


def opening(raw):
    return Opening(**{key: raw[key] for key in FIELDS})


def source_files(root):
    return sorted([*root.joinpath("src").rglob("*.py"),
                   root / "scripts/check_p02_strength.py", root / "pyproject.toml", root / "uv.lock"])


def prepare(root, destination):
    """Freeze settings/openings/identities only. No evaluator or game calls."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Plan already exists; never overwrite a frozen plan")
    archive = root / "artifacts/p02-cpu-001-preserved/manifest.json"
    preserved = json.loads(archive.read_text())
    checkpoints = {}
    for role, step in (("supervised-candidate", 1410), ("untrained", 0)):
        relative = f"artifacts/p02-cpu-001/step-{step:06d}.pt"
        expected = preserved["checkpoints"][relative]["sha256"]
        if sha_file(root / relative) != expected:
            raise ValueError("Checkpoint differs from preserved completed run")
        checkpoints[role] = dict(path=relative, sha256=expected,
                                checkpoint_id=f"pilot-step-{step}", kind=role)
    plan = dict(schema="p02-strength-plan-v1", match_id=MATCH_ID, comparison=COMPARISON,
                split="development", project_P_selected=False, seed=SEED,
                planned_pairs=64, planned_games=128, workers=4,
                settings=SETTINGS, ply_cap=512,
                pairing="identical-replay-root; candidate-white-leg0; candidate-black-leg1",
                seed_algorithm="sha256(match_id,master_seed,opening_id,leg)-v1",
                checkpoints=checkpoints, openings=authored_openings(),
                runtime=runtime_identity(), preservation=dict(path=str(archive.relative_to(root)),
                    sha256=sha_file(archive)),
                sources={str(p.relative_to(root)): sha_file(p) for p in source_files(root)},
                statistics=dict(bootstrap_seed=SEED + 1, bootstrap_samples=10000,
                    confidence=0.95, unit="opening-pair", target_score=0.70,
                    scoring="completed-rule-outcomes-only; complete-pairs-only for uncertainty",
                    stopping="fixed 128 scheduled games; no replacements or outcome-dependent stopping"))
    plan["schedule"] = list(jobs(plan))
    atomic_json(destination, dict(**plan, sha256=digest(plan)))
    return load_plan(destination, root)


def load_plan(path, root):
    plan = json.loads(Path(path).read_text())
    content = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != digest(content):
        raise ValueError("Frozen plan digest mismatch")
    if (plan["schema"] != "p02-strength-plan-v1" or plan["comparison"] != COMPARISON
            or plan["split"] != "development" or plan["project_P_selected"] is not False
            or plan["settings"] != SETTINGS or plan["ply_cap"] != 512
            or plan["planned_pairs"] != 64 or plan["planned_games"] != 128 or plan["workers"] != 4):
        raise ValueError("Unsupported pilot strength protocol")
    if plan["openings"] != authored_openings():
        raise ValueError("Opening freeze differs from authored development plan")
    if plan["schedule"] != list(jobs(plan)):
        raise ValueError("Frozen game seeds/pairing differ from scheduled games")
    if plan["runtime"] != runtime_identity():
        raise ValueError("Runtime differs from the frozen environment")
    root = Path(root).resolve()
    actual_sources = {str(p.relative_to(root)): sha_file(p) for p in source_files(root)}
    if actual_sources != plan["sources"]:
        raise ValueError("Evaluation source differs from frozen plan; do not mix results across revisions")
    preserved = plan["preservation"]
    if sha_file(root / preserved["path"]) != preserved["sha256"]:
        raise ValueError("Preservation manifest changed")
    archive = json.loads((root / preserved["path"]).read_text())
    for relative, identity in archive["files"].items():
        if sha_file((root / preserved["path"]).parent / "files" / relative) != identity["sha256"]:
            raise ValueError("Preserved source/configuration archive changed")
    for role, step in (("supervised-candidate", 1410), ("untrained", 0)):
        checkpoint = plan["checkpoints"][role]
        relative = f"artifacts/p02-cpu-001/step-{step:06d}.pt"
        if (checkpoint["kind"] != role or checkpoint["path"] != relative
                or checkpoint["checkpoint_id"] != f"pilot-step-{step}"
                or checkpoint["sha256"] != archive["checkpoints"][relative]["sha256"]
                or sha_file(root / relative) != checkpoint["sha256"]):
            raise ValueError("Checkpoint routing/hash mismatch")
    return plan


def jobs(plan):
    for raw in plan["openings"]:
        pair_id = f"{plan['match_id']}:{raw['opening_id']}"
        for leg in (0, 1):
            yield dict(game_id=f"{raw['opening_id']}:leg{leg}", pair_id=pair_id, leg=leg,
                game_seed=int(digest([plan["match_id"], plan["seed"], raw["opening_id"], leg])[:16], 16),
                opening=raw)


def load_players(plan, root):
    """Always bind the two real checkpoint files to production RealEvaluator."""
    from chess_rl.search.real_evaluator import RealEvaluator
    from chess_rl.training.checkpoints import ModelConfig
    players = []
    for role in ("supervised-candidate", "untrained"):
        raw = plan["checkpoints"][role]
        path = Path(root) / raw["path"]
        identity = Checkpoint(raw["checkpoint_id"], str(path.resolve()), raw["sha256"], role)
        identity.verify()
        evaluator = RealEvaluator(path, expected_config=ModelConfig(), device="cpu")
        loaded = evaluator.checkpoint
        if loaded.checkpoint_id != raw["checkpoint_id"]:
            raise ValueError("Loaded checkpoint ID mismatch")
        if loaded.provenance["purpose"] != "supervised pilot; P selection remains a separate decision":
            raise ValueError("Loaded checkpoint provenance mismatch")
        identity.verify()  # Also reject a file changed while loading.
        players.append(SearchPlayer(role, role, f"real-evaluator:{raw['sha256']}",
                                    evaluator, SearchSettings(32), identity))
    if (players[0].settings != players[1].settings
            or asdict(players[0].evaluator.checkpoint.config) != asdict(players[1].evaluator.checkpoint.config)
            or players[0].evaluator.checkpoint.provenance != players[1].evaluator.checkpoint.provenance):
        raise ValueError("Models must share architecture, provenance and search settings")
    return tuple(players)


def connect(path):
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA synchronous=FULL")
    return db


def initialize_store(path, plan):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS games (game_id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        saved = db.execute("SELECT value FROM metadata WHERE key='plan'").fetchone()
        value = json.dumps(plan, sort_keys=True, allow_nan=False)
        if saved is None:
            db.execute("INSERT INTO metadata VALUES ('plan', ?)", (value,))
        elif saved[0] != value:
            raise ValueError("Result database belongs to a different frozen plan")
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish(path, job, plan, record, elapsed, runtime):
    """Worker commits before notifying parent; a killed parent cannot lose it."""
    validate_game(record, job, plan)
    value = dict(game_id=job["game_id"], plan_sha256=plan["sha256"],
                 elapsed_seconds=elapsed, runtime=runtime, record=record)
    payload = json.dumps(value, sort_keys=True, allow_nan=False)
    with connect(path) as db:
        # No replacement of completed, truncated, failed or interrupted records.
        db.execute("INSERT INTO games VALUES (?, ?, ?)", (job["game_id"], payload, digest(value)))


def validate_game(record, job, plan):
    if (record["pair_id"] != job["pair_id"] or record["leg"] != job["leg"]
            or record["game_seed"] != job["game_seed"] or record["ply_cap"] != 512
            or record["opening"] != snapshot(opening(job["opening"]).to_dict())):
        raise ValueError("Stored game differs from scheduled pair/seed/opening")
    roles = ("supervised-candidate", "untrained")
    for color, role in zip(("white", "black"), roles[::1 if job["leg"] == 0 else -1], strict=True):
        meta = record["players"][color]
        identity = plan["checkpoints"][role]
        if (meta["player_id"] != role or meta["role"] != role or meta["kind"] != "search"
                or meta["settings"] != SETTINGS or meta["checkpoint"]["kind"] != role
                or meta["checkpoint"]["sha256"] != identity["sha256"]
                or meta["checkpoint"]["checkpoint_id"] != identity["checkpoint_id"]
                or meta["evaluator_id"] != f"real-evaluator:{identity['sha256']}"):
            raise ValueError("Stored game checkpoint/color routing mismatch")
    state = opening(job["opening"]).state()
    for i, attempt in enumerate(record["attempts"]):
        ply = len(state.board.move_stack)
        color = "white" if state.board.turn else "black"
        role = record["players"][color]["player_id"]
        if (terminal_result(state) is not None or ply >= 512 or attempt["ply"] != ply
                or attempt["color"] != color or attempt["player_id"] != role
                or attempt["seed"] != turn_seed(job["game_seed"], ply, role)):
            raise ValueError("Stored move routing/history mismatch")
        if attempt["applied"]:
            from chess_rl.adapter import apply_move, id_to_move
            details = attempt["details"]
            edges = details["visited_edges"]
            if (attempt["status"] != "completed" or details["requested_simulations"] != 32
                    or details["completed_simulations"] != 32
                    or sum(edge[1] for edge in edges) != 32
                    or max(edges, key=lambda e: (e[1], -e[0]))[0] != attempt["move_id"]
                    or id_to_move(state, attempt["move_id"]).uci() != attempt["move_uci"]):
                raise ValueError("Stored applied search/move mismatch")
            apply_move(state, attempt["move_id"])
        elif (i != len(record["attempts"]) - 1 or attempt["status"] != record["status"]
              or record["status"] not in ("failed", "interrupted")):
            raise ValueError("Unapplied move must end a failed/interrupted game")
    if any(record[k] != v for k, v in state_trace(state).items()):
        raise ValueError("Stored final history mismatch")
    terminal = terminal_result(state)
    if record["status"] == "completed":
        if terminal is None:
            raise ValueError("Only actual terminal outcomes can be scored")
        winner = None if terminal.winner is None else "white" if terminal.winner else "black"
        if (record["winner"] != winner or record["outcome"] != {None: "1/2-1/2", "white": "1-0", "black": "0-1"}[winner]
                or record["ending_reason"] != terminal.ending_reason):
            raise ValueError("Stored rule outcome mismatch")
    else:
        if record["winner"] is not None or record["outcome"] is not None:
            raise ValueError("Truncated/failed/interrupted games have no score")
        if record["status"] == "truncated":
            if terminal is not None or len(state.board.move_stack) < 512 or record["ending_reason"] != "ply_limit":
                raise ValueError("Invalid truncation")
        elif record["status"] not in ("failed", "interrupted") or record["error"] is None:
            raise ValueError("Invalid failure status")


def read_results(path, plan):
    with connect(path) as db:
        saved = db.execute("SELECT value FROM metadata WHERE key='plan'").fetchone()
        if saved is None or json.loads(saved[0]) != snapshot(plan):
            raise ValueError("Database plan mismatch")
        rows = db.execute("SELECT game_id, payload, sha256 FROM games ORDER BY game_id").fetchall()
    scheduled = {j["game_id"]: j for j in jobs(plan)}
    values = {}
    for game_id, payload, checksum in rows:
        value = json.loads(payload)
        if (game_id not in scheduled or value["game_id"] != game_id
                or value["plan_sha256"] != plan["sha256"] or checksum != digest(value)
                or value["runtime"] != plan["runtime"]):
            raise ValueError("Stored result identity/digest/runtime mismatch")
        validate_game(value["record"], scheduled[game_id], plan)
        values[game_id] = value
    return values


def report(path, plan):
    values = read_results(path, plan)
    summary = summarize([v["record"] for v in values.values()], "supervised-candidate",
        seed=plan["statistics"]["bootstrap_seed"],
        bootstrap_samples=plan["statistics"]["bootstrap_samples"], allow_partial=True)
    counts = summary["status_counts"]
    attempted = sum(counts.values())
    summary.update(comparison=COMPARISON, split="development", project_P_selected=False,
        plan_sha256=plan["sha256"], planned_games=128, planned_pairs=64,
        recorded_games=attempted, pending_games=128 - attempted,
        status_rates_among_recorded={k: None if not attempted else v / attempted for k, v in counts.items()},
        status_fractions_of_planned={k: v / 128 for k, v in counts.items()},
        missing_or_incomplete_pairs=64 - summary["complete_pairs"],
        uncertainty_caveat="Conditional on fully completed opening pairs; attrition and shared opening families limit inference; bootstrap may degenerate for identical scores.")
    n = summary["complete_pairs"]
    score = summary["complete_pair_score"]
    # Pair scores are bounded [0,1]. This bound stays nonzero even when every
    # observed pair has the same score. Independence remains an assumption.
    radius = None if not n else math.sqrt(math.log(40) / (2 * n))
    summary["pair_hoeffding_95_interval"] = None if not n else [max(0, score - radius), min(1, score + radius)]
    summary["pair_hoeffding_assumption"] = "Independent representative opening pairs; fixed authored suite/family dependence may violate this. No population strength or P acceptance claim."
    return summary


@contextmanager
def run_lock(output):
    """Reject concurrent coordinators, including orphaned active workers."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as coordinator, (output / "workers.lock").open("a+") as workers:
        try:
            fcntl.flock(coordinator, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(workers, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another coordinator or game worker is active for this output") from exc
        fcntl.flock(workers, fcntl.LOCK_SH)
        yield


_WORKER = None
_WORKER_LOCK = None


def initialize_worker(plan, root, database):
    global _WORKER, _WORKER_LOCK
    # Keep the output protected if the coordinator dies while a worker lives.
    _WORKER_LOCK = Path(database).with_name("workers.lock").open("a+")
    fcntl.flock(_WORKER_LOCK, fcntl.LOCK_SH)
    from chess_rl.runtime import configure
    configure("cpu", threads=1)
    _WORKER = (plan, database, load_players(plan, root), runtime_identity())


def execute_job(job):
    plan, database, players, runtime = _WORKER
    # Durable finished records are immutable and never dispatched on resume.
    white, black = players if job["leg"] == 0 else players[::-1]
    started = time.monotonic()
    record = play_game(opening(job["opening"]), white, black, pair_id=job["pair_id"],
                       leg=job["leg"], game_seed=job["game_seed"], ply_cap=plan["ply_cap"])
    publish(database, job, plan, record, time.monotonic() - started, runtime)
    return job["game_id"], record["status"]


def pending_jobs(plan, values):
    return [j for j in jobs(plan) if j["game_id"] not in values]


def run(plan, root, output):
    output = Path(output).resolve()
    database = output / "results.sqlite3"
    with run_lock(output):
        initialize_store(database, plan)
        # Retain a human-readable plan alongside the authoritative DB copy.
        atomic_json(output / "plan.json", plan)
        values = read_results(database, plan)
        pending = pending_jobs(plan, values)
        print(json.dumps(dict(resuming_recorded=len(values), pending=len(pending), workers=4)), flush=True)
        try:
            if pending:
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    os.environ[name] = "1"
                with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_worker, initargs=(plan, str(root), str(database))) as pool:
                    futures = [pool.submit(execute_job, job) for job in pending]
                    for future in as_completed(futures):
                        game_id, status = future.result()
                        print(json.dumps(dict(game_id=game_id, status=status, durable=True)), flush=True)
        except BaseException as exc:
            with connect(database) as db:
                db.execute("INSERT INTO events(payload) VALUES (?)", (json.dumps(dict(
                    event="coordinator_stopped", type=type(exc).__name__, message=str(exc))),))
            raise
        finally:
            summary = report(database, plan)
            atomic_json(output / "summary.json", summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
    return summary
