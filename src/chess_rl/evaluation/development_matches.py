"""Isolated synthetic development match plans, never production candidate records."""
from pathlib import Path
import random

from chess_rl.adapter import make_state, terminal_result
from . import match_campaign as M
from .matches import Opening
from .execution import execution_for

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
import platform
import signal
import time

from .execution import Execution
from .players import SearchPlayer
from .match_campaign import (SearchSettings, _unavailable, _now, sha_file, search_player,
                             opponent_player, snapshot, runtime_identity, worker_runtime, opening, play_game,
                             run_lock, initialize_store, read_results, verify_inputs, connect, publish, status)
from . import pilot

_WORKER = None
_WORKER_LOCK = None
_COLLECTOR = None



def freeze(directory, root, checkpoints, execution, runtime, *, pairs=8, simulations=128, ply_cap=512):
    import chess
    from chess_rl.training.checkpoints import load_inference
    directory, root = Path(directory), Path(root).resolve()
    if (directory / 'plan.json').exists():
        raise ValueError('development plan already exists; use run to resume')
    if pairs < 1 or pairs > 128:
        raise ValueError('development pairs must be in [1,128]')
    players = []
    settings = SearchSettings(simulations)
    for side, path in zip(('left','right'), checkpoints, strict=True):
        path = Path(path).resolve()
        sha = M.sha_file(path)
        loaded = load_inference(path, device='cpu')
        spec = dict(name=side, path=str(path), sha256=sha, checkpoint_id=loaded.checkpoint_id,
                    kind='rl-R')
        if M.sha_file(path) != sha:
            raise ValueError('checkpoint changed while freezing development plan')
        player = M.search_player(spec, root, settings)
        players.append(dict(side=side, type='search', metadata=M.snapshot(player.metadata()), **spec))
        del loaded
    # Identical synthetic roots and seeds for every execution layout. These are
    # generated without accessing any data/heldout or campaign opening files.
    rng, openings = random.Random(2026101808), []
    while len(openings) < pairs:
        board = chess.Board()
        for _ in range(8):
            board.push(rng.choice(sorted(board.legal_moves, key=lambda m:m.uci())))
        state = make_state(moves_uci=[m.uci() for m in board.move_stack])
        if terminal_result(state) is None:
            openings.append(Opening.from_state(f'development-{len(openings):03}', state,
                             source='seeded synthetic development v1; no registry access').to_dict())
    plan = dict(schema=M.PLAN_SCHEMA, protocol='fixture', kind='smoke', match_id='development-match',
                split='development', workers=execution.processes, execution=execution.to_dict(),
                device=runtime['device'], runtime=runtime, settings=settings.to_dict(), players=players,
                openings=openings, opening_ids=[o['opening_id'] for o in openings],
                planned_pairs=len(openings), planned_games=2*len(openings), seed=2026101809,
                ply_cap=ply_cap, openings_file=None, candidates_file=None, selection_file=None,
                sources=M.source_hashes(root), development_benchmark=dict(warmup_calls=3), created_utc=M._now())
    execution_for(plan)
    plan['schedule'] = list(M.jobs(plan))
    M.publish_once(directory / 'plan.json', M.sealed(plan))
    return load_plan(directory, root)



def load_plan(directory, root, *, check_sources=True):
    plan = M.load_plan(directory, root, check_sources=check_sources)
    if plan['protocol'] != 'fixture' or plan['split'] != 'development':
        raise ValueError('standalone runner accepts synthetic development plans only')
    execution_for(plan)
    return plan


def timing_report(plan, values):
    report = M.timing_report(plan, values)
    report['concurrent_games'] = min(plan['workers'] * execution_for(plan).games_per_process, plan['planned_games'])
    return report

def load_reference(entry, root, device):
    """Hash before/after loading; called on the model-owning collector thread."""
    from chess_rl.search.real_evaluator import RealEvaluator
    path = Path(root) / entry["path"]
    if sha_file(path) != entry["sha256"]:
        raise ValueError(f"{entry['path']}: checkpoint bytes differ before loading")
    evaluator = RealEvaluator(path, device=device)
    if evaluator.checkpoint.checkpoint_id != entry["checkpoint_id"]:
        raise ValueError("loaded checkpoint id mismatch")
    if sha_file(path) != entry["sha256"]:
        raise ValueError(f"{entry['path']}: checkpoint changed while loading")
    return evaluator


def load_players(plan, root, *, deferred=False):
    """Bind each search player to its verified checkpoint bytes; both networks load once per worker."""
    settings = SearchSettings(plan["settings"]["simulation_budget"])
    players = []
    for entry in plan["players"]:
        if entry["type"] == "search":
            evaluator = _unavailable if deferred else load_reference(entry, root, plan["device"])
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


def initialize_worker(plan, root, database, ready_barrier=None):
    global _WORKER, _WORKER_LOCK, _COLLECTOR
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
    batched = execution_for(plan).mode == "batched"
    _WORKER = (plan, database, load_players(plan, root, deferred=batched), runtime)
    if batched:
        from .concurrent_matches import collector_for
        _COLLECTOR = collector_for(plan, root)
    if plan.get("development_benchmark"):
        if batched:
            _COLLECTOR.wait_ready()
        else:
            from chess_rl.adapter import make_state
            for player in _WORKER[2]:
                if isinstance(player, SearchPlayer):
                    for _ in range(plan["development_benchmark"]["warmup_calls"]):
                        player.evaluator([make_state()])
        if ready_barrier is not None:
            ready_barrier.wait(timeout=120)


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
    if plan.get("development_benchmark"):
        write_worker_memory(database, plan)
    return job["game_id"], record["status"]


def write_worker_memory(database, plan):
    import resource
    import sys
    metrics = dict(process_id=os.getpid(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if sys.platform == "darwin" else 1024))
    if plan["device"] == "cuda:0":
        import torch
        metrics.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    pilot.atomic_json(Path(database).with_name(f"worker-memory-{os.getpid()}.json"), metrics)


def execute_lane(jobs):
    from .concurrent_matches import execute_games
    plan, database, players, runtime = _WORKER
    outcomes = execute_games(jobs, plan, database, players, runtime, _COLLECTOR)
    # Separate diagnostics: the immutable per-game record schema is unchanged.
    write_worker_memory(database, plan)
    metrics = dict(_COLLECTOR.metrics)
    metrics["process_id"] = os.getpid()
    if plan["device"] == "cuda:0":
        import torch
        metrics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    pilot.atomic_json(Path(database).with_name(f"collector-{os.getpid()}.json"), metrics)
    return outcomes


def pending_jobs(plan, values):
    return [j for j in plan["schedule"] if j["game_id"] not in values]


def run(directory, root, *, execution=None, log=print):
    """Play every game without a stored row; resumable after any kill."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    plan = load_plan(directory, root)
    frozen_execution = execution_for(plan)
    if execution is not None and execution != frozen_execution:
        raise ValueError("requested execution differs from frozen plan")
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
                context = multiprocessing.get_context("spawn")
                barrier = context.Barrier(min(plan["workers"], len(pending))) if plan.get("development_benchmark") else None
                pool = ProcessPoolExecutor(max_workers=min(plan["workers"], len(pending)),
                                           mp_context=multiprocessing.get_context("spawn"),
                                           initializer=initialize_worker,
                                           initargs=(plan, str(root), str(database), barrier))
                clean = False
                try:
                    batched = frozen_execution.mode == "batched"
                    if batched:
                        lanes = min(plan["workers"], len(pending))
                        futures = [pool.submit(execute_lane, pending[i::lanes]) for i in range(lanes)]
                    else:
                        futures = [pool.submit(execute_job, job) for job in pending]
                    lane_errors = []
                    for future in as_completed(futures):
                        try:
                            outcomes = future.result() if batched else [future.result()]
                        except Exception as exc:
                            if not batched:
                                raise
                            lane_errors.append(exc)
                            continue
                        for game_id, outcome_status in outcomes:
                            log(json.dumps(dict(event="game", game_id=game_id, status=outcome_status, durable=True,
                                                utc=_now())))
                    clean = True
                    if lane_errors:
                        raise RuntimeError(f"{len(lane_errors)} match worker lanes failed") from lane_errors[0]
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
