"""Self-play campaigns: many resumable games from one frozen network.

One directory per campaign. create() freezes the plan (network bytes, search
settings, opening, per-game seeds) before any game exists. run() plays every
unfinished game in a refilled process pool; each game is a journaled Session
directory, so a crash loses at most the move being searched and finished games
are never rewritten. archive() audits every finished game and writes one game
per JSONL line for scripts/run_r02.py. Game directories are the source of
truth; events.jsonl is a diagnostic log only.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import time

import chess

from chess_rl.evaluation.matches import Opening, digest
from chess_rl.selfplay import Phase, Session, Settings, _read, _write

SCHEMA = "selfplay-campaign-v1"
FINAL = ("completed", "truncated")
MAX_PASSES = 3  # a pass ends early if a worker dies; the next resumes every unfinished game
STANDARD_OPENING = Opening("standard-start", chess.STARTING_FEN, (), True,
                           "standard initial position; variety from move sampling and root noise")
# Phase bins for per-move timing, by ply counted from the initial position.
PHASES = (("opening", 0, 30), ("middle", 30, 100), ("late", 100, None))


def frozen_settings(simulations):
    """Frozen round settings; only the simulation budget is a campaign choice."""
    return Settings(simulation_budget=simulations, ply_cap=512, c_puct=1.5, unvisited_value=0.0,
                    phases=(Phase(0, 1.0, 0.25, 0.3), Phase(30, 0.0, 0.25, 0.3)))


def game_seed(seed, index):
    return int(digest(["selfplay-campaign-game-v1", seed, index])[:16], 16)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL)
        return dict(commit=commit.decode().strip(), dirty=bool(dirty.strip()))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def create(directory, checkpoint, *, games, seed, purpose, simulations=None, settings=None,
           device="cpu", opening=STANDARD_OPENING):
    """Freeze a new campaign; plays nothing. Tests pass small settings; the CLI passes simulations."""
    if (simulations is None) == (settings is None):
        raise ValueError("give exactly one of simulations or settings")
    settings = frozen_settings(simulations) if settings is None else settings
    if not isinstance(settings, Settings) or not isinstance(opening, Opening) or not opening.history_complete:
        raise ValueError("settings must be Settings and opening a complete-history Opening")
    for name, value, minimum in (("games", games, 1), ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a nonempty description")
    checkpoint = Path(checkpoint).resolve()
    plan = dict(schema=SCHEMA, created_utc=_now(), purpose=purpose, source=_source(),
        checkpoint=dict(path=str(checkpoint), sha256=_sha256(checkpoint), bytes=checkpoint.stat().st_size),
        device=device, settings=asdict(settings), opening=asdict(opening), seed=seed,
        games=[dict(index=i, directory=f"games/{i:04d}", seed=game_seed(seed, i)) for i in range(games)],
        note="Game records carry the fixed checkpoint role text from selfplay._checkpoint; purpose here is authoritative.")
    plan = json.loads(json.dumps(plan))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "games").mkdir()
    _write(directory / "campaign.json", plan)
    return plan


def load(directory):
    plan = _read(Path(directory) / "campaign.json")
    if plan.get("schema") != SCHEMA:
        raise ValueError("not a self-play campaign directory")
    return plan


def _header(path):
    boundary = Path(path) / "boundary.json"
    return _read(boundary) if boundary.exists() else None


def _finished(path):
    header = _header(path)
    return header is not None and header["status"] in FINAL


@contextmanager
def _lock(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{message} ({path})") from None
        yield


class _Events:
    """Append-only, fsynced JSONL; diagnostic only, never read for correctness."""
    def __init__(self, path):
        self.path = Path(path)

    def write(self, event, **fields):
        line = json.dumps(dict(event=event, time=time.time(), utc=_now(), **fields), sort_keys=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self):
        events = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut short by a hard kill
        return events


def _init_worker(device):
    # The parent owns stopping; a terminated worker loses at most its uncommitted search.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from chess_rl.runtime import configure
    configure(device, threads=1)


def _set_aside(path):
    # Only reached without boundary.json: creation stopped before any search, so no move exists.
    target = path.parent / f".abandoned-{path.name}-{time.time_ns()}"
    path.rename(target)
    return target


def _play(task):
    started = time.monotonic()
    path = Path(task["directory"])
    result = {}
    try:
        if (path / "boundary.json").exists():
            session = Session.load(path, checkpoint_path=task["checkpoint"])
        else:
            if path.exists():
                result["set_aside"] = str(_set_aside(path))
            session = Session.create(path, task["checkpoint"], Opening(**task["opening"]),
                settings=Settings.from_dict(task["settings"]), seed=task["seed"],
                opening_selection_seed=task["campaign_seed"], device=task["device"], journal=True)
        before = len(session.record["positions"])
        game = session.run()["game"]
        result.update(status=game["status"], ending_reason=game["ending_reason"], outcome=game["outcome"],
                      plies=len(game["moves_uci"]), new_moves=len(game["positions"]) - before)
        if game["status"] == "interrupted" and game["interruptions"]:
            last = dict(game["interruptions"][-1])
            last.pop("discarded_visits", None)
            result["interruption"] = last
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    return dict(index=task["index"], pid=os.getpid(), seconds=time.monotonic() - started, **result)


def _kill_workers(executor):
    # Executor has no public terminate; a killed worker loses at most its uncommitted search.
    for process in list((getattr(executor, "_processes", None) or {}).values()):
        if process.is_alive():
            process.kill()


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def run(directory, *, workers, heavy_lock=None, log=print):
    """Play or resume every unfinished game in up to MAX_PASSES passes; leftovers are reported.

    A worker that dies (killed, crashed, out of memory) breaks its pass: the other
    workers are stopped, every game keeps its committed moves, and the next pass
    resumes all unfinished games.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    directory = Path(directory).resolve()
    plan = load(directory)
    checkpoint = Path(plan["checkpoint"]["path"])
    if _sha256(checkpoint) != plan["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint bytes differ from the frozen campaign plan: {checkpoint}")
    with ExitStack() as stack:
        stack.enter_context(_lock(directory / ".campaign.lock", "another run of this campaign is active"))
        if heavy_lock is not None:
            stack.enter_context(_lock(heavy_lock, "another heavy job holds the lock"))
        previous = {s: signal.signal(s, _raise_interrupt) for s in (signal.SIGTERM, signal.SIGHUP)}
        stack.callback(lambda: [signal.signal(s, h) for s, h in previous.items()])
        events = _Events(directory / "events.jsonl")
        events.write("run-start", workers=workers, device=plan["device"], host=platform.node(), pid=os.getpid())
        context = multiprocessing.get_context("spawn")
        for attempt in range(1, MAX_PASSES + 1):
            tasks = [dict(index=g["index"], directory=str(directory / g["directory"]), seed=g["seed"],
                          checkpoint=str(checkpoint), opening=plan["opening"], settings=plan["settings"],
                          campaign_seed=plan["seed"], device=plan["device"])
                     for g in plan["games"] if not _finished(directory / g["directory"])]
            if not tasks:
                break
            log(f"pass {attempt}: {len(tasks)} unfinished games, {min(workers, len(tasks))} workers")
            executor = ProcessPoolExecutor(min(workers, len(tasks)), mp_context=context,
                                           initializer=_init_worker, initargs=(plan["device"],))
            broken, clean = None, False
            try:
                for future in as_completed([executor.submit(_play, task) for task in tasks]):
                    try:
                        result = future.result()
                    except BrokenProcessPool as error:
                        broken = error
                        continue
                    events.write("game-end", attempt=attempt, **result)
                    log(json.dumps(result, sort_keys=True))
                clean = broken is None
            except KeyboardInterrupt as stop:
                events.write("run-stopped", reason=str(stop) or "SIGINT")
                raise
            finally:
                if not clean:
                    _kill_workers(executor)
                executor.shutdown(wait=True, cancel_futures=True)
            if broken is not None:
                events.write("pool-broken", attempt=attempt, error=str(broken))
                log(f"pass {attempt}: a worker died ({broken}); unfinished games resume next pass")
        report = status(directory)
        events.write("run-end", status_counts=report["status_counts"])
        return report


def _move_seconds(path, opening_plies):
    """Per-move wall seconds from journal file times. A resumed game's first gap includes the stop;
    medians absorb that."""
    times = [(path / "configuration.json").stat().st_mtime]
    moves = sorted((path / "moves").glob("*.json"))
    times += [p.stat().st_mtime for p in moves]
    return [(opening_plies + i, b - a) for i, (a, b) in enumerate(zip(times, times[1:]))]


def status(directory):
    """Progress, game lengths, per-move timing by phase and a finish projection for this run."""
    directory = Path(directory)
    plan = load(directory)
    counts = dict(completed=0, truncated=0, interrupted=0, not_started=0)
    plies, reasons, outcomes, timing = [], Counter(), Counter(), []
    opening_plies = len(plan["opening"]["moves_uci"])
    for game in plan["games"]:
        path = directory / game["directory"]
        header = _header(path)
        if header is None:
            counts["not_started"] += 1
            continue
        counts[header["status"]] += 1
        timing += _move_seconds(path, opening_plies)  # unfinished games too, for early projections
        if header["status"] in FINAL:
            plies.append(len(header["moves_uci"]))
            reasons[header["ending_reason"]] += 1
            outcomes[header["outcome"] or "none"] += 1
    per_move = {}
    for name, low, high in PHASES:
        values = [s for ply, s in timing if ply >= low and (high is None or ply < high)]
        if values:
            per_move[name] = dict(moves=len(values), median_seconds=statistics.median(values))
    events = _Events(directory / "events.jsonl").read()
    starts = [e for e in events if e["event"] == "run-start"]
    projection = None
    if starts:
        start = starts[-1]
        ended = [e for e in events if e["time"] >= start["time"] and e["event"] in ("run-end", "run-stopped")]
        stop = ended[-1]["time"] if ended else time.time()
        done = [e for e in events if e["event"] == "game-end" and e["time"] >= start["time"]
                and e.get("status") in FINAL]
        hours = (stop - start["time"]) / 3600
        remaining = len(plan["games"]) - counts["completed"] - counts["truncated"]
        rate = len(done) / hours if hours > 0 else 0.0
        projection = dict(run_started_utc=start["utc"], workers=start["workers"], elapsed_hours=hours,
                          finished_this_run=len(done), games_per_hour=rate, remaining_games=remaining,
                          remaining_hours=remaining / rate if rate else None)
    return dict(games=len(plan["games"]), status_counts=counts,
                plies=dict(mean=statistics.mean(plies), median=statistics.median(plies), max=max(plies)) if plies else None,
                ending_reasons=dict(reasons), outcomes=dict(outcomes), per_move=per_move, projection=projection)


def archive(directory, output=None):
    """Audit every finished game into one JSONL line each, plus a summary written last.

    Idempotent: a complete archive (summary present and matching) is returned as is.
    """
    from chess_rl.replay import read_source
    directory = Path(directory)
    plan = load(directory)
    output = Path(output) if output else directory / "games.jsonl"
    summary_path = output.with_name(output.stem + ".summary.json")
    if output.exists():
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["archive_sha256"] != _sha256(output):
                raise ValueError(f"{output} does not match {summary_path}")
            return summary  # already complete
        # An earlier attempt stopped between publishing the archive and its summary; redo both.
        output.rename(output.with_name(f".{output.name}.incomplete-{time.time_ns()}"))
    unfinished = [g["index"] for g in plan["games"] if not _finished(directory / g["directory"])]
    if unfinished:
        raise ValueError(f"{len(unfinished)} games unfinished (first: {unfinished[:10]}); run again first")
    pending = output.with_name(f".{output.name}.pending-{os.getpid()}")
    counts, reasons, keys, plies = Counter(), Counter(), set(), []
    positions = completed_positions = duplicates = 0
    try:
        with pending.open("x") as stream:
            for game in plan["games"]:
                snapshot = read_source(directory / game["directory"])  # full audit; one game in memory
                configuration, record = snapshot["configuration"], snapshot["game"]
                if (configuration["checkpoint"]["sha256"] != plan["checkpoint"]["sha256"]
                        or configuration["settings"] != plan["settings"] or configuration["seed"] != game["seed"]):
                    raise ValueError(f"game {game['index']} does not match the campaign plan")
                stream.write(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                counts[record["status"]] += 1
                reasons[record["ending_reason"]] += 1
                plies.append(len(record["moves_uci"]))
                positions += len(record["positions"])
                if record["status"] == "completed":
                    completed_positions += len(record["positions"])
                    key = digest([record["initial_fen"], record["moves_uci"]])
                    duplicates += key in keys
                    keys.add(key)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, output)
    finally:
        if pending.exists():
            pending.unlink()
    summary = dict(archive=str(output), archive_sha256=_sha256(output), campaign_sha256=digest(plan),
        checkpoint=plan["checkpoint"], settings=plan["settings"], games=len(plan["games"]),
        status_counts=dict(counts), ending_reasons=dict(reasons), positions=positions,
        positions_in_completed_games=completed_positions, duplicate_completed_games=duplicates,
        plies=dict(mean=statistics.mean(plies), median=statistics.median(plies), max=max(plies)))
    _publish_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _publish_text(path, text):
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    try:
        with pending.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        if pending.exists():
            pending.unlink()
