"""Extending a recorded self-play round by halves, and the one pointer training reads.

choose_prefix() records, once, which extension halves join a recorded two-half
core round. Half k counts only if every one of its games finished by the
deadline and half k-1 counted, so the counted halves are always a prefix. The
choice stays open while the first uncounted half could still finish before the
deadline. Halves that were never started (e.g. a failed disk check) are listed
as skipped and end the prefix.

write_training_pointer() records, once, the replay buffer mixed training reads:
the extended set if its choice exists and every included half archived and
loaded exactly, otherwise the core choice. The core buffer must itself verify;
if it does not, nothing is written. A recorded pointer is final, so a fallback
closes the extension for good. Both records are published whole and never
overwritten (selfplay_campaign._publish_text).

Disk projection measures a finished core half on disk (allocated blocks, not
apparent size) and scales it to the halves still to be played and the buffer
still to be built, net of anything already on disk.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from chess_rl.evaluation.matches import digest
from chess_rl.selfplay_campaign import _now, _publish_text, _sha256, finished_by, load

EXTENSION_SCHEMA = "selfplay-extension-choice-v1"
POINTER_SCHEMA = "selfplay-training-buffer-v1"


class UncleanBuffer(ValueError):
    """A buffer or archive that does not hold exactly the intended games; triggers the fallback."""


class DiskUnmeasured(RuntimeError):
    """Disk use could not be measured completely; the disk check then counts as not fitting."""


# Content failures of the extended set that fall back to the core. OSError is not among them:
# a transient I/O failure stops the job (resubmit) instead of closing the extension for good.
EXTENDED_FAILURES = (UncleanBuffer, sqlite3.DatabaseError, json.JSONDecodeError, KeyError, TypeError, ValueError)


def _record(path):
    return json.loads(Path(path).read_text())


def _publish_json(path, value):
    _publish_text(Path(path), json.dumps(value, indent=2, sort_keys=True) + "\n")


def choose_prefix(output, *, core, halves, buffers, deadline, skipped=(), now=None):
    """Record once which extension halves (a prefix of `halves`) join the core round.

    buffers[k] is the buffer for the core plus halves[:k + 1]. skipped is a list of
    (directory, reason) for halves never started; they follow `halves` and never count.
    An existing record is returned unchanged.
    """
    output = Path(output)
    if output.exists():
        return _record(output)
    halves, buffers = [Path(h) for h in halves], [Path(b) for b in buffers]
    if len(halves) != len(buffers):
        raise ValueError("give one buffer per extension half")
    core = Path(core)
    core_record = _record(core)
    if len(core_record["halves"]) != 2:
        raise ValueError(f"core choice {core} has {len(core_record['halves'])} halves; the extension needs both")
    counted = 0
    for half in halves:
        if not finished_by(half, deadline):
            break
        counted += 1
    if counted < len(halves) and (time.time() if now is None else now) < deadline:
        raise ValueError(f"extension half {halves[counted]} can still finish before the deadline; choice stays open")
    included = [str(h) for h in halves[:counted]]
    every = core_record["halves"] + included
    choice = dict(schema=EXTENSION_SCHEMA, decided_utc=_now(),
                  deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                  rule="extension half k counts only if all its games finished by the deadline and half k-1 counted",
                  core=dict(path=str(core), sha256=_sha256(core), halves=core_record["halves"],
                            buffer=core_record["buffer"]),
                  considered=[str(h) for h in halves], skipped=[dict(half=str(h), reason=r) for h, r in skipped],
                  counted=included, halves=every, games=sum(len(load(h)["games"]) for h in every),
                  buffer=str(buffers[counted - 1]) if counted else None)
    _publish_json(output, choice)
    return choice


def _unclean(condition, message):
    if not condition:
        raise UncleanBuffer(message)


def verify_buffer(buffer, halves):
    """Check that buffer/replay.sqlite holds exactly the finished games of `halves`, once each.

    Per half: archive summary matches its archive and plan (archive() already checked every
    line against the plan's per-game seed and checkpoint); the half's ingest report exists, was
    made from that archive and has one receipt per archive line, none interrupted. Buffer: its
    source ids are exactly the receipts' source ids, once each, and a full ReplayBuffer audit
    (which re-derives every source id from its payload) passes. Returns the measured counts.
    """
    from chess_rl.replay import ReplayBuffer
    buffer = Path(buffer)
    expected, checkpoints, per_half = set(), set(), []
    for half in map(Path, halves):
        plan = load(half)
        summary_path, archive = half / "games.summary.json", half / "games.jsonl"
        _unclean(summary_path.exists() and archive.exists(), f"{half}: archive or its summary missing")
        summary = _record(summary_path)
        _unclean(summary["archive_sha256"] == _sha256(archive), f"{half}: archive does not match its summary")
        _unclean(summary["campaign_sha256"] == digest(plan), f"{half}: archive summary is for another plan")
        _unclean(summary["games"] == len(plan["games"]), f"{half}: archive does not hold every planned game")
        report_path = buffer / f"ingest-{half.name}.json"
        _unclean(report_path.exists(), f"{half}: no ingest report in {buffer}")
        try:
            report = _record(report_path)
        except json.JSONDecodeError:
            raise UncleanBuffer(f"{report_path}: unreadable") from None
        _unclean(report["records_sha256"] == summary["archive_sha256"], f"{report_path}: made from another archive")
        receipts = report["receipts"]
        _unclean(len(receipts) == len(plan["games"]), f"{report_path}: {len(receipts)} receipts for "
                 f"{len(plan['games'])} games")
        _unclean(not any(r["disposition"] == "excluded-interrupted" for r in receipts),
                 f"{report_path}: an interrupted game was ingested")
        ids = {r["source_id"] for r in receipts}
        _unclean(len(ids) == len(receipts) and not ids & expected, f"{report_path}: repeated source ids")
        expected |= ids
        checkpoints.add(plan["checkpoint"]["sha256"])
        per_half.append(dict(half=str(half), games=len(plan["games"]), report=str(report_path)))
    _unclean(len(checkpoints) == 1, f"halves were played by different checkpoints: {sorted(checkpoints)}")
    database = buffer / "replay.sqlite"
    _unclean(database.is_file(), f"{database} missing")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        stored = [source for (source,) in connection.execute("SELECT source_id FROM sources")]
    finally:
        connection.close()
    _unclean(len(stored) == len(set(stored)) == len(expected) and set(stored) == expected,
             f"{database}: sources are not exactly the archived games (stored {len(stored)}, expected "
             f"{len(expected)}, missing {len(expected - set(stored))}, extra {len(set(stored) - expected)})")
    started = time.monotonic()
    try:
        summary = ReplayBuffer(database).summary()  # full integrity and target re-audit
    except ValueError as error:
        raise UncleanBuffer(f"{database}: {error}") from error
    counts = summary["source_counts"]
    _unclean(counts["excluded-interrupted"] == 0 and sum(counts.values()) == len(expected),
             f"{database}: audited source counts {counts} != {len(expected)} games")
    return dict(buffer=str(buffer), games=len(expected), checkpoint_sha256=checkpoints.pop(), halves=per_half,
                source_counts=counts, training_positions=summary["training_positions"],
                sqlite_bytes=database.stat().st_size, audit_seconds=round(time.monotonic() - started, 1))


def write_training_pointer(output, *, core, extension=None, note=None):
    """Record once which buffer training reads, with the branch taken and every check made."""
    output = Path(output)
    if output.exists():
        return _record(output)
    core = Path(core)
    core_record = _record(core)
    extension = Path(extension) if extension is not None and Path(extension).exists() else None
    ext_record = _record(extension) if extension is not None else None
    checks, chosen = {}, None
    if len(core_record["halves"]) != 2:
        branch, reason = "core-a-only", "core choice is the first half alone; the extension needs both"
    elif ext_record is None:
        branch, reason = "core-extension-not-run", note or "no extension choice record"
    else:
        if ext_record["core"]["sha256"] != _sha256(core):
            raise ValueError(f"{extension} was made against a different core choice than {core}")
        if not ext_record["counted"]:
            branch, reason = "core-extension-not-counted", "no extension half finished by the deadline"
        else:
            try:
                checks["extended"] = verify_buffer(ext_record["buffer"], ext_record["halves"])
                branch, reason, chosen = "extended", None, ext_record
            except EXTENDED_FAILURES as error:
                branch, reason = "core-extension-unclean", f"{type(error).__name__}: {error}"
    if chosen is None:
        checks["core"] = verify_buffer(core_record["buffer"], core_record["halves"])  # raises: no pointer
        chosen = core_record
    measured = checks["extended" if branch == "extended" else "core"]
    pointer = dict(schema=POINTER_SCHEMA, decided_utc=_now(), branch=branch, reason=reason, note=note,
                   rule="extended set if its choice exists and every included half archived and loaded "
                        "exactly, else the core choice; final once written",
                   buffer=chosen["buffer"], replay=str(Path(chosen["buffer"]) / "replay.sqlite"),
                   halves=chosen["halves"], games=measured["games"],
                   training_positions_before_holdout=measured["training_positions"],
                   core_choice=dict(path=str(core), sha256=_sha256(core)),
                   extension_choice=None if extension is None else dict(path=str(extension),
                                                                        sha256=_sha256(extension)),
                   checks=checks)
    _publish_json(output, pointer)
    return pointer


# Disk projection.

def allocated_bytes(*paths):
    """Allocated bytes (what a block quota charges) of existing paths; missing paths count 0."""
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return 0
    result = subprocess.run(["du", "-s", "-B1", "-c", *paths], capture_output=True, text=True)
    total = [line for line in result.stdout.splitlines() if line.endswith("\ttotal")]
    # du prints a partial total when it cannot read part of a tree; that total undercounts.
    if result.returncode != 0 or not total:
        raise DiskUnmeasured(f"du exited {result.returncode}: {result.stderr.strip()[:500]}")
    return int(total[-1].split("\t")[0])


def _half_store(half):
    half = Path(half)
    return allocated_bytes(half / "games", half / "games.jsonl", half / "events.jsonl")


def project_need(*, core_halves, core_buffer, play, buffer, buffer_halves, margin=1.3, headroom=1 << 30):
    """Bytes still needed to play `play` and build `buffer` (buffer_halves halves), net of what exists.

    A half's journals and archive are projected from the mean core half; a buffer from the core
    buffer's bytes per half. Game-length variance is what the margin is for.
    """
    store = [_half_store(h) for h in core_halves]
    per_half_store = sum(store) / len(store)
    per_half_buffer = allocated_bytes(Path(core_buffer) / "replay.sqlite") / len(core_halves)
    remaining_play = {str(h): max(0.0, per_half_store - _half_store(h)) for h in play}
    remaining_buffer = max(0.0, buffer_halves * per_half_buffer - allocated_bytes(Path(buffer) / "replay.sqlite"))
    raw = sum(remaining_play.values()) + remaining_buffer
    return dict(per_half_store_bytes=per_half_store, per_half_buffer_bytes=per_half_buffer,
                remaining_play_bytes=remaining_play, remaining_buffer_bytes=remaining_buffer,
                raw_need_bytes=raw, margin=margin, headroom_bytes=headroom, need_bytes=raw * margin + headroom)


def parse_quota(text):
    """(used, limit) bytes from `quota -w` output with one filesystem row, else None."""
    rows = []
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if "blocks" in line.split():
            for row in lines[number + 1:]:
                fields = row.split()
                if len(fields) >= 4:
                    try:
                        rows.append([int(f.rstrip("*")) for f in fields[1:4]])
                    except ValueError:
                        return None
            break
    if len(rows) != 1:
        return None
    used, soft, hard = rows[0]
    limit = soft or hard
    return (used * 1024, limit * 1024) if limit else None


def free_quota(limit_bytes, home=None):
    """Free bytes under the user's quota: `quota -w` if it parses, else limit - `du` of home."""
    home = home or os.path.expanduser("~")
    try:
        result = subprocess.run(["quota", "-w"], capture_output=True, text=True, timeout=60)
        parsed = parse_quota(result.stdout)
        raw = result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        parsed, raw = None, f"{type(error).__name__}: {error}"
    if parsed is not None:
        used, limit = parsed
        return dict(method="quota -w", raw=raw, used_bytes=used, limit_bytes=limit, free_bytes=limit - used)
    used = allocated_bytes(home)
    return dict(method=f"du -s -B1 {home} against {limit_bytes} bytes", quota_output=raw, used_bytes=used,
                limit_bytes=limit_bytes, free_bytes=limit_bytes - used)
