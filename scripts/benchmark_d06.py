"""Run only D06: fixed 1,000-position training-only annotation benchmark."""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

from chess_rl.annotation_benchmark import (SEED, SELECTION_VERSION, TransportClock,
    distribution, key, read_rows, select_references)
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.stockfish_worker import (AnnotationWorker, engine_attempt,
    replay_root, utc_now, verify_artifacts, verify_teacher_bytes)

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def persist(handle, value):
    handle.write(json_line(value))
    handle.flush()
    os.fsync(handle.fileno())


async def run(args):
    invocation_started, full_clock = utc_now(), time.perf_counter()
    # Never reuse a partial or completed benchmark directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    manifest = DatasetManifest.load(args.manifest)
    identities = verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared)
    teacher = verify_teacher_bytes(manifest, args.executable, ROOT / "runs/d01/stockfish-provenance.json")
    d05_paths = [ROOT / "runs/d05" / name / "selected.jsonl" for name in ("tiny-real", "tiny-final")]
    d05_sets = [{key(r) for r in read_rows(p)} for p in d05_paths]
    if d05_sets[0] != d05_sets[1] or len(d05_sets[0]) != 12:
        raise ValueError("Accepted D05 membership mismatch")
    excluded = d05_sets[0]
    refs, candidates = select_references(read_rows(args.prepared / "samples.jsonl"), excluded)
    wanted = {r["game_id"] for r in refs}
    positions = {r["rule_position"] for r in refs}
    if any(r["rule_position"] in positions for r in read_rows(args.prepared / "protected-heldout.jsonl")):
        raise ValueError("Selected position is protected")
    if any(r["game_id"] in wanted for r in read_rows(args.prepared / "heldout-games.jsonl")):
        raise ValueError("Selected game is held out")
    games = {}
    for game in read_rows(args.games):
        if game["game_id"] in wanted:
            if game["game_id"] in games:
                raise ValueError("Duplicate canonical game")
            games[game["game_id"]] = game
    if set(games) != wanted:
        raise ValueError("Missing canonical games")
    for ref in refs:
        replay_root(ref, games[ref["game_id"]], manifest)
    if verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared) != identities:
        raise ValueError("Inputs changed during selection")
    coverage = dict(Counter(r["phase"]+":"+r["rule_position"].split()[1] for r in refs))
    if len(coverage) != 6:
        raise ValueError("Selected benchmark lacks phase/color coverage")
    with (out / "selected.jsonl").open("x") as handle:
        for ref in refs:
            persist(handle, ref)
    settings = dict(manifest=str(args.manifest.resolve()), games_path=str(args.games.resolve()),
        prepared_path=str(args.prepared.resolve()), split_audit_path=str(args.split_audit.resolve()),
        artifact_identities=identities, teacher=teacher, requested_settings=asdict(manifest.teacher),
        timeout_seconds=args.timeout, max_attempts=2,
        failure_policy="At most two attempts per fixed reference; exhaustion is exclusion, interruption propagates; no replacement or resume; attempts are never labels",
        selection_version=SELECTION_VERSION, seed=SEED,
        selection="Lowest 1000 SHA256 ranks of distinct audited train rule positions using compact sorted ASCII JSON {selection_version,seed,rule_position}; smallest (game_id,sample_ply) representative after removing 12 D05 references; annotate in rank order",
        d05_membership_identities={str(p.relative_to(ROOT)): file_identity(p) for p in d05_paths},
        excluded_d05_references=sorted([list(k) for k in excluded]),
        selected=1000, distinct_rule_positions=len(positions), distinct_games=len(games),
        coverage=coverage, eligible_reference_coverage=candidates,
        selected_identity=file_identity(out / "selected.jsonl"), frozen_at_utc=utc_now(),
        setup_seconds_before_settings=time.perf_counter()-full_clock,
        argv=sys.argv, cwd=str(Path.cwd()), platform=platform.platform(), machine=platform.machine(),
        python=sys.version, scope="D06 benchmark only; never D08 production or a final expansion allocation",
        measurement_policy="Sequential fresh D05 engine process per attempt; perf_counter wall timing; raw transport observation; returned engine time separately; fsync writes included; shared-adapter replay/encoding measured by independent post-run audit; setup/selection and post-input hash audit excluded from annotation loop")
    save(out / "settings.json", settings)
    logging.basicConfig(filename=out / "uci-transport.log", level=logging.DEBUG,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    labels, attempts, exclusions, latencies = [], [], [], []
    status, loop_started_at = "interrupted", utc_now()
    loop_clock = time.perf_counter()
    loop_elapsed, post_audit_seconds = None, None
    with (out / "attempts.jsonl").open("x") as af, (out / "labels.jsonl").open("x") as lf, (out / "exclusions.jsonl").open("x") as ef, (out / "latencies.jsonl").open("x") as tf:
        def sink(record):
            persist(af, record)
            attempts.append(record)

        async def attempt(state, trace):
            clock = TransportClock()
            logger = logging.getLogger("chess.engine")
            logger.addHandler(clock)
            try:
                verify_clock = time.perf_counter()
                if file_identity(args.executable) != teacher["executable_identity"]:
                    raise ValueError("Executable changed since verification")
                trace["executable_rehash_seconds"] = time.perf_counter()-verify_clock
                return await engine_attempt(state, manifest, args.executable, args.timeout, trace)
            finally:
                trace["benchmark_timing"] = clock.metrics(time.perf_counter()-clock.started)
                logger.removeHandler(clock)

        worker = AnnotationWorker(manifest, dict(manifest_sha256=identities["manifest"]["sha256"],
            games_sha256=identities["games.jsonl"]["sha256"], samples_sha256=identities["samples.jsonl"]["sha256"],
            overlap_audit_sha256=identities["overlap-audit.json"]["sha256"],
            executable_sha256=teacher["executable_identity"]["sha256"], embedded_nnue=teacher["embedded_nnue"]),
            attempt, sink, max_attempts=2)
        try:
            for index, ref in enumerate(refs):
                started = time.perf_counter()
                try:
                    label = await worker.annotate(ref, games[ref["game_id"]])
                    if label is not None:
                        persist(lf, label)
                        labels.append(label)
                    else:
                        exclusion = dict(reference=ref, reason="attempts_exhausted", attempts=2)
                        persist(ef, exclusion)
                        exclusions.append(exclusion)
                finally:
                    latency = dict(reference=ref, position_seconds=time.perf_counter()-started)
                    persist(tf, latency)
                    latencies.append(latency)
                if (index+1) % 100 == 0:
                    print(json.dumps(dict(processed=index+1, successful=len(labels), attempts=len(attempts), elapsed_seconds=time.perf_counter()-loop_clock)), flush=True)
            loop_elapsed = time.perf_counter()-loop_clock
            audit_clock = time.perf_counter()
            if verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared) != identities:
                raise ValueError("Inputs changed during annotation")
            if verify_teacher_bytes(manifest, args.executable, ROOT / "runs/d01/stockfish-provenance.json") != teacher:
                raise ValueError("Teacher changed during annotation")
            post_audit_seconds = time.perf_counter()-audit_clock
            status = "complete"
        finally:
            if loop_elapsed is None:
                loop_elapsed = time.perf_counter()-loop_clock
            finished = {key(r) for r in labels} | {key(r["reference"]) for r in exclusions}
            logging.shutdown()
            summary = dict(status=status, invocation_started_at_utc=invocation_started,
                annotation_started_at_utc=loop_started_at, finished_at_utc=utc_now(),
                selected=len(refs), successful_labels=len(labels), attempts=len(attempts),
                failed_attempts=sum(r["status"] == "failed" for r in attempts),
                interrupted_attempts=sum(r["status"] == "interrupted" for r in attempts),
                retries=sum(r["attempt"] > 1 for r in attempts), excluded=len(exclusions),
                pending_references=[r for r in refs if key(r) not in finished],
                annotation_loop_seconds=loop_elapsed, setup_seconds=settings["setup_seconds_before_settings"],
                post_input_audit_seconds=post_audit_seconds, total_seconds=time.perf_counter()-full_clock,
                successful_labels_per_second=len(labels)/loop_elapsed,
                position_latency_seconds=distribution([r["position_seconds"] for r in latencies]),
                output_identities={p.name: file_identity(p) for p in out.iterdir() if p.is_file()})
            save(out / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "games", "split-audit", "prepared", "executable", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("timeout must be in (0,60]")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
