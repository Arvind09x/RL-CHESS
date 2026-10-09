"""Freeze, then execute only the initial D08 9000/500/500 production quota."""
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

from chess_rl.annotation_benchmark import distribution, read_rows
from chess_rl.data_manifest import DatasetManifest
from chess_rl.initial_labels import (QUOTAS, SEED, SELECTION_VERSION, accounting, key,
                                     replay_production_root, select_references)
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.stockfish_worker import (AnnotationWorker, engine_attempt, utc_now,
                                      verify_artifacts, verify_teacher_bytes)
from chess_rl.supervised import BUNDLE_FILES

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("scripts/produce_d08.py", "src/chess_rl/initial_labels.py",
                "src/chess_rl/stockfish_worker.py", "src/chess_rl/supervised.py",
                "src/chess_rl/annotation_benchmark.py", "src/chess_rl/position_sampling.py")


def save(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def persist(handle, value):
    handle.write(json_line(value))
    handle.flush()
    os.fsync(handle.fileno())


def input_gate(settings):
    manifest = DatasetManifest.load(Path(settings["manifest"]))
    identities = verify_artifacts(manifest, Path(settings["manifest"]), Path(settings["games_path"]),
                                  Path(settings["split_audit_path"]), Path(settings["prepared_path"]))
    teacher = verify_teacher_bytes(manifest, Path(settings["executable"]), ROOT / "runs/d01/stockfish-provenance.json")
    if identities != settings["artifact_identities"] or teacher != settings["teacher"]:
        raise ValueError("Frozen input/teacher identity changed")
    for path, identity in settings["prior_membership_identities"].items():
        if file_identity(Path(path)) != identity:
            raise ValueError("Prior D05/D06 membership changed")
    if {name: file_identity(ROOT / name) for name in SOURCE_FILES} != settings["production_source_identities"]:
        raise ValueError("Production source changed after freeze")
    return manifest


def join_games(refs, path):
    wanted = {r["game_id"] for r in refs}
    games = {}
    for game in read_rows(path):
        if game["game_id"] in wanted:
            if game["game_id"] in games:
                raise ValueError("Duplicate canonical game")
            games[game["game_id"]] = game
    if set(games) != wanted:
        raise ValueError("Missing canonical game")
    return games


def freeze(args):
    started, clock = utc_now(), time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = DatasetManifest.load(args.manifest)
    identities = verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared)
    teacher = verify_teacher_bytes(manifest, args.executable, ROOT / "runs/d01/stockfish-provenance.json")
    prior_paths = [ROOT / "runs/d05" / name / "selected.jsonl" for name in ("tiny-real", "tiny-final")]
    prior_paths.append(args.d06_bundle / "selected.jsonl")
    d06_expected = json.loads((ROOT / "runs/d06/artifact-index.json").read_text())["artifacts"]
    for name in ("selected.jsonl", "labels.jsonl", "settings.json", "summary.json"):
        if file_identity(args.d06_bundle / name) != d06_expected[name]:
            raise ValueError("Accepted D06 evidence mismatch")
    prior = [list(read_rows(p)) for p in prior_paths]
    if prior[0] != prior[1] or len(prior[0]) != 12 or len(prior[2]) != 1000:
        raise ValueError("Accepted prior memberships mismatch")
    excluded = {key(r) for rows in prior for r in rows}
    if len(excluded) != 1012:
        raise ValueError("Expected 1012 distinct prior references")
    refs, eligible = select_references(read_rows(args.prepared / "samples.jsonl"), excluded)
    refs_by_key = {key(r): r for r in refs}
    positions = {r["rule_position"]: r["split"] for r in refs}
    held_games, protected, protected_refs = {}, {}, set()
    for row in read_rows(args.prepared / "heldout-games.jsonl"):
        held_games[row["game_id"]] = row["split"]
    for row in read_rows(args.prepared / "protected-heldout.jsonl"):
        if row["rule_position"] in positions:
            protected[row["rule_position"]] = row["split"]
        if key(row) in refs_by_key:
            if row != refs_by_key[key(row)]:
                raise ValueError("Protected reference mismatch")
            protected_refs.add(key(row))
    for ref in refs:
        expected = ref["split"] if ref["split"] != "train" else None
        if held_games.get(ref["game_id"]) != expected or protected.get(ref["rule_position"]) != expected:
            raise ValueError("Held-out game/rule ownership mismatch")
        if ref["split"] != "train" and key(ref) not in protected_refs:
            raise ValueError("Missing exact protected held-out reference")
    games = join_games(refs, args.games)
    for ref in refs:
        replay_production_root(ref, games[ref["game_id"]], manifest)
    coverage = dict(Counter(r["split"]+":"+r["phase"]+":"+r["rule_position"].split()[1] for r in refs))
    if len(coverage) != 18:
        raise ValueError("Missing split/phase/color coverage")
    with (out / "selected.jsonl").open("x") as handle:
        for ref in refs:
            persist(handle, ref)
    settings = dict(manifest=str(args.manifest.resolve()), games_path=str(args.games.resolve()),
        prepared_path=str(args.prepared.resolve()), split_audit_path=str(args.split_audit.resolve()),
        executable=str(args.executable.resolve()), artifact_identities=identities, teacher=teacher,
        requested_settings=asdict(manifest.teacher), timeout_seconds=30, max_attempts=2,
        selection_version=SELECTION_VERSION, seed=SEED, split_quotas=QUOTAS, selected=10000,
        selection="Per split, smallest SHA256 ranks of compact sorted ASCII JSON {selection_version,seed,split,game_id,sample_ply}; ties by game_id then ply; reference weighted; exclude exact 12 D05 and 1000 D06 references before ranking; annotate train, validation, test in rank order",
        prior_membership_identities={str(p.resolve()): file_identity(p) for p in prior_paths},
        excluded_prior_references=sorted([list(k) for k in excluded]), eligible_by_split=eligible,
        selected_identity=file_identity(out / "selected.jsonl"), coverage=coverage,
        distinct_games=len(games), distinct_rule_positions_by_split={s: len({r["rule_position"] for r in refs if r["split"] == s}) for s in QUOTAS},
        maximum_references_per_game=max(Counter(r["game_id"] for r in refs).values()),
        no_refill=True, failure_policy="Two attempts per fixed reference; failed attempts are separate ledger entries; exhaustion is explicit exclusion; interruption propagates with pending membership; no replacement, refill, reuse of prior labels or resume; shortfall never completes D08",
        production_source_identities={name: file_identity(ROOT / name) for name in SOURCE_FILES},
        freeze_started_at_utc=started, frozen_at_utc=utc_now(), freeze_seconds=time.perf_counter()-clock,
        argv=sys.argv, cwd=str(Path.cwd()), platform=platform.platform(), machine=platform.machine(), python=sys.version,
        scope="D08 initial production only; no training, held-out tuning or expansion decision",
        measurement_policy="Sequential accepted D05 fresh process/hash reset; per-attempt executable rehash; raw UCI and parsed rows retained; annotation wall includes replay/checks/fsync/retries; loader/audit costs measured separately; no tensor cache")
    input_gate(settings)
    save(out / "settings.json", settings)
    print(json.dumps({k: settings[k] for k in ("selected", "split_quotas", "coverage", "selected_identity", "frozen_at_utc", "freeze_seconds")}, indent=2))


async def execute(out):
    settings = json.loads((out / "settings.json").read_text())
    # A single execution only: any prior attempt/label output prevents rerun.
    if set(p.name for p in out.iterdir()) != {"selected.jsonl", "settings.json"}:
        raise ValueError("Output directory is not a pristine frozen run; no resume")
    if (settings["split_quotas"] != QUOTAS or settings["selected"] != 10000
            or settings["max_attempts"] != 2 or settings["timeout_seconds"] != 30
            or settings["selection_version"] != SELECTION_VERSION or settings["seed"] != SEED
            or settings["no_refill"] is not True):
        raise ValueError("Frozen D08 policy mismatch")
    manifest = input_gate(settings)
    if file_identity(out / "selected.jsonl") != settings["selected_identity"]:
        raise ValueError("Frozen membership changed")
    refs = list(read_rows(out / "selected.jsonl"))
    recomputed, eligible = select_references(read_rows(Path(settings["prepared_path"]) / "samples.jsonl"),
                                          {tuple(k) for k in settings["excluded_prior_references"]})
    if refs != recomputed or eligible != settings["eligible_by_split"]:
        raise ValueError("Frozen membership/order differs from deterministic selection")
    games = join_games(refs, Path(settings["games_path"]))
    logging.basicConfig(filename=out / "uci-transport.log", level=logging.DEBUG,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    labels, attempts, exclusions, latencies = [], [], [], []
    status, started, clock = "interrupted", utc_now(), time.perf_counter()
    with (out / "attempts.jsonl").open("x") as af, (out / "labels.jsonl").open("x") as lf, (out / "exclusions.jsonl").open("x") as ef, (out / "latencies.jsonl").open("x") as tf:
        def sink(record):
            persist(af, record)
            attempts.append(record)

        async def attempt(state, trace):
            executable = Path(settings["executable"])
            if file_identity(executable) != settings["teacher"]["executable_identity"]:
                raise ValueError("Teacher executable changed")
            return await engine_attempt(state, manifest, executable, 30, trace)

        identities, teacher = settings["artifact_identities"], settings["teacher"]
        provenance = dict(manifest_sha256=identities["manifest"]["sha256"],
            games_sha256=identities["games.jsonl"]["sha256"], samples_sha256=identities["samples.jsonl"]["sha256"],
            overlap_audit_sha256=identities["overlap-audit.json"]["sha256"],
            executable_sha256=teacher["executable_identity"]["sha256"], embedded_nnue=teacher["embedded_nnue"])
        worker = AnnotationWorker(manifest, provenance, attempt, sink, max_attempts=2,
                                  root_replayer=replay_production_root)
        try:
            for index, ref in enumerate(refs):
                position_clock = time.perf_counter()
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
                    latency = dict(reference=ref, position_seconds=time.perf_counter()-position_clock)
                    persist(tf, latency)
                    latencies.append(latency)
                if (index+1) % 100 == 0:
                    print(json.dumps(dict(processed=index+1, successful=len(labels), attempts=len(attempts), elapsed_seconds=time.perf_counter()-clock)), flush=True)
            annotation_seconds = time.perf_counter()-clock
            audit_clock = time.perf_counter()
            input_gate(settings)
            post_audit_seconds = time.perf_counter()-audit_clock
            status = "complete" if len(labels) == 10000 and not exclusions else "shortfall"
        finally:
            elapsed = time.perf_counter()-clock
            logging.shutdown()
            counts = accounting(refs, labels, attempts, exclusions)
            summary = dict(status=status, selected=len(refs), successful_labels=len(labels), attempts=len(attempts),
                annotation_started_at_utc=started, finished_at_utc=utc_now(),
                annotation_loop_seconds=locals().get("annotation_seconds", elapsed),
                post_input_audit_seconds=locals().get("post_audit_seconds"),
                position_latency_seconds=distribution([r["position_seconds"] for r in latencies]),
                **counts, output_identities={name: file_identity(out / name) for name in BUNDLE_FILES},
                evidence_identities={name: file_identity(out / name) for name in ("latencies.jsonl", "uci-transport.log")})
            save(out / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "complete":
        raise RuntimeError("D08 shortfall; frozen membership was not refilled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    for name in ("manifest", "games", "split-audit", "prepared", "executable", "d06-bundle", "output-dir"):
        frozen.add_argument("--"+name, type=Path, required=True)
    running = commands.add_parser("execute")
    running.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        asyncio.run(execute(args.output_dir))


if __name__ == "__main__":
    main()
