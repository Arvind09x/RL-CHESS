"""D09 label expansion: freeze membership, label with K workers, finalize prefix bundles, compare.

  freeze   --data-config C --d08-bundle D --output OUT --train N --validation N --test N
           [--per-game-cap 4] [--seed S] [--without-d08]
  run      --output OUT --executable E --teacher-evidence EV --workers K
           --engine-policy {fresh-process-v1,persistent-clear-hash-v1} [--timeout 30] [--max-attempts 2]
  finalize --output OUT --bundle B [--train-prefix N] [--allow-mismatch]
  compare  --left BUNDLE_A --right BUNDLE_B

`run` is resumable: each invocation is a new segment; references already
completed by any segment are skipped. `finalize` may run while workers continue
and only consumes completed records. Every finalize re-labels nothing; it merges.
"""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from chess_rl.data_manifest import DatasetManifest
from chess_rl.initial_labels import key, replay_production_root
from chess_rl.label_expansion import (LEDGER_VERSION, SELECTION_VERSION, SPLITS, compact_attempt,
    compare_labels, completing_records, read_rows, select_membership)
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.runtime import data_paths
from chess_rl.stockfish_worker import (AnnotationWorker, PersistentEngine, engine_attempt,
    teacher_provenance, utc_now, verify_artifacts, verify_teacher_bytes)
from chess_rl.supervised import BUNDLE_FILES, open_dataset

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fresh-process-v1", PersistentEngine.policy)


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load(path):
    return json.loads(Path(path).read_text())


def persist(handle, value):
    handle.write(json_line(value))
    handle.flush()
    os.fsync(handle.fileno())


def join_games(refs, games_path):
    wanted = {r["game_id"] for r in refs}
    games = {}
    for game in read_rows(games_path):
        if game["game_id"] in wanted:
            if game["game_id"] in games:
                raise ValueError("duplicate canonical game")
            games[game["game_id"]] = game
    if set(games) != wanted:
        raise ValueError("missing canonical game")
    return games


# ----------------------------------------------------------------------------- freeze

def freeze(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    paths = data_paths(args.data_config)
    manifest = DatasetManifest.load(paths["manifest_path"])
    identities = verify_artifacts(manifest, paths["manifest_path"], paths["games_path"],
                                  paths["split_audit_path"], paths["prepared"])
    prior, prior_identity = [], None
    if not args.without_d08:
        prior = list(read_rows(args.d08_bundle / "selected.jsonl"))
        prior_identity = {name: file_identity(args.d08_bundle / name) for name in ("selected.jsonl", "labels.jsonl")}
    quotas = dict(validation=args.validation, test=args.test, train=args.train)
    clock = time.perf_counter()
    refs, capacity, eligible = select_membership(paths["prepared"] / "samples.jsonl", prior, quotas,
                                                 seed=args.seed, per_game_cap=args.per_game_cap)
    # Held-out ownership: every non-train reference is in the protected registry
    # and its game in the held-out game list; no train reference is.
    wanted = {r["game_id"] for r in refs}
    held_games, protected = {}, {}
    for row in read_rows(paths["prepared"] / "heldout-games.jsonl"):
        if row["game_id"] in wanted:
            held_games[row["game_id"]] = row["split"]
    positions = {r["rule_position"] for r in refs}
    by_key = {key(r): r for r in refs}
    protected_keys = set()
    for row in read_rows(paths["prepared"] / "protected-heldout.jsonl"):
        if row["rule_position"] in positions:
            protected[row["rule_position"]] = row["split"]
        if key(row) in by_key:
            if row != by_key[key(row)]:
                raise ValueError("protected reference mismatch")
            protected_keys.add(key(row))
    for ref in refs:
        expected = ref["split"] if ref["split"] != "train" else None
        if held_games.get(ref["game_id"]) != expected or protected.get(ref["rule_position"]) != expected:
            raise ValueError("held-out game/position ownership mismatch")
        if ref["split"] != "train" and key(ref) not in protected_keys:
            raise ValueError("missing exact protected held-out reference")
    with (out / "selected.jsonl").open("x") as handle:
        for ref in refs:
            persist(handle, ref)
    coverage = dict(Counter(f"{r['split']}:{r['phase']}:{r['rule_position'].split()[1]}" for r in refs))
    plan = dict(schema="d09-plan-v1", selection_version=SELECTION_VERSION, seed=args.seed,
        per_game_cap=args.per_game_cap, quotas=quotas, prior_references=len(prior),
        prior_bundle=str(args.d08_bundle.resolve()) if prior else None, prior_bundle_identity=prior_identity,
        data_config=str(args.data_config.resolve()), paths={k: str(v) for k, v in paths.items()},
        artifact_identities=identities, requested_settings=asdict(manifest.teacher),
        capacity_after_cap=capacity, eligible_after_prior=eligible, selected=len(refs),
        selected_by_split=dict(Counter(r["split"] for r in refs)), coverage=coverage,
        distinct_games=len(wanted), max_references_per_game=max(Counter(r["game_id"] for r in refs).values()),
        order="prior D08 references first, then validation, test and train by rank; any train prefix is the deterministic selection for that size",
        selected_identity=file_identity(out / "selected.jsonl"), frozen_at_utc=utc_now(),
        freeze_seconds=time.perf_counter() - clock, argv=sys.argv, platform=platform.platform())
    save_json(out / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("selected", "selected_by_split", "capacity_after_cap", "distinct_games",
                                            "max_references_per_game", "freeze_seconds")}, indent=2))


# ----------------------------------------------------------------------------- run / worker

def segment_files(out, worker, segment):
    base = out / "segments"
    return {name: base / f"{name}-w{worker:02d}-s{segment:03d}.jsonl" for name in ("attempts", "labels", "exclusions", "done")}


def completed_keys(out):
    done = set()
    base = out / "segments"
    if base.exists():
        for path in sorted(base.glob("done-w*-s*.jsonl")):
            for row in read_rows(path):
                done.add((row["game_id"], row["sample_ply"]))
    return done


def run(args):
    out = args.output
    plan = load(out / "plan.json")
    if file_identity(out / "selected.jsonl") != plan["selected_identity"]:
        raise ValueError("frozen membership changed")
    run_path = out / "run.json"
    if args.data_config is not None:
        # Another machine: re-resolve the inputs and prove they are the frozen bytes.
        paths = data_paths(args.data_config)
        manifest = DatasetManifest.load(paths["manifest_path"])
        identities = verify_artifacts(manifest, paths["manifest_path"], paths["games_path"],
                                      paths["split_audit_path"], paths["prepared"])
        if identities != plan["artifact_identities"]:
            raise ValueError("data config names different input bytes from the frozen plan")
    elif run_path.exists():
        paths = {k: Path(v) for k, v in load(run_path)["paths"].items()}
    else:
        paths = {k: Path(v) for k, v in plan["paths"].items()}
    manifest = DatasetManifest.load(paths["manifest_path"])
    teacher = verify_teacher_bytes(manifest, args.executable, args.teacher_evidence)
    state = load(run_path) if run_path.exists() else dict(schema="d09-run-v1", segments=[], teacher=teacher,
        engine_policy=args.engine_policy, timeout_seconds=args.timeout, max_attempts=args.max_attempts,
        attempt_ledger=LEDGER_VERSION, executable=str(args.executable.resolve()),
        teacher_evidence=str(args.teacher_evidence.resolve()), paths={k: str(v) for k, v in paths.items()})
    state["paths"] = {k: str(v) for k, v in paths.items()}
    for name, value in (("teacher", teacher), ("engine_policy", args.engine_policy),
                        ("timeout_seconds", args.timeout), ("max_attempts", args.max_attempts)):
        if state[name] != value:
            raise ValueError(f"resume with a different {name} is refused; start a new output directory")
    segment = len(state["segments"]) + 1
    (out / "segments").mkdir(exist_ok=True)
    total = plan["selected"]
    done_before = len(completed_keys(out))
    state["segments"].append(dict(segment=segment, workers=args.workers, started_at_utc=utc_now(),
                                  completed_before=done_before, platform=platform.platform(), argv=sys.argv))
    save_json(run_path, state)
    print(json.dumps(dict(segment=segment, workers=args.workers, completed_before=done_before, total=total)), flush=True)
    procs = []
    for worker in range(args.workers):
        log = (out / "segments" / f"worker-w{worker:02d}-s{segment:03d}.log").open("w")
        procs.append((subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(out),
            "--worker", str(worker), "--workers", str(args.workers), "--segment", str(segment)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=dict(os.environ, PYTHONPATH="src:.")), log))
    clock = time.perf_counter()
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(args.poll_seconds)
            done = len(completed_keys(out))
            elapsed = time.perf_counter() - clock
            rate = (done - done_before) / elapsed if elapsed else 0
            remaining = (total - done) / rate if rate else float("inf")
            print(json.dumps(dict(completed=done, total=total, labels_per_second=round(rate, 2),
                                  eta_minutes=round(remaining / 60, 1), elapsed_minutes=round(elapsed / 60, 1))), flush=True)
    finally:
        # Also on interruption: stop workers, record what completed, keep the segment resumable.
        for p, log in procs:
            if p.poll() is None:
                p.terminate()
        for p, log in procs:
            p.wait()
            log.close()
        codes = [p.returncode for p, _ in procs]
        state["segments"][-1].update(finished_at_utc=utc_now(), exit_codes=codes, completed_after=len(completed_keys(out)))
        save_json(run_path, state)
        print(json.dumps(dict(segment=segment, exit_codes=codes, completed=state["segments"][-1]["completed_after"], total=total)), flush=True)
    if any(codes):
        raise SystemExit("one or more workers failed; rerun `run` to resume")


async def worker_main(args):
    out = args.output
    plan, state = load(out / "plan.json"), load(out / "run.json")
    paths = {k: Path(v) for k, v in state["paths"].items()}
    manifest = DatasetManifest.load(paths["manifest_path"])
    teacher = state["teacher"]
    executable, evidence = Path(state["executable"]), Path(state["teacher_evidence"])
    if verify_teacher_bytes(manifest, executable, evidence) != teacher:
        raise ValueError("teacher changed since the run started")
    refs = list(read_rows(out / "selected.jsonl"))
    done = completed_keys(out)
    mine = [r for i, r in enumerate(refs) if i % args.workers == args.worker and key(r) not in done]
    games = join_games(mine, paths["games_path"])
    files = segment_files(out, args.worker, args.segment)
    logging.basicConfig(filename=out / "segments" / f"uci-w{args.worker:02d}-s{args.segment:03d}.log",
                        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    policy = state["engine_policy"]
    pool = (PersistentEngine(manifest, executable, state["timeout_seconds"],
                             expected_sha256=teacher["executable_identity"]["sha256"])
            if policy == PersistentEngine.policy else None)
    provenance = dict(teacher_provenance(plan["artifact_identities"], teacher), process_policy=policy)
    print(json.dumps(dict(worker=args.worker, segment=args.segment, assigned=len(mine))), flush=True)
    clock, count = time.perf_counter(), 0
    with files["attempts"].open("x") as af, files["labels"].open("x") as lf, \
            files["exclusions"].open("x") as ef, files["done"].open("x") as df:
        def sink(record):
            persist(af, compact_attempt(record))

        async def attempt(state_, trace):
            if pool is not None:
                return await pool.attempt(state_, trace)
            if file_identity(executable) != teacher["executable_identity"]:
                raise ValueError("executable changed since verification")
            trace["process_policy"] = "fresh-process-v1"
            return await engine_attempt(state_, manifest, executable, state["timeout_seconds"], trace)

        worker = AnnotationWorker(manifest, provenance, attempt, sink, max_attempts=state["max_attempts"],
                                  root_replayer=replay_production_root)
        try:
            for ref in mine:
                label = await worker.annotate(ref, games[ref["game_id"]])
                if label is not None:
                    persist(lf, label)
                else:
                    persist(ef, dict(reference=ref, reason="attempts_exhausted", attempts=state["max_attempts"]))
                persist(df, dict(game_id=ref["game_id"], sample_ply=ref["sample_ply"]))
                count += 1
                if count % 200 == 0:
                    print(json.dumps(dict(worker=args.worker, done=count, assigned=len(mine),
                                          per_second=round(count / (time.perf_counter() - clock), 2))), flush=True)
        finally:
            if pool is not None:
                await pool.close()
    print(json.dumps(dict(worker=args.worker, finished=count, seconds=round(time.perf_counter() - clock, 1))), flush=True)


# ----------------------------------------------------------------------------- finalize / compare

def load_segments(out):
    base = out / "segments"
    segments = {}
    for path in sorted(base.glob("*-w*-s*.jsonl")):
        name, rest = path.stem.split("-w", 1)
        worker, seg = rest.split("-s")
        segments.setdefault((int(seg), int(worker)), {"attempts": [], "labels": [], "exclusions": [], "done": []})[name] = list(read_rows(path))
    return [segments[k] for k in sorted(segments)]


def finalize(args):
    out, bundle = args.output, args.bundle
    plan, state = load(out / "plan.json"), load(out / "run.json")
    if file_identity(out / "selected.jsonl") != plan["selected_identity"]:
        raise ValueError("frozen membership changed")
    refs = list(read_rows(out / "selected.jsonl"))
    labels, exclusions, attempts, discarded = completing_records(load_segments(out))
    train = [r for r in refs if r["split"] == "train"]
    if args.train_prefix is not None:
        if not 0 < args.train_prefix <= len(train):
            raise ValueError("train prefix outside the frozen train membership")
        train = train[:args.train_prefix]
    members = [r for r in refs if r["split"] != "train"] + train
    pending = [key(r) for r in members if key(r) not in labels and key(r) not in exclusions]
    if pending:
        raise SystemExit(f"{len(pending)} selected references in this bundle are not yet labelled; "
                         f"completed overall: {len(labels) + len(exclusions)} / {len(refs)}")
    bundle.mkdir(parents=True, exist_ok=False)
    with (bundle / "selected.jsonl").open("x") as sf, (bundle / "labels.jsonl").open("x") as lf, \
            (bundle / "attempts.jsonl").open("x") as af, (bundle / "exclusions.jsonl").open("x") as ef:
        for ref in members:
            k = key(ref)
            persist(sf, ref)
            for row in attempts[k]:
                persist(af, row)
            if k in labels:
                persist(lf, labels[k])
            else:
                persist(ef, exclusions[k])
    teacher = state["teacher"]
    settings = dict(manifest=state["paths"]["manifest_path"], games_path=state["paths"]["games_path"],
        prepared_path=state["paths"]["prepared"], split_audit_path=state["paths"]["split_audit_path"],
        artifact_identities=plan["artifact_identities"], teacher=teacher,
        requested_settings=plan["requested_settings"], timeout_seconds=state["timeout_seconds"],
        max_attempts=state["max_attempts"], process_policy=state["engine_policy"], attempt_ledger=LEDGER_VERSION,
        selection_version=plan["selection_version"], seed=plan["seed"], per_game_cap=plan["per_game_cap"],
        frozen_quotas=plan["quotas"], train_prefix=args.train_prefix, prior_bundle=plan["prior_bundle"],
        prior_bundle_identity=plan["prior_bundle_identity"], capacity_after_cap=plan["capacity_after_cap"],
        plan_identity=file_identity(out / "plan.json"), run_segments=state["segments"],
        no_refill=True, scope="D09 expansion bundle; nested prefix of the frozen membership")
    save_json(bundle / "settings.json", settings)
    # The bundle carries its teacher evidence so a data config can point inside it.
    (bundle / "teacher-evidence.json").write_bytes(Path(state["teacher_evidence"]).read_bytes())
    # Equivalence against the accepted D08 labels on the overlapping references.
    equivalence = None
    prior_bundle = args.prior_bundle or (Path(plan["prior_bundle"]) if plan["prior_bundle"] else None)
    if prior_bundle is not None:
        if file_identity(prior_bundle / "labels.jsonl") != plan["prior_bundle_identity"]["labels.jsonl"]:
            raise ValueError("prior bundle labels differ from the bytes frozen in the plan")
        prior_labels = list(read_rows(prior_bundle / "labels.jsonl"))
        member_keys = {key(r) for r in members}
        equivalence = compare_labels([r for r in prior_labels if key(r) in member_keys], labels.values())
        equivalence["prior_bundle"] = str(prior_bundle)
    counts = Counter(r["split"] for r in members)
    summary = dict(status="complete", pending_references=[], selected=len(members),
        successful_labels=sum(1 for r in members if key(r) in labels),
        excluded=sum(1 for r in members if key(r) in exclusions), by_split=dict(counts),
        attempts=sum(len(attempts[key(r)]) for r in members),
        failed_attempts=sum(a["status"] == "failed" for r in members for a in attempts[key(r)]),
        discarded_partial_attempts=discarded,
        train_prefix=args.train_prefix, equivalence_with_prior=equivalence, finalized_at_utc=utc_now(),
        output_identities={name: file_identity(bundle / name) for name in BUNDLE_FILES})
    save_json(bundle / "summary.json", summary)
    if equivalence and equivalence["mismatched"] and not args.allow_mismatch:
        raise SystemExit(f"label equivalence failed on {equivalence['mismatched']} of {equivalence['compared']} "
                         f"overlapping references: {equivalence['mismatched_fields']}; bundle written for inspection")
    # Acceptance: the production loader must open every split.
    paths = {k: Path(v) for k, v in state["paths"].items()}
    loaded = {}
    for split in SPLITS:
        data = open_dataset(bundle=bundle, manifest_path=paths["manifest_path"], games_path=paths["games_path"],
                            split_audit_path=paths["split_audit_path"], prepared=paths["prepared"],
                            teacher_evidence=bundle / "teacher-evidence.json", split=split)
        loaded[split] = len(data)
    summary["loader_acceptance"] = loaded
    save_json(bundle / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("selected", "by_split", "successful_labels", "excluded",
                                               "discarded_partial_attempts", "loader_acceptance")}, indent=2))
    if equivalence:
        print(json.dumps({k: equivalence[k] for k in ("compared", "matched", "mismatched", "mismatched_fields")}))


def compare(args):
    result = compare_labels(read_rows(args.left / "labels.jsonl"), read_rows(args.right / "labels.jsonl"))
    result.update(left=str(args.left), right=str(args.right), compared_at_utc=utc_now())
    if args.output:
        save_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("compared", "matched", "mismatched", "mismatched_fields", "left_only", "right_only")}, indent=2))
    if result["mismatched"]:
        raise SystemExit("labels differ under the frozen equivalence criterion")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--data-config", type=Path, required=True)
    f.add_argument("--d08-bundle", type=Path, default=None)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--train", type=int, required=True)
    f.add_argument("--validation", type=int, required=True)
    f.add_argument("--test", type=int, required=True)
    f.add_argument("--per-game-cap", type=int, default=4)
    f.add_argument("--seed", type=int, default=20261013)
    f.add_argument("--without-d08", action="store_true", help="equivalence experiments only; no prior references")
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--executable", type=Path, required=True)
    r.add_argument("--teacher-evidence", type=Path, required=True)
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--engine-policy", choices=POLICIES, required=True)
    r.add_argument("--timeout", type=float, default=30)
    r.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=2)
    r.add_argument("--poll-seconds", type=float, default=30)
    r.add_argument("--data-config", type=Path, default=None,
                   help="re-resolve the frozen inputs on this machine; bytes must match the plan")
    w = sub.add_parser("worker")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--worker", type=int, required=True)
    w.add_argument("--workers", type=int, required=True)
    w.add_argument("--segment", type=int, required=True)
    z = sub.add_parser("finalize")
    z.add_argument("--output", type=Path, required=True)
    z.add_argument("--bundle", type=Path, required=True)
    z.add_argument("--train-prefix", type=int, default=None)
    z.add_argument("--allow-mismatch", action="store_true")
    z.add_argument("--prior-bundle", type=Path, default=None, help="location of the frozen prior bundle on this machine")
    c = sub.add_parser("compare")
    c.add_argument("--left", type=Path, required=True)
    c.add_argument("--right", type=Path, required=True)
    c.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.without_d08 and args.d08_bundle is None:
            parser.error("--d08-bundle is required unless --without-d08")
        freeze(args)
    elif args.command == "run":
        if args.workers < 1:
            parser.error("workers must be positive")
        run(args)
    elif args.command == "worker":
        asyncio.run(worker_main(args))
    elif args.command == "finalize":
        finalize(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
