"""Round-2 training from the round-1 selection (R1-mix): build the round-2 arrays, train R2-mix or R2-pure.

disk          upper bound of the round-2 arrays plus a reserve against the measured free quota
              (exit 0 fits, 10 short); nothing is read from the buffer.
data          build runs/train/r2-data once from the round-2 buffer record (round2-buffer-v1): the record's
              start checkpoint must be R1-mix and the final round-1 selection, a fresh verify_buffer must
              equal the record's checks, the full replay audit must pass, and every sample must have been
              played by R1-mix. If the arrays exist they are reopened and verified against the record's and
              the selection's sha256 and R1-mix, and the buffer is not opened (it may have been deleted).
train         run or resume one recipe on the union of the round data sets (round 1 then round 2) from
              R1-mix's weights with a fresh optimizer. P is measured fresh first and must reproduce the
              expected validation loss; R1-mix is measured fresh and its total validation loss must be
              within --start-tolerance of its own sidecar (the exact differences are recorded). Every
              forgetting delta is against P; paired intervals against R1-mix are reported alongside.

compare, inference and timings work on round-2 run directories unchanged: use scripts/train_mixed.py.
Updates use chess_rl.training.round_training; the union and the round-2 rules are in
chess_rl.training.round2_training.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from chess_rl.dataset_cache import CachedSupervisedDataset, current_source_identity
from chess_rl.evaluation.matches import digest
from chess_rl.replay import ReplayBuffer
from chess_rl.runtime import configure, data_paths
from chess_rl.selfplay_rounds import free_quota, verify_buffer
from chess_rl.training import round2_training as U
from chess_rl.training import round_training as R
from chess_rl.training.checkpoints import (Progress, cuda_runtime_config, load_inference, runtime_config,
                                           save_checkpoint)
from chess_rl.training.mixed_state import MixedState
from chess_rl.training.optimizers import describe
from scripts.train_mixed import (GIB, Evaluator, Mismatch, check, check_verified, checkpoint_updates, disk_guard,
                                 reference_record, sha256)

RECORD_SCHEMA = "round2-buffer-v1"
LEGAL_MOVES_BOUND = 218  # most legal moves in any chess position


# disk and data

def read_selection(path, start_sha):
    selection = json.loads(Path(path).read_text())
    check(selection.get("final") is True and selection.get("status") == "final", f"{path}: selection is not final")
    check(selection["selected"]["sha256"] == start_sha, f"{path}: the selected network is not the start checkpoint")
    return selection


def check_record(record, *, start_sha, games):
    """Static checks of the round-2 buffer record against what round-2 training expects."""
    check(record.get("schema") == RECORD_SCHEMA, f"not a {RECORD_SCHEMA} record")
    check(record["start_checkpoint"]["sha256"] == start_sha, "the record's games were played from another checkpoint")
    recorded = record["checks"]
    check(record["buffer"] == recorded["buffer"] and record["replay"] == str(Path(record["buffer"]) / "replay.sqlite"),
          "record buffer differs from its recorded check")
    check([h["half"] for h in recorded["halves"]] == record["halves"], "record halves differ from its recorded check")
    check(record["games"] == recorded["games"] == games, f"record games {record['games']}, expected {games}")
    check(record["training_positions_before_holdout"] == recorded["training_positions"],
          "record positions differ from its recorded check")
    check(recorded["checkpoint_sha256"] == start_sha, "the recorded check names another checkpoint")
    return recorded


def array_bound(positions):
    return R.planned_bytes({"all": positions}, {"all": positions * LEGAL_MOVES_BOUND})


def command_disk(args):
    record = json.loads(Path(args.record).read_text())
    need = array_bound(record["training_positions_before_holdout"]) + args.reserve_bytes
    free = free_quota(args.quota_bytes)
    fits = free["free_bytes"] >= need
    print(json.dumps(dict(event="disk", method=free["method"], free_gib=round(free["free_bytes"] / GIB, 3),
                          need_gib=round(need / GIB, 3), arrays_bound_gib=round((need - args.reserve_bytes) / GIB, 3),
                          reserve_gib=round(args.reserve_bytes / GIB, 3), fits=fits)), flush=True)
    return 0 if fits else 10


def command_data(args):
    output = Path(args.output)
    paths = data_paths(args.data_config)
    identity = current_source_identity(split="train", **paths)
    start_sha = sha256(args.start_checkpoint)
    check(start_sha == args.start_sha256, f"{args.start_checkpoint} sha256 {start_sha} != {args.start_sha256}")
    record_sha, selection_sha = sha256(args.record), sha256(args.selection)
    registries = {n: identity["authoritative"][n] for n in R.REGISTRIES}
    if output.exists():
        # Reopen only: the buffer may be gone, and the arrays are verified by their own manifest.
        started = time.perf_counter()
        data = R.RoundData.open(output, registries=registries)
        source = data.manifest["source"]
        check(source.get("schema") == RECORD_SCHEMA and source["record"]["sha256"] == record_sha,
              f"{output} was built from another buffer record")
        check(source["selection"]["sha256"] == selection_sha, f"{output} was built against another selection")
        check(data.manifest["checkpoint_sha256"] == start_sha, f"{output} holds games played by another checkpoint")
        check(data.manifest["supervised_source_id"] == digest(identity), f"{output} was built for another Stockfish source")
        check(data.manifest["holdout"]["rule"] == R.HOLDOUT_RULE, f"{output} uses another hold-out rule")
        print(json.dumps(dict(event="reused", output=str(output), replay_id=data.replay_id,
                              open_seconds=time.perf_counter() - started, counts=data.manifest["counts"])), flush=True)
        return 0
    read_selection(args.selection, start_sha)
    record = json.loads(Path(args.record).read_text())
    recorded = check_record(record, start_sha=start_sha, games=args.expected_games)
    require_free = disk_guard(args)
    require_free(array_bound(recorded["training_positions"]) + args.reserve_bytes,
                 "round-2 arrays (upper bound) plus reserve")
    timings = {}
    started = time.perf_counter()
    verified = verify_buffer(record["buffer"], record["halves"])
    timings["verify_buffer_seconds"] = time.perf_counter() - started
    check_verified(recorded, verified, start_sha)
    started = time.perf_counter()
    buffer = ReplayBuffer(record["replay"])  # full integrity and target re-audit
    timings["replay_open_seconds"] = time.perf_counter() - started
    summary = buffer.summary()
    check(summary["source_counts"] == verified["source_counts"]
          and summary["training_positions"] == verified["training_positions"], "buffer changed after verification")
    source = dict(schema=RECORD_SCHEMA, record=dict(path=str(args.record), sha256=record_sha),
                  selection=dict(path=str(args.selection), sha256=selection_sha),
                  start_checkpoint=dict(path=str(args.start_checkpoint), sha256=start_sha),
                  buffer=record["buffer"], replay=record["replay"], verified=verified)
    data = R.build_round_data(output, buffer=buffer, prepared=paths["prepared"], identity=identity,
                              checkpoint_sha256=start_sha, source=source, require_free=require_free,
                              extra_bytes=args.reserve_bytes, timings=timings)
    print(json.dumps(dict(event="built", output=str(output), replay_id=data.replay_id, counts=data.manifest["counts"],
                          timings=data.manifest["timings"])), flush=True)
    return 0


# train

class RoundEvaluator(Evaluator):
    """Round 1's evaluator; the hold-out summary adds one entry per round (sliced, no extra compute)."""

    def __init__(self, validation, holdout, batch_size):
        super().__init__(validation, holdout, batch_size)
        self.groups_by_round = [list(holdout.game_ids[s]) for s in holdout.part_slices()]

    def holdout_summary(self, hold):
        summary = super().holdout_summary(hold)
        summary["by_round"] = [dict(round=k + 1, **{name: R.clustered_bootstrap(part[name], groups)
                                                    for name in ("policy", "wdl", "total")})
                               for k, (part, groups) in enumerate(zip(U.holdout_by_round(hold, self.holdout),
                                                                      self.groups_by_round))]
        return summary


def run_provenance(recipe, *, start, start_sha, p, p_sha, union, supervised, validation, total, every, batch_size,
                   hard_stop=True, amendment=None):
    source_id, membership = R.supervised_identity(supervised)
    record = dict(schema=U.RUN_VERSION,
        purpose="round-2 training from the round-1 selection on the round-1 and round-2 self-play arrays",
        recipe=recipe, recipe_details=U.recipe_details(recipe),
        initial_checkpoint=dict(sha256=start_sha, checkpoint_id=start.checkpoint_id,
                                role="start weights: the final round-1 selection (R1-mix)"),
        forgetting_reference=dict(sha256=p_sha, checkpoint_id=p.checkpoint_id,
                                  role="P: every forgetting delta, flag and stop is against P, measured fresh"),
        round_data=dict(union=union.manifest, replay_id=union.replay_id, training_positions=len(union.train),
                        holdout_positions=len(union.holdout)),
        stockfish=dict(source_id=source_id, membership_sha256=membership, positions=len(supervised)),
        validation=dict(source_id=digest(validation.source_identity), positions=len(validation)),
        sampler=dict(settings=asdict(R.SETTINGS), seed=U.SAMPLING_SEED, note=U.SAMPLER_NOTE),
        optimizer=dict(describe(R.OPTIMIZER_SETTINGS), schedule="constant learning rate, no warmup",
                       fresh="new AdamW at update 0 (R1-mix's optimizer state is not carried over)"),
        loss_weights=dict(policy=R.SETTINGS.policy_weight, wdl=R.SETTINGS.wdl_weight),
        run_seed=U.RUN_SEED, total_updates=total, checkpoint_every=every, passes_over_selfplay=1,
        thresholds=dict(flag=R.FLAG_DELTA, stop=R.STOP_DELTA,
                        delta="checkpoint validation total loss minus P's, P measured fresh in the same job",
                        fallback=U.FALLBACK_RULE),
        evaluation=dict(batch_size=batch_size, bootstrap=dict(resamples=R.BOOTSTRAP_RESAMPLES, seed=R.BOOTSTRAP_SEED),
                        versus_start="paired intervals against the start checkpoint, report-only"))
    if not hard_stop:
        record["thresholds"]["hard_stop_enforced"] = False
        record["exploratory"] = dict(amendment=amendment, rule=(
            "every breach of the stop threshold is recorded but training continues; the candidate is the "
            "final checkpoint; the original rule's candidate is reported alongside; report-only, never selected"))
    return record


def measure_start(evaluator, start, sidecar, tolerance, reference):
    """R1-mix measured fresh against its own sidecar: total within tolerance, exact differences recorded."""
    official, rows, hold, _ = evaluator.measure(start.model)
    recorded = sidecar["evaluation"]["validation"]
    differences = {k: official[k] - recorded[k] for k in ("policy_loss", "wdl_loss", "total_loss", "policy_accuracy")}
    check(abs(differences["total_loss"]) <= tolerance,
          f"start checkpoint validation total loss {official['total_loss']:.6f} differs from its sidecar's "
          f"{recorded['total_loss']:.6f} by more than {tolerance}; stopping before training")
    delta, paired = evaluator.compare(official, rows, hold, reference)
    record = dict(reference_record(evaluator, official, rows, hold), sidecar_validation=recorded,
                  difference_from_sidecar=differences, identical_to_sidecar=official == recorded, tolerance=tolerance,
                  delta_versus_p=delta, paired_versus_p=paired, sidecar_delta_total=sidecar["delta_total"])
    return (official, rows, hold), record


def round_data_specs(values):
    return [(Path(directory), played_by) for directory, played_by in values]


def command_train(args):
    output = Path(args.output)
    if (output / "summary.json").exists():
        print(json.dumps(dict(event="already-complete", summary=str(output / "summary.json"))), flush=True)
        return 0
    configure(args.device, threads=args.threads)
    timings = {}
    started = time.perf_counter()
    paths = data_paths(args.data_config)
    validation = CachedSupervisedDataset.open(split="validation", **paths)
    supervised = CachedSupervisedDataset.open(split="train", **paths)
    timings["cache_open_seconds"] = time.perf_counter() - started
    start_sha, p_sha = sha256(args.start_checkpoint), sha256(args.p_checkpoint)
    check(start_sha == args.start_sha256, f"{args.start_checkpoint} sha256 {start_sha} != {args.start_sha256}")
    check(p_sha == args.p_sha256, f"{args.p_checkpoint} sha256 {p_sha} != {args.p_sha256}")
    check(start_sha != p_sha, "the start checkpoint must differ from the forgetting reference")
    specs = round_data_specs(args.round_data)
    check(len(specs) >= 2 and specs[0][1] == p_sha and specs[-1][1] == start_sha,
          "round data: round 1 (played by P) first, the last round played by the start checkpoint")
    started = time.perf_counter()
    union = U.RoundUnion.open(specs, registries={n: supervised.source_identity["authoritative"][n] for n in R.REGISTRIES})
    timings["round_data_open_seconds"] = time.perf_counter() - started
    check(union.manifest["supervised_source_id"] == digest(supervised.source_identity),
          "round data was filtered against another Stockfish source")
    check(union.manifest["holdout"]["rule"] == R.HOLDOUT_RULE, f"round data must use the hold-out rule {R.HOLDOUT_RULE}")
    sidecar_path = Path(args.start_checkpoint).with_suffix(".json")
    start_sidecar = R.read_json(sidecar_path)
    check(start_sidecar["checkpoint_sha256"] == start_sha, f"{sidecar_path} is not the start checkpoint's sidecar")
    total = args.updates or math.ceil(len(union.train) / R.SETTINGS.selfplay_batch_size)
    every = args.checkpoint_every
    stops = checkpoint_updates(total, every)
    check(args.stop_after is None or args.stop_after in stops, f"--stop-after must be a checkpoint update: {stops}")
    p = load_inference(args.p_checkpoint, device=args.device)
    start = load_inference(args.start_checkpoint, device=args.device)
    check(replace(start.config, device="cpu") == replace(p.config, device="cpu"),
          "start checkpoint and P have different model configurations")
    config = replace(start.config, device=args.device)
    enforced = not args.no_hard_stop
    provenance = run_provenance(args.recipe, start=start, start_sha=start_sha, p=p, p_sha=p_sha, union=union,
                                supervised=supervised, validation=validation, total=total, every=every,
                                batch_size=args.eval_batch_size, hard_stop=enforced, amendment=args.amendment_note)
    run_config = dict(version=U.RUN_VERSION, provenance=provenance, model_config=asdict(config), device=args.device,
                      threads=args.threads, runtime=runtime_config(),
                      cuda_runtime=cuda_runtime_config() if args.device == "cuda:0" else None,
                      start_checkpoint=str(args.start_checkpoint), p_checkpoint=str(args.p_checkpoint),
                      round_data=[[str(d), s] for d, s in specs], data_config=str(args.data_config),
                      expect_p_total=args.expect_p_total, p_tolerance=args.p_tolerance,
                      start_tolerance=args.start_tolerance)
    if (output / "config.json").exists():
        check(R.read_json(output / "config.json") == json.loads(json.dumps(run_config)),
              f"{output} holds a run with other settings, sources or runtime; resume needs identical ones")
    else:
        check(not output.exists() or not any(output.iterdir()), f"{output} exists without config.json")
        output.mkdir(parents=True, exist_ok=True)
        R.publish_json(output / "config.json", run_config)
    sources = R.sampler_sources(supervised, union)
    initial = MixedState.initial(sources, R.SETTINGS, seed=U.SAMPLING_SEED)
    identity = R.sampler_identity(initial)
    latest, orphans = R.resume_point(output)
    records, superseded, partial = R.read_metrics(output)
    begin = latest[0] if latest else 0
    check(args.stop_after is None or args.stop_after >= begin, f"--stop-after {args.stop_after} is before the resume point {begin}")
    check(len(records) >= begin, "metrics are missing updates before the latest checkpoint")
    if latest:
        check(digest(records[:begin]) == latest[1]["metrics_sha256"], "metrics before the latest checkpoint were altered")
    later = len(records) - begin
    records = records[:begin]
    moved = R.set_aside(output, orphans)
    evaluator = RoundEvaluator(validation, union.holdout, args.eval_batch_size)
    # P, measured fresh in this job; every forgetting delta below is against this measurement.
    official, rows, hold, measure_timings = evaluator.measure(p.model)
    timings["p_measure_seconds"] = measure_timings["validation_seconds"] + measure_timings["holdout_seconds"]
    if args.expect_p_total is not None:
        check(abs(official["total_loss"] - args.expect_p_total) <= args.p_tolerance,
              f"P validation total loss {official['total_loss']:.6f} does not reproduce "
              f"{args.expect_p_total} within {args.p_tolerance}; stopping before training")
    reference = (official, rows, hold)
    started = time.perf_counter()
    start_reference, start_record = measure_start(evaluator, start, start_sidecar, args.start_tolerance, reference)
    timings["start_measure_seconds"] = time.perf_counter() - started
    measured = json.loads(json.dumps(dict(p=reference_record(evaluator, official, rows, hold), start=start_record)))
    if (output / "reference.json").exists():
        check(R.read_json(output / "reference.json") == measured,
              "P's or the start checkpoint's fresh measurement differs from this run's earlier measurement")
    else:
        R.publish_json(output / "reference.json", measured)
    del p
    remaining = [u for u in stops if u > begin]
    existing = [s.stat().st_size for s in output.glob("update-*.pt")]
    per_checkpoint = max(existing) if existing else math.ceil(Path(args.start_checkpoint).stat().st_size * 1.25)
    disk_guard(args)(len(remaining) * per_checkpoint + GIB, f"{len(remaining)} checkpoints and reserve")
    started = time.perf_counter()
    if latest:
        restored = R.restore_round(output / R.checkpoint_name(begin), expected_config=config,
                                   expected_provenance=provenance, expected_settings=R.SETTINGS,
                                   expected_identity=identity)
        model, optimizer, state = restored.model, restored.optimizer, restored.state
        check(state.updates == begin, "restored update count differs from its sidecar")
    else:
        random.seed(U.RUN_SEED)
        np.random.seed(U.RUN_SEED)
        torch.manual_seed(U.RUN_SEED)
        model = start.model
        model.train()
        optimizer = R.build_round_optimizer(model)
        state = initial
    timings["restore_seconds" if latest else "initialise_seconds"] = time.perf_counter() - started
    number = len(R.segments(output)) + 1
    segment = output / f"metrics-{number:03d}-from-{begin:06d}.jsonl"
    header = dict(event="segment", segment=number, from_update=begin, superseded_lines=later,
                  partial_lines=partial, orphans_set_aside=moved,
                  p_reference_rows_sha256=measured["p"]["validation_rows_sha256"],
                  start_reference_rows_sha256=measured["start"]["validation_rows_sha256"],
                  started_utc=R._now(), timings=timings)
    print(json.dumps(header), flush=True)
    if args.device == "cuda:0":
        torch.cuda.reset_peak_memory_stats(0)
    finished = latest is not None and (begin == total or R.stopped_training(latest[1]))
    with segment.open("x") as log:
        log.write(json.dumps(header, allow_nan=False) + "\n")
        log.flush()
        os.fsync(log.fileno())
        if not finished and args.stop_after != begin:
            segment_start = time.perf_counter()
            update = begin
            while update < total:
                before = time.perf_counter()
                record, state = R.round_step(model, optimizer, recipe=args.recipe, supervised=supervised,
                                             rows=union.train, state=state,
                                             max_gradient_norm=R.OPTIMIZER_SETTINGS.max_gradient_norm)
                update = state.updates
                record["update_seconds"] = time.perf_counter() - before
                record["elapsed_seconds"] = time.perf_counter() - segment_start
                records.append(record)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if update in stops:
                    sidecar = write_pair(output, update, model=model, optimizer=optimizer, state=state,
                                         provenance=provenance, recipe=args.recipe, records=records, log=log,
                                         evaluator=evaluator, reference=reference, start_reference=start_reference,
                                         enforce_stop=enforced)
                    print(json.dumps(dict(event="checkpoint", update=update, delta=sidecar["evaluation"]["delta"],
                                          delta_versus_start=sidecar["evaluation"]["versus_start"]["delta"],
                                          validation=sidecar["evaluation"]["validation"],
                                          timings=sidecar["timings"])), flush=True)
                    if sidecar["training_stopped"] or update == args.stop_after:
                        break
    if done(output, total):
        return finalize(output, args, total=total, union=union, provenance=provenance, start=start,
                        start_sha=start_sha)
    print(json.dumps(dict(event="stopped-as-requested", update=args.stop_after)), flush=True)
    return 0


def done(output, total):
    latest, _ = R.resume_point(output)
    return latest is not None and (latest[0] == total or R.stopped_training(latest[1]))


def write_pair(output, update, *, model, optimizer, state, provenance, recipe, records, log, evaluator, reference,
               start_reference, enforce_stop=True):
    """Round 1's write_pair with r2- checkpoint ids and a report-only comparison with the start checkpoint."""
    official, rows, hold, measure_timings = evaluator.measure(model)
    delta, paired = evaluator.compare(official, rows, hold, reference)
    start_delta, start_paired = evaluator.compare(official, rows, hold, start_reference)
    log.flush()
    os.fsync(log.fileno())
    path = output / R.checkpoint_name(update)
    started = time.perf_counter()
    save_checkpoint(path, model=model, optimizer=optimizer, checkpoint_id=f"r2-{recipe}-update-{update:06d}",
                    provenance=provenance, weights=R.SETTINGS.weights,
                    progress=Progress(update, 0, state.supervised.samples_seen + state.selfplay.samples_seen),
                    mixed_state=state)
    save_seconds = time.perf_counter() - started
    sidecar = dict(version=R.SIDECAR_VERSION, update=update, recipe=recipe, checkpoint=path.name,
                   checkpoint_sha256=sha256(path), checkpoint_bytes=path.stat().st_size,
                   metrics_records=len(records), metrics_sha256=digest(records),
                   evaluation=dict(validation=official, delta=delta, paired=paired,
                                   paired_labels=R.paired_labels(paired["validation_total"]),
                                   validation_rows_sha256=R.rows_digest(rows),
                                   holdout=evaluator.holdout_summary(hold), holdout_rows_sha256=R.rows_digest(hold),
                                   versus_start=dict(delta=start_delta, paired=start_paired, role="report-only")),
                   delta_total=delta["total_loss"], flag=bool(delta["total_loss"] > R.FLAG_DELTA),
                   # hard_stop records the breach; training_stopped whether training ended here.
                   hard_stop=bool(delta["total_loss"] > R.STOP_DELTA),
                   training_stopped=bool(enforce_stop and delta["total_loss"] > R.STOP_DELTA),
                   sampler=dict(selfplay_epoch=state.selfplay.epoch, selfplay_cursor=state.selfplay.cursor,
                                selfplay_samples_drawn=state.selfplay.samples_seen),
                   timings=dict(measure_timings, checkpoint_save_seconds=save_seconds), written_utc=R._now())
    R.publish_json(output / R.sidecar_name(update), sidecar)
    return sidecar


def finalize(output, args, *, total, union, provenance, start, start_sha):
    if (output / "summary.json").exists():
        return 0
    latest, _ = R.resume_point(output)
    check(latest is not None, "no checkpoint to summarise")
    sidecars = [R.read_json(output / R.sidecar_name(u)) for u in
                sorted(int(p.name[7:13]) for p in output.glob("update-*.json"))]
    evaluations = [dict(update=s["update"], checkpoint=s["checkpoint"], delta_total=s["delta_total"],
                        flag=s["flag"], hard_stop=s["hard_stop"], paired_labels=s["evaluation"]["paired_labels"],
                        delta_total_versus_start=s["evaluation"]["versus_start"]["delta"]["total_loss"])
                   for s in sidecars]
    stopped = R.stopped_training(latest[1])
    completed = latest[0] == total and not stopped
    check(completed or stopped, "run is neither complete nor stopped")
    original = R.select_candidate(evaluations, completed=completed, flag=R.FLAG_DELTA, stop=R.STOP_DELTA)
    enforced = not args.no_hard_stop
    mapped = U.start_candidate(original, start_path=args.start_checkpoint, start_sha256=start_sha,
                               start_checkpoint_id=start.checkpoint_id)
    choice = mapped if enforced else R.exploratory_candidate(evaluations, completed=completed,
                                                              flag=R.FLAG_DELTA, stop=R.STOP_DELTA)
    records, superseded, partial = R.read_metrics(output)
    updates = len(records)
    n = len(union.train)
    details = provenance["recipe_details"]
    per_update = details["trained_positions_per_update"]
    drawn = updates * R.SETTINGS.selfplay_batch_size
    seconds = [r["update_seconds"] for r in records]
    selectable = enforced and provenance["recipe"] == "mix"
    summary = dict(version=U.RUN_VERSION, recipe=provenance["recipe"], name=details["name"], completed=completed,
        updates=updates, total_updates=total, hard_stop_enforced=enforced, exploratory=not enforced,
        selectable=selectable, role="selectable" if selectable else "report-only",
        selection=choice, original_rule_selection=mapped,
        threshold_breaches=[e["update"] for e in evaluations if e["hard_stop"]],
        candidate=U.candidate_record(choice, output=output, sha256=sha256),
        start_checkpoint=dict(path=str(args.start_checkpoint), sha256=start_sha, checkpoint_id=start.checkpoint_id),
        final_delta_total=latest[1]["delta_total"], final_flag_over_0_05=bool(latest[1]["delta_total"] > R.FLAG_DELTA),
        final_paired_labels=latest[1]["evaluation"]["paired_labels"],
        final_versus_start=latest[1]["evaluation"]["versus_start"],
        evaluations=evaluations,
        trained_positions=dict(selfplay=updates * per_update["selfplay"], stockfish=updates * per_update["stockfish"]),
        discarded_sampler_indices=updates * details["discarded_sampler_indices_per_update"],
        selfplay=dict(training_positions=n, by_round=[len(p.train) for p in union.parts], drawn=drawn,
                      seen_twice=max(0, drawn - n), epochs_started=records[-1]["selfplay_epoch"] if records else 0),
        exclusions=[p.manifest["counts"] for p in union.parts], superseded_metric_lines=superseded,
        partial_metric_lines=partial,
        update_seconds=dict(median=float(np.median(seconds)), mean=float(np.mean(seconds)),
                            p90=float(np.quantile(seconds, 0.9))) if seconds else None,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0) if args.device == "cuda:0" else None,
        test_split_used=False)
    R.publish_json(output / "summary.json", summary)
    print(json.dumps(dict(event="summary", name=summary["name"], role=summary["role"], selection=choice,
                          original_rule_selection=mapped, candidate=summary["candidate"],
                          final_delta_total=summary["final_delta_total"])), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def disk(command):
        command.add_argument("--quota-bytes", type=int, default=100_000_000_000,
                             help="quota for the du fallback when `quota -w` does not parse")
        command.add_argument("--skip-disk-check", action="store_true", help="tests and machines without a quota only")

    space = commands.add_parser("disk", help="do the round-2 arrays (upper bound) plus the reserve fit the quota")
    space.add_argument("--record", type=Path, required=True)
    space.add_argument("--reserve-bytes", type=int, default=GIB)
    space.add_argument("--quota-bytes", type=int, default=100_000_000_000)
    data = commands.add_parser("data", help="build the round-2 arrays once, or reopen and verify them")
    data.add_argument("--record", type=Path, required=True)
    data.add_argument("--selection", type=Path, required=True)
    data.add_argument("--data-config", type=Path, required=True)
    data.add_argument("--start-checkpoint", type=Path, required=True)
    data.add_argument("--start-sha256", required=True)
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--expected-games", type=int, required=True)
    data.add_argument("--reserve-bytes", type=int, default=GIB)
    disk(data)
    train = commands.add_parser("train", help="run or resume one recipe")
    train.add_argument("--recipe", choices=R.RECIPES, required=True)
    train.add_argument("--data-config", type=Path, required=True)
    train.add_argument("--round-data", nargs=2, action="append", required=True, metavar=("DIRECTORY", "PLAYED_BY_SHA256"),
                       help="one per round, in round order")
    train.add_argument("--start-checkpoint", type=Path, required=True)
    train.add_argument("--start-sha256", required=True)
    train.add_argument("--p-checkpoint", type=Path, required=True)
    train.add_argument("--p-sha256", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    train.add_argument("--threads", type=int, default=1)
    train.add_argument("--updates", type=int, help="total updates (default ceil(training positions / 24)); dry runs only")
    train.add_argument("--checkpoint-every", type=int, default=R.CHECKPOINT_EVERY)
    train.add_argument("--stop-after", type=int, help="exit after the checkpoint at this update (resume tests)")
    train.add_argument("--no-hard-stop", action="store_true",
                       help="report-only runs: record stop-threshold breaches but keep training (needs --amendment-note)")
    train.add_argument("--amendment-note", help="the approved decision behind --no-hard-stop, recorded in the provenance")
    train.add_argument("--eval-batch-size", type=int, default=32)
    train.add_argument("--expect-p-total", type=float, help="P's recorded validation total loss to reproduce")
    train.add_argument("--p-tolerance", type=float, default=0.001)
    train.add_argument("--start-tolerance", type=float, default=U.START_TOLERANCE,
                       help="largest allowed |fresh - sidecar| validation total loss of the start checkpoint")
    disk(train)
    args = parser.parse_args(argv)
    if args.command == "train" and (args.checkpoint_every < 1 or (args.updates is not None and args.updates < 1)):
        parser.error("--checkpoint-every and --updates must be positive")
    if args.command == "train" and args.no_hard_stop != bool(args.amendment_note):
        parser.error("--no-hard-stop needs --amendment-note, and --amendment-note needs --no-hard-stop")
    handler = dict(disk=command_disk, data=command_data, train=command_train)[args.command]
    try:
        return handler(args)
    except Mismatch as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
