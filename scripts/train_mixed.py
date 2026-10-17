"""Round-1 training from P: build the self-play arrays, train R1-mix or R1-pure, compare and check runs.

data          verify the training pointer and its buffer, stream the replay buffer once into
              compact filtered arrays (shared by every run), or reopen and verify existing ones.
baseline      measure P only (no updates): validation metrics, per-position loss spread and the
              interval of its mean across validation games, self-play hold-out losses. Prints a
              report to review before full training; a rerun must reproduce the recorded one.
train         run or resume one recipe in its own directory: P is measured fresh first (it must
              reproduce the expected validation loss), then updates with a checkpoint and sidecar
              every --checkpoint-every updates and at the last one, the forgetting check at each
              checkpoint (flag and stop on the measured delta; paired-interval labels reported
              alongside), and summary.json at the end. Rerun the same command after any stop.
compare       exact comparison of two runs at one update: weights, optimizer, sampler, RNG and
              training metrics (timing excluded). Exit 1 on any difference.
inference     load a checkpoint through load_inference and check its validation metrics equal the
              ones its training process recorded. Exit 1 on any difference.
timings       one JSON report of the measured timings of the data build and of the given runs.

Updates use chess_rl.training.round_training; see it for the guarantees.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
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
from chess_rl.selfplay_rounds import POINTER_SCHEMA, free_quota, verify_buffer
from chess_rl.training import round_training as R
from chess_rl.training.checkpoints import (Progress, cuda_runtime_config, load_inference, runtime_config,
                                           save_checkpoint)
from chess_rl.training.mixed_state import MixedState
from chess_rl.training.optimizers import describe
from scripts.train_supervised import validate

GIB = 1 << 30


class Mismatch(ValueError):
    """Inputs, outputs or a measurement differ from what this run requires; nothing was changed."""


def check(condition, message):
    if not condition:
        raise Mismatch(message)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def disk_guard(args):
    """require_free(bytes, what): raises unless the quota has that much free; skipped only on request."""
    def require_free(need, what):
        if args.skip_disk_check:
            return
        free = free_quota(args.quota_bytes)
        check(free["free_bytes"] >= need, f"not enough quota for {what}: need {need / GIB:.2f} GiB, "
              f"free {free['free_bytes'] / GIB:.2f} GiB ({free['method']})")
    return require_free


def checkpoint_updates(total, every):
    return sorted({u for u in range(every, total + 1, every)} | {total})


# data

def check_pointer(pointer, *, branch, games, positions):
    """Static checks of the training pointer against what this round expects."""
    check(pointer.get("schema") == POINTER_SCHEMA, "not a training-buffer pointer")
    check(pointer["branch"] == branch, f"pointer branch {pointer['branch']!r}, expected {branch!r}")
    recorded = pointer["checks"]["extended" if branch == "extended" else "core"]
    check(pointer["buffer"] == recorded["buffer"] and pointer["replay"] == str(Path(pointer["buffer"]) / "replay.sqlite"),
          "pointer buffer differs from its recorded check")
    check([h["half"] for h in recorded["halves"]] == pointer["halves"], "pointer halves differ from its recorded check")
    check(pointer["games"] == recorded["games"] == games, f"pointer games {pointer['games']}, expected {games}")
    check(pointer["training_positions_before_holdout"] == recorded["training_positions"] == positions,
          f"pointer positions {pointer['training_positions_before_holdout']}, expected {positions}")
    for name in ("core_choice", "extension_choice"):
        if pointer.get(name) is not None:
            check(sha256(pointer[name]["path"]) == pointer[name]["sha256"], f"{pointer[name]['path']} changed since the pointer")
    return recorded


def check_verified(recorded, verified, checkpoint_sha256):
    """verify_buffer's fresh result must equal the pointer's recorded check (timing aside)."""
    for key in set(recorded) | set(verified):
        if key != "audit_seconds":
            check(recorded.get(key) == verified.get(key), f"buffer verification differs from the pointer: {key}")
    check(verified["checkpoint_sha256"] == checkpoint_sha256, "the buffer's games were played by another checkpoint")


def command_data(args):
    output = Path(args.output)
    paths = data_paths(args.data_config)
    identity = current_source_identity(split="train", **paths)
    p_sha = sha256(args.p_checkpoint)
    check(p_sha == args.p_sha256, f"{args.p_checkpoint} sha256 {p_sha} != {args.p_sha256}")
    pointer_sha = sha256(args.pointer)
    if output.exists():
        started = time.perf_counter()
        data = R.RoundData.open(output, registries={n: identity["authoritative"][n] for n in R.REGISTRIES})
        source = data.manifest["source"]
        check(source["pointer"]["sha256"] == pointer_sha, f"{output} was built from another pointer")
        check(data.manifest["checkpoint_sha256"] == p_sha and data.manifest["supervised_source_id"] == digest(identity),
              f"{output} was built for another P or Stockfish source")
        print(json.dumps(dict(event="reused", output=str(output), replay_id=data.replay_id,
                              open_seconds=time.perf_counter() - started, counts=data.manifest["counts"])), flush=True)
        return 0
    pointer = json.loads(Path(args.pointer).read_text())
    recorded = check_pointer(pointer, branch=args.expected_branch, games=args.expected_games,
                             positions=args.expected_positions)
    require_free = disk_guard(args)
    # Upper bound before 20 minutes of audits: every position kept, 218 legal moves each.
    per_checkpoint = math.ceil(Path(args.p_checkpoint).stat().st_size * 1.25)
    checkpoints = 2 * len(checkpoint_updates(math.ceil(args.expected_positions / R.SETTINGS.selfplay_batch_size),
                                             R.CHECKPOINT_EVERY)) + args.extra_checkpoints
    extra = checkpoints * per_checkpoint + GIB
    bound = R.planned_bytes({"all": args.expected_positions}, {"all": args.expected_positions * 218})
    require_free(bound + extra, "round data (upper bound) plus planned checkpoints and reserve")
    timings = {}
    started = time.perf_counter()
    verified = verify_buffer(pointer["buffer"], pointer["halves"])
    timings["verify_buffer_seconds"] = time.perf_counter() - started
    check_verified(recorded, verified, p_sha)
    started = time.perf_counter()
    buffer = ReplayBuffer(pointer["replay"])  # full integrity and target re-audit
    timings["replay_open_seconds"] = time.perf_counter() - started
    summary = buffer.summary()
    check(summary["source_counts"] == verified["source_counts"]
          and summary["training_positions"] == verified["training_positions"], "buffer changed after verification")
    source = dict(pointer=dict(path=str(args.pointer), sha256=pointer_sha, branch=pointer["branch"]),
                  buffer=pointer["buffer"], replay=pointer["replay"], verified=verified)
    data = R.build_round_data(output, buffer=buffer, prepared=paths["prepared"], identity=identity,
                              checkpoint_sha256=p_sha, source=source, require_free=require_free,
                              extra_bytes=extra, timings=timings)
    print(json.dumps(dict(event="built", output=str(output), replay_id=data.replay_id, counts=data.manifest["counts"],
                          timings=data.manifest["timings"])), flush=True)
    return 0


# train

class Evaluator:
    """Validation (official validate() plus per-position rows) and self-play hold-out metrics."""

    def __init__(self, validation, holdout, batch_size):
        self.validation, self.holdout, self.batch_size = validation, holdout, batch_size
        self.validation_groups = [validation.reference(i)[0] for i in range(len(validation))]
        self.holdout_groups = list(holdout.game_ids)

    def measure(self, model):
        started = time.perf_counter()
        official = {k: v.item() if isinstance(v, np.generic) else v
                    for k, v in validate(model, self.validation, self.batch_size).items()}
        rows = R.validation_rows(model, self.validation, self.batch_size)
        validation_seconds = time.perf_counter() - started
        # Self-check of the per-position code against validate(); not a training tolerance.
        check(abs(float(rows["total"].mean()) - official["total_loss"]) <= 1e-4,
              "per-position validation losses disagree with validate()")
        started = time.perf_counter()
        hold = R.holdout_rows(model, self.holdout, self.batch_size)
        holdout_seconds = time.perf_counter() - started
        return official, rows, hold, dict(validation_seconds=validation_seconds, holdout_seconds=holdout_seconds)

    def holdout_summary(self, hold):
        return {k: R.clustered_bootstrap(hold[k], self.holdout_groups) for k in ("policy", "wdl", "total")}

    def compare(self, official, rows, hold, reference):
        ref_official, ref_rows, ref_hold = reference
        delta = {k: official[k] - ref_official[k] for k in ("policy_loss", "wdl_loss", "total_loss", "policy_accuracy")}
        paired = dict(validation_total=R.clustered_bootstrap(rows["total"] - ref_rows["total"], self.validation_groups),
                      validation_policy=R.clustered_bootstrap(rows["policy"] - ref_rows["policy"], self.validation_groups),
                      validation_wdl=R.clustered_bootstrap(rows["wdl"] - ref_rows["wdl"], self.validation_groups),
                      holdout_total=R.clustered_bootstrap(hold["total"] - ref_hold["total"], self.holdout_groups))
        return delta, paired


def reference_record(evaluator, official, rows, hold):
    return dict(validation=official, validation_rows_sha256=R.rows_digest(rows),
                holdout_rows_sha256=R.rows_digest(hold),
                across_validation_games=dict(note=R.UNCERTAINTY_NOTE,
                    per_position_total=R.describe_values(rows["total"]),
                    mean_total_interval=R.clustered_bootstrap(rows["total"], evaluator.validation_groups)),
                holdout=evaluator.holdout_summary(hold))


BASELINE_VERSION = "round1-p-baseline-v1"


def baseline_banner(record):
    """The human-readable baseline report printed before the JSON."""
    v, spread = record["validation"], record["across_validation_games"]
    interval, quantiles = spread["mean_total_interval"], spread["per_position_total"]["quantiles"]
    expected = record["expected"]
    lines = ["=" * 100, "P BASELINE: review with the completed dry-run results before submitting train-r1.sbatch",
             f"P {record['p']['sha256'][:12]} ({record['p']['checkpoint_id']}) on {v['positions']} validation positions "
             f"from {interval['games']} games",
             f"  total {v['total_loss']:.6f}  policy {v['policy_loss']:.6f}  wdl {v['wdl_loss']:.6f}  "
             f"top-1 {v['policy_accuracy']:.4f}"]
    if expected["total"] is not None:
        lines.append(f"  reproduces {expected['total']} within {expected['tolerance']}: "
                     f"{'yes' if expected['reproduces'] else 'NO (training refuses to start)'}")
    lines += [f"  per-position total loss: mean {spread['per_position_total']['mean']:.4f}  "
              f"sd {spread['per_position_total']['sd']:.4f}  q05 {quantiles['0.05']:.4f}  q50 {quantiles['0.5']:.4f}  "
              f"q95 {quantiles['0.95']:.4f}",
              f"  mean total loss, 95% interval across validation games: [{interval['low']:.6f}, {interval['high']:.6f}] "
              f"(half-width {(interval['high'] - interval['low']) / 2:.6f})"]
    for name, h in record["holdout"].items():
        if h["mean"] is not None:
            lines.append(f"  self-play hold-out {name}: {h['mean']:.6f} [{h['low']:.6f}, {h['high']:.6f}] "
                         f"({h['positions']} positions, {h['games']} games)")
    lines += [f"  {R.UNCERTAINTY_NOTE}.",
              "  The forgetting delta is paired (same positions); its uncertainty is the paired interval recorded "
              "at each checkpoint, not this one.", "=" * 100]
    return "\n".join(lines)


def command_baseline(args):
    output = Path(args.output)
    configure(args.device, threads=args.threads)
    paths = data_paths(args.data_config)
    identity = current_source_identity(split="train", **paths)
    started = time.perf_counter()
    validation = CachedSupervisedDataset.open(split="validation", **paths)
    open_seconds = time.perf_counter() - started
    data = R.RoundData.open(args.round_data, registries={n: identity["authoritative"][n] for n in R.REGISTRIES})
    p_sha = sha256(args.p_checkpoint)
    check(p_sha == args.p_sha256, f"{args.p_checkpoint} sha256 {p_sha} != {args.p_sha256}")
    check(data.manifest["checkpoint_sha256"] == p_sha and data.manifest["supervised_source_id"] == digest(identity),
          "round data was built for another P or Stockfish source")
    p = load_inference(args.p_checkpoint, device=args.device)
    evaluator = Evaluator(validation, data.holdout, args.eval_batch_size)
    official, rows, hold, timings = evaluator.measure(p.model)
    reproduces = None if args.expect_p_total is None else bool(
        abs(official["total_loss"] - args.expect_p_total) <= args.p_tolerance)
    record = dict(version=BASELINE_VERSION, p=dict(path=str(args.p_checkpoint), sha256=p_sha, checkpoint_id=p.checkpoint_id),
                  round_data=dict(path=str(args.round_data), replay_id=data.replay_id),
                  validation_source_id=digest(validation.source_identity), eval_batch_size=args.eval_batch_size,
                  device=args.device, threads=args.threads, runtime=runtime_config(),
                  expected=dict(total=args.expect_p_total, tolerance=args.p_tolerance, reproduces=reproduces),
                  **reference_record(evaluator, official, rows, hold),
                  timings=dict(timings, validation_cache_open_seconds=open_seconds))
    record = json.loads(json.dumps(record))
    path = output / "baseline.json"
    if path.exists():
        check(R.strip_timing(R.read_json(path)) == R.strip_timing(record),
              f"P's measurement differs from the recorded {path}")
        event = "baseline-reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        R.publish_json(path, record)
        event = "baseline-recorded"
    print(baseline_banner(record), flush=True)
    print(json.dumps(dict(event=event, path=str(path), timings=record["timings"])), flush=True)
    check(reproduces is not False, "P does not reproduce its recorded validation loss; do not train")
    return 0


def run_provenance(recipe, *, p, p_sha, data, supervised, validation, total, every, batch_size,
                   hard_stop=True, amendment=None, passes=None):
    source_id, membership = R.supervised_identity(supervised)
    record = dict(schema=R.RUN_VERSION, purpose="round-1 training from P on the round-1 self-play replay buffer",
        recipe=recipe, recipe_details=R.recipe_provenance(recipe),
        initial_checkpoint=dict(sha256=p_sha, checkpoint_id=p.checkpoint_id),
        round_data=dict(replay_id=data.replay_id, training_positions=len(data.train),
                        holdout_positions=len(data.holdout), counts=data.manifest["counts"]),
        stockfish=dict(source_id=source_id, membership_sha256=membership, positions=len(supervised)),
        validation=dict(source_id=digest(validation.source_identity), positions=len(validation)),
        sampler=dict(settings=asdict(R.SETTINGS), seed=R.SAMPLING_SEED, note=R.SAMPLER_NOTE),
        optimizer=dict(describe(R.OPTIMIZER_SETTINGS), schedule="constant learning rate, no warmup",
                       fresh="new AdamW at update 0"),
        loss_weights=dict(policy=R.SETTINGS.policy_weight, wdl=R.SETTINGS.wdl_weight),
        run_seed=R.RUN_SEED, total_updates=total, checkpoint_every=every,
        thresholds=dict(flag=R.FLAG_DELTA, stop=R.STOP_DELTA,
                        delta="checkpoint validation total loss minus P's, P measured fresh in the same job"),
        evaluation=dict(batch_size=batch_size, bootstrap=dict(resamples=R.BOOTSTRAP_RESAMPLES, seed=R.BOOTSTRAP_SEED)))
    # Amendment keys appear only when used, so runs under the original plan keep their provenance unchanged.
    if passes is not None:
        record["passes_over_selfplay"] = passes
    if amendment is not None and hard_stop:
        record["amendment"] = amendment
    if not hard_stop:
        record["thresholds"]["hard_stop_enforced"] = False
        record["exploratory"] = dict(amendment=amendment, rule=(
            "every breach of the stop threshold is recorded but training continues; the candidate is the "
            "final checkpoint; the original rule's candidate is reported alongside and is not changed"))
    return record


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
    started = time.perf_counter()
    data = R.RoundData.open(args.round_data,
                            registries={n: supervised.source_identity["authoritative"][n] for n in R.REGISTRIES})
    timings["round_data_open_seconds"] = time.perf_counter() - started
    p_sha = sha256(args.p_checkpoint)
    check(p_sha == args.p_sha256, f"{args.p_checkpoint} sha256 {p_sha} != {args.p_sha256}")
    check(data.manifest["checkpoint_sha256"] == p_sha, "round data was built for another P")
    check(data.manifest["supervised_source_id"] == digest(supervised.source_identity),
          "round data was filtered against another Stockfish source")
    total = args.updates or math.ceil((args.passes or 1) * len(data.train) / R.SETTINGS.selfplay_batch_size)
    every = args.checkpoint_every
    stops = checkpoint_updates(total, every)
    check(args.stop_after is None or args.stop_after in stops, f"--stop-after must be a checkpoint update: {stops}")
    p = load_inference(args.p_checkpoint, device=args.device)
    config = replace(p.config, device=args.device)
    provenance = run_provenance(args.recipe, p=p, p_sha=p_sha, data=data, supervised=supervised,
                                validation=validation, total=total, every=every, batch_size=args.eval_batch_size,
                                hard_stop=not args.no_hard_stop, amendment=args.amendment_note, passes=args.passes)
    run_config = dict(version=R.RUN_VERSION, provenance=provenance, model_config=asdict(config), device=args.device,
                      threads=args.threads, runtime=runtime_config(),
                      cuda_runtime=cuda_runtime_config() if args.device == "cuda:0" else None,
                      p_checkpoint=str(args.p_checkpoint), round_data=str(args.round_data),
                      data_config=str(args.data_config), expect_p_total=args.expect_p_total,
                      p_tolerance=args.p_tolerance)
    if (output / "config.json").exists():
        check(R.read_json(output / "config.json") == json.loads(json.dumps(run_config)),
              f"{output} holds a run with other settings, sources or runtime; resume needs identical ones")
    else:
        check(not output.exists() or not any(output.iterdir()), f"{output} exists without config.json")
        output.mkdir(parents=True, exist_ok=True)
        R.publish_json(output / "config.json", run_config)
    sources = R.sampler_sources(supervised, data)
    initial = MixedState.initial(sources, R.SETTINGS, seed=R.SAMPLING_SEED)
    identity = R.sampler_identity(initial)
    latest, orphans = R.resume_point(output)
    records, superseded, partial = R.read_metrics(output)
    start = latest[0] if latest else 0
    check(args.stop_after is None or args.stop_after >= start, f"--stop-after {args.stop_after} is before the resume point {start}")
    check(len(records) >= start, "metrics are missing updates before the latest checkpoint")
    if latest:
        check(digest(records[:start]) == latest[1]["metrics_sha256"], "metrics before the latest checkpoint were altered")
    later = len(records) - start
    records = records[:start]
    moved = R.set_aside(output, orphans)
    evaluator = Evaluator(validation, data.holdout, args.eval_batch_size)
    # P, measured fresh in this job; every delta below is against this measurement.
    official, rows, hold, measure_timings = evaluator.measure(p.model)
    timings["p_measure_seconds"] = measure_timings["validation_seconds"] + measure_timings["holdout_seconds"]
    if args.expect_p_total is not None:
        check(abs(official["total_loss"] - args.expect_p_total) <= args.p_tolerance,
              f"P validation total loss {official['total_loss']:.6f} does not reproduce "
              f"{args.expect_p_total} within {args.p_tolerance}; stopping before training")
    reference = (official, rows, hold)
    p_record = reference_record(evaluator, official, rows, hold)
    if (output / "reference.json").exists():
        saved = R.read_json(output / "reference.json")
        check(saved == json.loads(json.dumps(p_record)),
              "P's fresh measurement differs from this run's earlier measurement of P")
    else:
        R.publish_json(output / "reference.json", p_record)
    remaining = [u for u in stops if u > start]
    existing = [s.stat().st_size for s in output.glob("update-*.pt")]
    per_checkpoint = max(existing) if existing else math.ceil(Path(args.p_checkpoint).stat().st_size * 1.25)
    disk_guard(args)(len(remaining) * per_checkpoint + GIB, f"{len(remaining)} checkpoints and reserve")
    started = time.perf_counter()
    if latest:
        restored = R.restore_round(output / R.checkpoint_name(start), expected_config=config,
                                   expected_provenance=provenance, expected_settings=R.SETTINGS,
                                   expected_identity=identity)
        model, optimizer, state = restored.model, restored.optimizer, restored.state
        check(state.updates == start, "restored update count differs from its sidecar")
    else:
        random.seed(R.RUN_SEED)
        np.random.seed(R.RUN_SEED)
        torch.manual_seed(R.RUN_SEED)
        model = p.model
        model.train()
        optimizer = R.build_round_optimizer(model)
        state = initial
    timings["restore_seconds" if latest else "initialise_seconds"] = time.perf_counter() - started
    number = len(R.segments(output)) + 1
    segment = output / f"metrics-{number:03d}-from-{start:06d}.jsonl"
    header = dict(event="segment", segment=number, from_update=start, superseded_lines=later,
                  partial_lines=partial, orphans_set_aside=moved, p_reference_rows_sha256=p_record["validation_rows_sha256"],
                  started_utc=R._now(), timings=timings)
    print(json.dumps(header), flush=True)
    if args.device == "cuda:0":
        torch.cuda.reset_peak_memory_stats(0)
    finished = latest is not None and (start == total or R.stopped_training(latest[1]))
    with segment.open("x") as log:
        log.write(json.dumps(header, allow_nan=False) + "\n")
        log.flush()
        os.fsync(log.fileno())
        if not finished and args.stop_after != start:
            segment_start = time.perf_counter()
            update = start
            while update < total:
                before = time.perf_counter()
                record, state = R.round_step(model, optimizer, recipe=args.recipe, supervised=supervised,
                                             rows=data.train, state=state,
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
                                         evaluator=evaluator, reference=reference, enforce_stop=not args.no_hard_stop)
                    print(json.dumps(dict(event="checkpoint", update=update, delta=sidecar["evaluation"]["delta"],
                                          validation=sidecar["evaluation"]["validation"],
                                          timings=sidecar["timings"])), flush=True)
                    if sidecar["training_stopped"] or update == args.stop_after:
                        break
    if done(output, total):
        return finalize(output, args, total=total, data=data, provenance=provenance)
    print(json.dumps(dict(event="stopped-as-requested", update=args.stop_after)), flush=True)
    return 0


def done(output, total):
    latest, _ = R.resume_point(output)
    return latest is not None and (latest[0] == total or R.stopped_training(latest[1]))


def write_pair(output, update, *, model, optimizer, state, provenance, recipe, records, log, evaluator, reference,
               enforce_stop=True):
    """Evaluate, make the metrics durable, then publish the checkpoint and after it the sidecar."""
    official, rows, hold, measure_timings = evaluator.measure(model)
    delta, paired = evaluator.compare(official, rows, hold, reference)
    log.flush()
    os.fsync(log.fileno())
    path = output / R.checkpoint_name(update)
    started = time.perf_counter()
    save_checkpoint(path, model=model, optimizer=optimizer, checkpoint_id=f"r1-{recipe}-update-{update:06d}",
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
                                   holdout=evaluator.holdout_summary(hold), holdout_rows_sha256=R.rows_digest(hold)),
                   delta_total=delta["total_loss"], flag=bool(delta["total_loss"] > R.FLAG_DELTA),
                   # hard_stop records the breach; training_stopped whether training ended here.
                   hard_stop=bool(delta["total_loss"] > R.STOP_DELTA),
                   training_stopped=bool(enforce_stop and delta["total_loss"] > R.STOP_DELTA),
                   sampler=dict(selfplay_epoch=state.selfplay.epoch, selfplay_cursor=state.selfplay.cursor,
                                selfplay_samples_drawn=state.selfplay.samples_seen),
                   timings=dict(measure_timings, checkpoint_save_seconds=save_seconds), written_utc=R._now())
    R.publish_json(output / R.sidecar_name(update), sidecar)
    return sidecar


def finalize(output, args, *, total, data, provenance):
    if (output / "summary.json").exists():
        return 0
    latest, _ = R.resume_point(output)
    check(latest is not None, "no checkpoint to summarise")
    sidecars = [R.read_json(output / R.sidecar_name(u)) for u in
                sorted(int(p.name[7:13]) for p in output.glob("update-*.json"))]
    evaluations = [dict(update=s["update"], checkpoint=s["checkpoint"], delta_total=s["delta_total"],
                        flag=s["flag"], hard_stop=s["hard_stop"], paired_labels=s["evaluation"]["paired_labels"])
                   for s in sidecars]
    stopped = R.stopped_training(latest[1])
    completed = latest[0] == total and not stopped
    check(completed or stopped, "run is neither complete nor stopped")
    original = R.select_candidate(evaluations, completed=completed, flag=R.FLAG_DELTA, stop=R.STOP_DELTA)
    enforced = not args.no_hard_stop
    choice = original if enforced else R.exploratory_candidate(evaluations, completed=completed,
                                                                flag=R.FLAG_DELTA, stop=R.STOP_DELTA)
    records, superseded, partial = R.read_metrics(output)
    updates = len(records)
    n = len(data.train)
    details = provenance["recipe_details"]
    per_update = details["trained_positions_per_update"]
    drawn = updates * R.SETTINGS.selfplay_batch_size
    seconds = [r["update_seconds"] for r in records]
    candidate = choice["candidate"]
    summary = dict(version=R.RUN_VERSION, recipe=provenance["recipe"], name=details["name"], completed=completed,
        updates=updates, total_updates=total, hard_stop_enforced=enforced, exploratory=not enforced,
        selection=choice, original_rule_selection=original,
        threshold_breaches=[e["update"] for e in evaluations if e["hard_stop"]],
        candidate=dict(checkpoint=candidate, sha256=None if candidate == "P" else sha256(output / candidate),
                       update=choice["candidate_update"]),
        final_delta_total=latest[1]["delta_total"], final_flag_over_0_05=bool(latest[1]["delta_total"] > R.FLAG_DELTA),
        final_paired_labels=latest[1]["evaluation"]["paired_labels"],
        evaluations=evaluations,
        trained_positions=dict(selfplay=updates * per_update["selfplay"], stockfish=updates * per_update["stockfish"]),
        discarded_sampler_indices=updates * details["discarded_sampler_indices_per_update"],
        selfplay=dict(training_positions=n, drawn=drawn, seen_twice=max(0, drawn - n),
                      epochs_started=records[-1]["selfplay_epoch"] if records else 0),
        exclusions=data.manifest["counts"], superseded_metric_lines=superseded, partial_metric_lines=partial,
        update_seconds=dict(median=float(np.median(seconds)), mean=float(np.mean(seconds)),
                            p90=float(np.quantile(seconds, 0.9))) if seconds else None,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0) if args.device == "cuda:0" else None,
        test_split_used=False)
    R.publish_json(output / "summary.json", summary)
    print(json.dumps(dict(event="summary", exploratory=not enforced, selection=choice, original_rule_selection=original,
                          final_delta_total=summary["final_delta_total"])), flush=True)
    return 0


# compare, inference, timings

def tree_differences(a, b, path, out, limit=50):
    if len(out) >= limit:
        return out
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and a.device == b.device and torch.equal(a, b))
        if not same:
            out.append(path)
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path} keys {sorted(set(a) ^ set(b))}")
        for key in sorted(set(a) & set(b), key=str):
            tree_differences(a[key], b[key], f"{path}.{key}", out, limit)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            out.append(f"{path} length/type")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                tree_differences(x, y, f"{path}[{i}]", out, limit)
    elif type(a) is not type(b) or a != b:
        out.append(path)
    return out


def command_compare(args):
    a, b, update = Path(args.a), Path(args.b), args.update
    name = R.checkpoint_name(update)
    pa = torch.load(a / name, map_location="cpu", weights_only=True)
    pb = torch.load(b / name, map_location="cpu", weights_only=True)
    report = {}
    for part, path in (("model weights", ("model_state",)), ("optimizer", ("training", "optimizer")),
                       ("sampler", ("training", "mixed_state")), ("rng", ("training", "rng")),
                       ("progress", ("training", "progress")), ("everything else", ())):
        x, y = pa, pb
        for key in path:
            x, y = x[key], y[key]
        if not path:
            skip = ("model_state", "training") + (("provenance",) if args.state_only else ())
            x = {k: v for k, v in pa.items() if k not in skip}
            y = {k: v for k, v in pb.items() if k not in skip}
            x["training"] = {k: v for k, v in pa["training"].items() if k not in ("optimizer", "mixed_state", "rng", "progress")}
            y["training"] = {k: v for k, v in pb["training"].items() if k not in ("optimizer", "mixed_state", "rng", "progress")}
        report[part] = tree_differences(x, y, part, [])
    ra = [R.strip_timing(r) for r in R.read_metrics(a)[0][:update]]
    rb = [R.strip_timing(r) for r in R.read_metrics(b)[0][:update]]
    check(len(ra) == len(rb) == update, "both runs need metrics for every update")
    report["training metrics"] = tree_differences(ra, rb, "metrics", [])
    ignore = {"metrics_sha256", "checkpoint_sha256", "checkpoint_bytes"}
    sa = {k: v for k, v in R.strip_timing(R.read_json(a / R.sidecar_name(update))).items() if k not in ignore}
    sb = {k: v for k, v in R.strip_timing(R.read_json(b / R.sidecar_name(update))).items() if k not in ignore}
    if args.state_only:
        # The stop setting differs by design; compare what was computed, not how the run is labelled.
        keep = ("update", "recipe", "checkpoint", "metrics_records", "evaluation", "delta_total", "flag", "sampler")
        sa, sb = ({k: v.get(k) for k in keep} for v in (sa, sb))
    report["sidecar evaluation"] = tree_differences(sa, sb, "sidecar", [])
    identical = not any(report.values())
    extra = {}
    if args.state_only:
        extra["provenance_differences_not_compared"] = tree_differences(pa["provenance"], pb["provenance"], "provenance", [])
    print(json.dumps(dict(event="compare", a=str(a), b=str(b), update=update, identical=identical,
                          state_only=args.state_only,
                          checkpoint_files_byte_identical=sha256(a / name) == sha256(b / name),
                          differences=report, **extra)), flush=True)
    return 0 if identical else 1


def command_inference(args):
    configure(args.device, threads=args.threads)
    paths = data_paths(args.data_config)
    validation = CachedSupervisedDataset.open(split="validation", **paths)
    started = time.perf_counter()
    loaded = load_inference(args.checkpoint, device=args.device)
    load_seconds = time.perf_counter() - started
    measured = validate(loaded.model, validation, args.eval_batch_size)
    sidecar = R.read_json(Path(args.checkpoint).with_suffix(".json"))
    recorded = sidecar["evaluation"]["validation"]
    same = measured == recorded
    print(json.dumps(dict(event="inference", checkpoint=str(args.checkpoint), checkpoint_id=loaded.checkpoint_id,
                          recipe=loaded.provenance["recipe"], load_seconds=load_seconds,
                          validation=measured, recorded=recorded, identical=same)), flush=True)
    return 0 if same else 1


def command_timings(args):
    report = {}
    if args.round_data:
        report["round_data"] = R.read_json(Path(args.round_data) / "manifest.json")["timings"]
    for run in args.runs:
        run = Path(run)
        records, _, _ = R.read_metrics(run)
        headers = [json.loads(p.read_text().split("\n", 1)[0]) for p in R.segments(run)]
        sidecars = [R.read_json(p) for p in sorted(run.glob("update-*.json"))]
        seconds = [r["update_seconds"] for r in records]
        report[str(run)] = dict(
            segments=[dict(segment=h["segment"], from_update=h["from_update"], timings=h["timings"]) for h in headers],
            update_seconds=dict(count=len(seconds), median=float(np.median(seconds)), max=float(np.max(seconds)),
                                first=seconds[0]) if seconds else None,
            checkpoints=[dict(update=s["update"], bytes=s["checkpoint_bytes"], **s["timings"]) for s in sidecars])
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def disk(command):
        command.add_argument("--quota-bytes", type=int, default=100_000_000_000,
                             help="quota for the du fallback when `quota -w` does not parse")
        command.add_argument("--skip-disk-check", action="store_true", help="tests and machines without a quota only")

    data = commands.add_parser("data", help="build or verify the shared self-play arrays")
    data.add_argument("--pointer", type=Path, required=True)
    data.add_argument("--data-config", type=Path, required=True)
    data.add_argument("--p-checkpoint", type=Path, required=True)
    data.add_argument("--p-sha256", required=True)
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--expected-branch", required=True)
    data.add_argument("--expected-games", type=int, required=True)
    data.add_argument("--expected-positions", type=int, required=True)
    data.add_argument("--extra-checkpoints", type=int, default=0, help="checkpoints beyond the two full runs (dry runs)")
    disk(data)
    train = commands.add_parser("train", help="run or resume one recipe")
    train.add_argument("--recipe", choices=R.RECIPES, required=True)
    train.add_argument("--data-config", type=Path, required=True)
    train.add_argument("--round-data", type=Path, required=True)
    train.add_argument("--p-checkpoint", type=Path, required=True)
    train.add_argument("--p-sha256", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    train.add_argument("--threads", type=int, default=1)
    train.add_argument("--updates", type=int, help="total updates (default ceil(training positions / 24)); dry runs only")
    train.add_argument("--passes", type=int, help="amended runs only: ceil(passes x training positions / 24) updates "
                       "(needs --amendment-note)")
    train.add_argument("--checkpoint-every", type=int, default=R.CHECKPOINT_EVERY)
    train.add_argument("--stop-after", type=int, help="exit after the checkpoint at this update (resume tests)")
    train.add_argument("--no-hard-stop", action="store_true",
                       help="exploratory runs only: record stop-threshold breaches but keep training (needs --amendment-note)")
    train.add_argument("--amendment-note", help="the approved amendment behind --no-hard-stop or --passes, recorded in the provenance")
    train.add_argument("--eval-batch-size", type=int, default=32)
    train.add_argument("--expect-p-total", type=float, help="P's recorded validation total loss to reproduce")
    train.add_argument("--p-tolerance", type=float, default=0.001)
    disk(train)
    compare = commands.add_parser("compare", help="exact comparison of two runs at one update")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--update", type=int, required=True)
    compare.add_argument("--state-only", action="store_true",
                         help="compare training state, metrics and evaluation but not provenance or run labels")
    inference = commands.add_parser("inference", help="load a checkpoint for inference and recheck its validation")
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("--data-config", type=Path, required=True)
    inference.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    inference.add_argument("--threads", type=int, default=1)
    inference.add_argument("--eval-batch-size", type=int, default=32)
    baseline = commands.add_parser("baseline", help="measure P only (no updates) and print its report for review")
    baseline.add_argument("--data-config", type=Path, required=True)
    baseline.add_argument("--round-data", type=Path, required=True)
    baseline.add_argument("--p-checkpoint", type=Path, required=True)
    baseline.add_argument("--p-sha256", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    baseline.add_argument("--threads", type=int, default=1)
    baseline.add_argument("--eval-batch-size", type=int, default=32)
    baseline.add_argument("--expect-p-total", type=float)
    baseline.add_argument("--p-tolerance", type=float, default=0.001)
    timings = commands.add_parser("timings", help="report measured timings")
    timings.add_argument("runs", nargs="*", type=Path)
    timings.add_argument("--round-data", type=Path)
    args = parser.parse_args(argv)
    if args.command == "train" and (args.checkpoint_every < 1 or (args.updates is not None and args.updates < 1)):
        parser.error("--checkpoint-every and --updates must be positive")
    if args.command == "train" and (args.no_hard_stop or args.passes is not None) and not args.amendment_note:
        parser.error("--no-hard-stop and --passes need --amendment-note")
    if args.command == "train" and args.amendment_note and not (args.no_hard_stop or args.passes is not None):
        parser.error("--amendment-note records the amendment behind --no-hard-stop or --passes; give one of them")
    if args.command == "train" and args.passes is not None and (args.passes < 1 or args.updates is not None):
        parser.error("--passes must be positive and cannot be combined with --updates")
    handler = dict(data=command_data, baseline=command_baseline, train=command_train, compare=command_compare,
                   inference=command_inference, timings=command_timings)[args.command]
    try:
        return handler(args)
    except Mismatch as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
