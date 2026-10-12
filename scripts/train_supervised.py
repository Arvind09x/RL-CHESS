"""Portable supervised pilot runner; test split is never opened.

Outputs are fresh immutable checkpoints, JSONL metrics and CSV/SVG learning curves. This command
does not designate a checkpoint as project P or start self-play.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chess_rl.runtime import configure, data_paths
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (DataOrder, ModelConfig, Progress,
    restore_training, runtime_config, save_checkpoint)
from chess_rl.training.losses import policy_class_cross_entropy, soft_wdl_cross_entropy
from chess_rl.training.trainer import LossWeights, supervised_step
from chess_rl.training.curves import write_learning_curves


def gradient_norm(model):
    """Global L2 norm of the last backward gradients; shared parameters count once.

    Adam leaves these gradients unchanged, so this is the norm before its update.
    No clipping or other gradient modification is applied.
    """
    gradients = [p.grad for p in model.parameters()]
    if not gradients or any(g is None for g in gradients):
        raise ValueError("gradient norm requires every model gradient")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for g in gradients]))
    if not torch.isfinite(norm):
        raise ValueError("gradient norm must be finite")
    return float(norm)


def validate(model, data, batch_size):
    mode = model.training
    model.eval()
    sums = np.zeros(3)
    count = 0
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for batch in data.batches(batch_size):
                tensors = batch.to_torch(device=device)
                policy, wdl = model(tensors["features"])
                n = len(batch.references)
                sums += [n * float(policy_class_cross_entropy(policy, tensors["legal_mask"], tensors["policy_classes"])),
                         n * float(soft_wdl_cross_entropy(wdl, tensors["wdl"])),
                         float((policy.masked_fill(~tensors["legal_mask"], -torch.inf).argmax(1) == tensors["policy_classes"]).sum())]
                count += n
    finally:
        model.train(mode)
    if not count:
        raise ValueError("validation split is empty")
    return dict(policy_loss=sums[0] / count, wdl_loss=sums[1] / count,
                policy_accuracy=sums[2] / count, positions=count,
                total_loss=(sums[0] + sums[1]) / count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--steps", type=int, required=True, help="target total update count, including resumed progress")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=1, help="positive Torch intraop thread count; recorded and checked on resume")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--validation-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.validation_every, args.threads) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("steps/batch-size/validation-every/threads and learning-rate must be positive")
    if args.output.exists():
        p.error("output must be a fresh directory; resume into a new directory")
    configure(args.device, threads=args.threads)
    paths = data_paths(args.data_config)
    train = open_dataset(**paths, split="train").freeze()
    validation = open_dataset(**paths, split="validation").freeze()
    source = hashlib.sha256(json.dumps(train.source_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = ModelConfig(device=args.device)
    provenance = dict(purpose="supervised pilot; P selection remains a separate decision",
        source_id=source, seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate)
    weights = LossWeights()
    if args.resume:
        loaded = restore_training(args.resume, expected_config=config, expected_provenance=provenance,
                                  dataset=train, source_id=source, expected_weights=weights)
        model, optimizer, order, progress = loaded.model, loaded.optimizer, loaded.data_order, loaded.progress
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = config.build()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, foreach=False, fused=False)
        order = DataOrder(source, DataOrder.reference_hash(train), tuple(map(int, np.random.permutation(len(train)))),
                          0, args.batch_size, False, 0)
        progress = Progress(0, 0, 0)
    if progress.step >= args.steps:
        p.error("target steps must exceed restored progress")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(dict(model=asdict(config), provenance=provenance,
        runtime=runtime_config(), target_steps=args.steps, validation_every=args.validation_every,
        training_positions=len(train), validation_positions=len(validation),
        resume_checkpoint=str(args.resume.resolve()) if args.resume else None), indent=2) + "\n")
    start = time.perf_counter()
    starting_samples = progress.samples_seen
    best = None
    last_checkpoint = None
    with (args.output / "metrics.jsonl").open("x") as log:
        baseline_checkpoint = args.output / f"step-{progress.step:06d}.pt"
        baseline = dict(step=progress.step, epoch=progress.epoch,
            event="resumed_baseline" if args.resume else "untrained_baseline",
            validation=validate(model, validation, args.batch_size),
            checkpoint=baseline_checkpoint.name, elapsed_seconds=time.perf_counter() - start)
        save_checkpoint(baseline_checkpoint, model=model, optimizer=optimizer,
            checkpoint_id=f"pilot-step-{progress.step}", provenance=provenance,
            weights=weights, progress=progress, data_order=order)
        (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        log.write(json.dumps(baseline, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps(baseline), flush=True)
        while progress.step < args.steps:
            if not order.next_indices():
                order = replace(order, order=tuple(map(int, np.random.permutation(len(train)))), cursor=0, epoch=order.epoch + 1)
            indices = order.next_indices()
            metrics = supervised_step(model, optimizer, train.batch(indices), weights=weights)
            order = replace(order, cursor=order.cursor + len(indices))
            progress = Progress(progress.step + 1, order.epoch, progress.samples_seen + len(indices))
            record = dict(step=progress.step, epoch=progress.epoch, event="update",
                train=asdict(metrics) | {"gradient_norm": gradient_norm(model)}, elapsed_seconds=time.perf_counter() - start)
            if progress.step % args.validation_every == 0 or progress.step == args.steps:
                record["validation"] = validate(model, validation, args.batch_size)
                last_checkpoint = args.output / f"step-{progress.step:06d}.pt"
                save_checkpoint(last_checkpoint, model=model, optimizer=optimizer, checkpoint_id=f"pilot-step-{progress.step}",
                    provenance=provenance, weights=weights, progress=progress, data_order=order)
                if best is None or record["validation"]["total_loss"] < best["validation"]["total_loss"]:
                    best = dict(checkpoint=last_checkpoint.name, validation=record["validation"], step=progress.step)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if "validation" in record:
                print(json.dumps(record), flush=True)
    elapsed = time.perf_counter() - start
    write_learning_curves(args.output)
    segment_samples = progress.samples_seen - starting_samples
    summary = dict(completed_steps=progress.step, final_checkpoint=last_checkpoint.name,
        best_validation_in_this_run=best, elapsed_seconds=elapsed,
        segment_training_samples=segment_samples, training_samples_per_second=segment_samples / elapsed,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0) if args.device == "cuda:0" else None,
        baseline="baseline.json", learning_curves=["metrics.jsonl", "learning-curves.csv", "learning-curves.svg"],
        test_split_used=False, project_P_selected=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
