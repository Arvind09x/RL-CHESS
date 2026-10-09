"""Bounded T02 real D05/D07 CPU continuation in a new Python process.

One warmup update creates Adam state; one next update verifies restoration.
No fitting loop, held-out batch, T03/T04 evaluator, pilot training or RL.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from chess_rl.supervised import open_dataset
from chess_rl.tensor_ops import legal_policy_probs
from chess_rl.training.checkpoints import (DataOrder, ModelConfig, Progress, load_inference,
    restore_training, runtime_config, save_checkpoint)
from chess_rl.training.trainer import LossWeights, supervised_step


def identity(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest_tree(value):
    """Hash every tensor's exact bytes, dtype and shape, without a tolerance."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict): return {str(k): digest_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [digest_tree(v) for v in value]
    return value


def random_state():
    return digest_tree(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))


def predictions(model, batch):
    prior_mode = model.training
    model.eval()
    with torch.inference_mode():
        tensors = batch.to_torch()
        p, w = model(tensors["features"])
        result = digest_tree(dict(policy_logits=p, wdl_logits=w,
            legal_priors=legal_policy_probs(p, tensors["legal_mask"]), wdl_probabilities=w.softmax(-1)))
    model.train(prior_mode)
    return result


def provenance():
    files = {n: identity(Path(n)) for n in (
        "data/manifests/elite-2025-01-d04-v1.json", "runs/d01/stockfish-provenance.json",
        "runs/d05/tiny-final/labels.jsonl", "runs/d05/tiny-final/settings.json", "runs/d05/tiny-final/summary.json",
        "/private/tmp/rl-chess-d03-prepared/games.jsonl", "/private/tmp/rl-chess-d03-prepared/split-audit.json",
        "/private/tmp/rl-chess-d04-prepared/samples.jsonl", "/private/tmp/rl-chess-d04-prepared/overlap-audit.json")}
    return dict(task="T02", reviewed_base="b26cf38f04a09fd04283023795c0752753ce3106",
        seed=20261009, dataset_split="train", files=files, sampler="NumPy global permutation; D07 explicit order",
        schedule="one warmup and one continuation update; no scheduler or accumulation")


def next_batch(dataset, order):
    for index, batch in enumerate(dataset.batches(order.batch_size, order=order.order, drop_last=order.drop_last)):
        if index == order.cursor // order.batch_size:
            assert batch.references == tuple(dataset[i].reference for i in order.next_indices())
            return batch
    raise AssertionError("next real batch required")


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = open_dataset(bundle=Path("runs/d05/tiny-final"),
        manifest_path=Path("data/manifests/elite-2025-01-d04-v1.json"),
        games_path=Path("/private/tmp/rl-chess-d03-prepared/games.jsonl"),
        split_audit_path=Path("/private/tmp/rl-chess-d03-prepared/split-audit.json"),
        prepared=Path("/private/tmp/rl-chess-d04-prepared"),
        teacher_evidence=Path("runs/d01/stockfish-provenance.json"), split="train")
    config = ModelConfig()  # accepted combined model, 0.1 dropout, widths 128
    source = identity(Path("runs/d05/tiny-final/labels.jsonl"))["sha256"]
    prov = provenance()
    checkpoint = args.directory / "step-1.pt"
    if args.worker == "uninterrupted":
        random.seed(20261009); np.random.seed(20261009); torch.manual_seed(20261009)
        model = config.build()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, betas=(.9, .999), eps=1e-8,
            weight_decay=0, amsgrad=False, foreach=False, fused=False)
        weights = LossWeights(.7, 1.3)
        order = DataOrder(source, DataOrder.reference_hash(dataset),
            tuple(int(i) for i in np.random.permutation(len(dataset))), 0, 2, False, 0)
        first_batch = next_batch(dataset, order)
        warmup = supervised_step(model, optimizer, first_batch, weights=weights)
        order = replace(order, cursor=2)
        # Cache a normal in both non-Torch streams to test full RNG state.
        random.gauss(0, 1); np.random.normal()
        before = random_state()
        save_checkpoint(checkpoint, model=model, optimizer=optimizer, checkpoint_id="T02-correctness-step-1",
            provenance=prov, weights=weights, progress=Progress(1, 0, 2), data_order=order)
        assert before == random_state()
        (args.directory / "warmup.json").write_text(json.dumps(dict(metrics=asdict(warmup), references=first_batch.references,
            config=asdict(config), weights=asdict(weights), progress=asdict(Progress(1, 0, 2)),
            data_order=asdict(order), provenance=prov, runtime=runtime_config(), checkpoint=identity(checkpoint)), indent=2) + "\n")
    else:
        # Deliberately scramble all streams and initialize an unrelated model.
        random.seed(999); np.random.seed(998); torch.manual_seed(997)
        _ = config.build()
        restored = restore_training(checkpoint, expected_config=config, expected_provenance=prov,
            dataset=dataset, source_id=source, expected_weights=LossWeights(.7, 1.3))
        assert restored.progress == Progress(1, 0, 2)
        model, optimizer, weights, order = restored.model, restored.optimizer, restored.weights, restored.data_order
    batch = next_batch(dataset, order)
    before_reload = random_state()
    loaded = load_inference(checkpoint, expected_config=config)
    assert before_reload == random_state()  # inference does not install or consume RNG
    before_predictions = predictions(model, batch)
    assert before_predictions == predictions(loaded.model, batch)
    assert before_reload == random_state()
    assert all(layer.self_attn.gab.bank is model.backbone.bank for layer in model.backbone.layers)
    assert sum(p is model.backbone.bank.weight for g in optimizer.param_groups for p in g["params"]) == 1
    before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    draws = [random.random(), random.gauss(0, 1), float(np.random.random()), float(np.random.normal())]
    captured = []
    def capture(module, inputs, outputs): captured.extend(digest_tree(outputs))
    hook = model.register_forward_hook(capture)
    metrics = supervised_step(model, optimizer, batch, weights=weights)
    hook.remove()
    assert len(optimizer.state) == 215 and all(s["step"].item() == 2 for s in optimizer.state.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.count_nonzero(p.grad) > 0 for n, p in model.named_parameters()
               if ".relative." in n or n == "backbone.bank.weight")
    report = dict(runtime=runtime_config(), config=asdict(config), references=batch.references, indices=order.next_indices(),
        batch=digest_tree({n: getattr(batch, n) for n in ("features", "legal_mask", "policy_classes", "wdl")}),
        draws=draws, before_predictions=before_predictions, training_forward=captured, metrics=asdict(metrics),
        parameters=digest_tree(dict(model.named_parameters())),
        gradients=digest_tree({n: p.grad for n, p in model.named_parameters()}),
        optimizer=digest_tree(optimizer.state_dict()), optimizer_defaults=digest_tree(optimizer.defaults),
        changed_parameter_tensors=sum(not torch.equal(p, before_parameters[n]) for n, p in model.named_parameters()),
        random_state_after_update=random_state(), post_update_predictions=predictions(model, batch), shared_bank_identity=True)
    (args.directory / f"{args.worker}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--worker", choices=("uninterrupted", "resumed"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.directory.mkdir(parents=True, exist_ok=False)
    for kind in ("uninterrupted", "resumed"):
        subprocess.run([sys.executable, "-B", __file__, "--directory", str(args.directory), "--worker", kind], check=True)
    a = json.loads((args.directory / "uninterrupted.json").read_text())
    b = json.loads((args.directory / "resumed.json").read_text())
    assert a == b, "real CPU continuation diverged"
    summary = dict(exact_match=True, separate_python_process=True, tolerance=0,
        checkpoint=identity(args.directory / "step-1.pt"),
        compared_fields=list(a), parameter_tensors=len(a["parameters"]), gradient_tensors=len(a["gradients"]),
        optimizer_states=len(a["optimizer"]["state"]), changed_parameter_tensors=a["changed_parameter_tensors"],
        metrics=a["metrics"], dropout=a["config"]["dropout"], runtime=a["runtime"],
        outputs={n: identity(args.directory / n) for n in ("warmup.json", "uninterrupted.json", "resumed.json")},
        limitations="local pinned CPU float32 correctness only; two updates total; no T03/T04, fit, P, pilot, RL or accelerator claim")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
