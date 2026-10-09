"""Bounded single-device correctness: warmup, save/reload, one next update twice.

The continuation runs in a fresh process. No fitting loop or held-out training.
Uses a sealed synthetic D07 fixture by default; --data-config opens unchanged
frozen artifacts using portable relative paths, always with split='train'.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from chess_rl.adapter import Evaluation, TerminalResult, legal_mask
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (DataOrder, ModelConfig, Progress,
    _cpu_tree, _rng_state, cuda_runtime_config, load_inference, restore_training,
    runtime_config, save_checkpoint)
from chess_rl.training.trainer import LossWeights, supervised_step
from tests.test_real_evaluator import nonterminal_cases, terminal_cases, preserved
from tests.test_supervised import fixture_bundle

# Frozen before execution. No CPU versus CUDA agreement is asserted.
TOLERANCES = dict(batch=dict(atol=2e-6, rtol=2e-5),
    oracle=dict(atol=1e-6, rtol=1e-5), continuation=dict(atol=1e-6, rtol=1e-5),
    reload=dict(atol=0, rtol=0), probability_sum=dict(atol=1e-6, rtol=0))
SEED = 20261010


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def configure(device):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if device == "cuda:0":
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def rng(device):
    state = _rng_state()
    if device == "cuda:0": state["cuda"] = torch.cuda.get_rng_state(0)
    return state


def compare(a, b, tolerance, maxima):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.is_floating_point():
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            maxima.append(float((a - b).abs().max()) if a.numel() else 0.)
            torch.testing.assert_close(a, b, **tolerance)
        else: assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: compare(a[k], b[k], tolerance, maxima)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y, tolerance, maxima)
    elif type(a) is float:
        assert type(b) is float and np.isfinite(a) and np.isfinite(b)
        maxima.append(abs(a - b))
        np.testing.assert_allclose(a, b, **tolerance)
    else: assert a == b


def dataset(args):
    if args.data_config:
        config_path = args.data_config.resolve()
        config = json.loads(config_path.read_text())
        assert config["version"] == "cuda-portable-data-v1"
        paths = {k: (config_path.parent / v).resolve() for k, v in config["paths"].items()}
    else:
        root = args.directory / "synthetic"
        if args.worker == "uninterrupted":
            root.mkdir()
            paths = fixture_bundle(root)
        else:
            root = root.resolve()
            paths = dict(bundle=root/"bundle", manifest_path=root/"manifest.json",
                games_path=root/"games.jsonl", split_audit_path=root/"split.json",
                prepared=root/"prepared",
                teacher_evidence=Path(__file__).resolve().parents[1]/"runs/d01/stockfish-provenance.json")
    data = open_dataset(**paths, split="train")
    return data, identity(paths["bundle"] / "labels.jsonl")["sha256"]


def batch_for(data, order):
    batches = data.batches(order.batch_size, order=order.order, drop_last=order.drop_last)
    for i, batch in enumerate(batches):
        if i == order.cursor // order.batch_size:
            assert batch.references == tuple(data[j].reference for j in order.next_indices())
            return batch
    raise AssertionError("next batch required")


def predictions(model, batch):
    mode = model.training
    model.eval()
    with torch.inference_mode():
        outputs = _cpu_tree(model(batch.to_torch(device=next(model.parameters()).device)["features"]))
    model.train(mode)
    return outputs


def evaluation(path, device):
    before = rng(device)
    evaluator = RealEvaluator(path, device=device)
    states = nonterminal_cases() + [c[0] for c in terminal_cases()] + [nonterminal_cases()[0]]
    history = [preserved(s) for s in states]
    calls = []
    hook = evaluator.checkpoint.model.register_forward_hook(
        lambda m, inputs, outputs: calls.append((list(inputs[0].shape), str(inputs[0].device))))
    results = evaluator(states)
    assert calls == [([14, 64, 128], device)]
    count = len(calls)
    assert evaluator([]) == []
    evaluator([c[0] for c in terminal_cases()])
    assert len(calls) == count
    hook.remove()
    maximum = 0.
    for state, result in zip(states, results):
        if isinstance(result, TerminalResult): continue
        assert isinstance(result, Evaluation)
        single, = evaluator([state])
        for name in ("priors", "wdl", "value"):
            np.testing.assert_allclose(getattr(result, name), getattr(single, name), **TOLERANCES["batch"])
            maximum = max(maximum, float(np.max(np.abs(np.asarray(getattr(result, name)) - getattr(single, name)))))
        mask = legal_mask(state)
        assert result.priors.dtype == result.wdl.dtype == np.float32
        assert result.priors.flags.owndata and result.wdl.flags.owndata
        assert not result.priors[~mask].any()
        np.testing.assert_allclose(result.priors.sum(), 1, **TOLERANCES["probability_sum"])
        np.testing.assert_allclose(result.wdl.sum(), 1, **TOLERANCES["probability_sum"])
        assert result.value == float(result.wdl[0] - result.wdl[2])
        # Independent float64 CPU normalization, retaining root WDL orientation.
        from chess_rl.adapter import encode_position
        with torch.inference_mode():
            p, w = evaluator.checkpoint.model(torch.tensor(encode_position(state).features[None], device=device))
        p, w = p[0].cpu().numpy().astype(np.float64), w[0].cpu().numpy().astype(np.float64)
        oracle = np.zeros(4352)
        legal = np.exp(p[mask] - p[mask].max()); oracle[mask] = legal / legal.sum()
        w = np.exp(w - w.max()); w /= w.sum()
        np.testing.assert_allclose(result.priors, oracle, **TOLERANCES["oracle"])
        np.testing.assert_allclose(result.wdl, w, **TOLERANCES["oracle"])
    assert [preserved(s) for s in states] == history
    a, b = evaluator([states[0], states[0]])
    assert not np.shares_memory(a.priors, b.priors) and not np.shares_memory(a.wdl, b.wdl)
    compare(before, rng(device), dict(atol=0, rtol=0), [])
    return dict(batch_max_abs_error=maximum, inputs=len(states), forward_device=device,
        terminal_bypass=True, owned_arrays=True, input_and_rng_preserved=True)


def worker(args):
    configure(args.device)
    data, source = dataset(args)
    config = ModelConfig(device=args.device)  # full accepted combined architecture
    provenance = dict(purpose="bounded CUDA backend preparation", seed=SEED, source_sha256=source)
    checkpoint = args.directory / "step-1.pt"
    if args.worker == "uninterrupted":
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model = config.build()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        weights = LossWeights(.7, 1.3)
        order = DataOrder(source, DataOrder.reference_hash(data), tuple(range(len(data))), 0, 2, False, 0)
        supervised_step(model, optimizer, batch_for(data, order), weights=weights)
        order = replace(order, cursor=2)
        before = rng(args.device)
        save_checkpoint(checkpoint, model=model, optimizer=optimizer, checkpoint_id="CUDA-preparation-step-1",
            provenance=provenance, weights=weights, progress=Progress(1, 0, 2), data_order=order)
        compare(before, rng(args.device), TOLERANCES["reload"], [])
    else:
        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        loaded = restore_training(checkpoint, expected_config=config, expected_provenance=provenance,
            dataset=data, source_id=source, expected_weights=LossWeights(.7, 1.3))
        assert loaded.progress == Progress(1, 0, 2)
        model, optimizer, weights, order = loaded.model, loaded.optimizer, loaded.weights, loaded.data_order
    assert model.training
    for p, state in optimizer.state.items():
        assert state["step"].device.type == "cpu"
        assert all(v.device == p.device for n, v in state.items() if n != "step")
    batch = batch_for(data, order)
    before = rng(args.device)
    inference = load_inference(checkpoint, device=args.device)
    compare(predictions(model, batch), predictions(inference.model, batch), TOLERANCES["reload"], [])
    compare(before, rng(args.device), TOLERANCES["reload"], [])
    evaluation_report = evaluation(checkpoint, args.device)
    if args.checkpoint: evaluation_report["accepted_cpu_checkpoint"] = evaluation(args.checkpoint, args.device)
    assert all(l.self_attn.gab.bank is model.backbone.bank for l in model.backbone.layers)
    assert sum(p is model.backbone.bank.weight for g in optimizer.param_groups for p in g["params"]) == 1
    captured = []
    hook = model.register_forward_hook(lambda m, inputs, outputs: captured.append(_cpu_tree(outputs)))
    metrics = supervised_step(model, optimizer, batch, weights=weights)
    hook.remove()
    assert len(optimizer.state) == 215 and all(v["step"].item() == 2 for v in optimizer.state.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = _cpu_tree(dict(parameters=model.state_dict(),
        gradients={n:p.grad for n,p in model.named_parameters()}, optimizer=optimizer.state_dict(),
        forward=captured, metrics=asdict(metrics), rng=rng(args.device),
        predictions=predictions(model, batch), references=batch.references, indices=order.next_indices()))
    torch.save(report, args.directory / f"{args.worker}.pt")
    (args.directory / f"{args.worker}.json").write_text(json.dumps(dict(runtime=runtime_config(),
        cuda_runtime=cuda_runtime_config() if args.device == "cuda:0" else None,
        config=asdict(config), metrics=asdict(metrics), evaluation=evaluation_report), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--data-config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--worker", choices=("uninterrupted", "resumed"))
    a = p.parse_args()
    if a.worker: return worker(a)
    a.directory.mkdir(parents=True, exist_ok=False)
    (a.directory / "config.json").write_text(json.dumps(dict(tolerances=TOLERANCES,
        seed=SEED, device=a.device, updates="one warmup; one continuation in each of two processes",
        dtype="float32", dropout=.1, amsgrad=False, foreach=False, fused=False), indent=2) + "\n")
    for kind in ("uninterrupted", "resumed"):
        subprocess.run([sys.executable, "-B", __file__, *sys.argv[1:], "--worker", kind], check=True)
    x, y = [torch.load(a.directory / f"{k}.pt", weights_only=True) for k in ("uninterrupted", "resumed")]
    maximum = []
    compare(x["rng"], y["rng"], TOLERANCES["reload"], [])
    # Same runtime/device only. Integer state, membership and RNG remain exact.
    tolerance = TOLERANCES["reload"] if a.device == "cpu" else TOLERANCES["continuation"]
    compare(x, y, tolerance, maximum)
    summary = dict(passed=True, device=a.device, separate_process_continuation=True,
        continuation_max_abs_error=max(maximum), tolerances=TOLERANCES,
        checkpoint=identity(a.directory/"step-1.pt"),
        outputs={path.name:identity(path) for path in a.directory.iterdir() if path.is_file()},
        hardware_verified=a.device == "cuda:0")
    (a.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
