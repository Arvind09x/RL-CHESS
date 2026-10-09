"""CPU-testable CUDA format boundaries; hardware checks live in check_cuda_support."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.training.checkpoints import (CUDA_CHECKPOINT_VERSION, CheckpointError,
    ModelConfig, load_inference, restore_training, _cpu_tree)
from .test_checkpoints import saved, threads, rng_snapshot, assert_rng_equal, assert_tree_equal


def cuda_payload(saved):
    payload = torch.load(saved["path"], weights_only=True)
    payload["version"] = CUDA_CHECKPOINT_VERSION
    payload["model_config"]["device"] = "cuda:0"
    training = payload["training"]
    # Synthetic metadata for CPU schema tests, never a hardware attestation.
    training["rng"]["cuda"] = dict(device="cuda:0", state=torch.ones(32, dtype=torch.uint8))
    training["runtime"].update(deterministic=True, deterministic_warn_only=False,
        default_dtype="torch.float32", matmul_precision="highest")
    training["cuda_runtime"] = dict(cuda="12.6", cudnn=91002, name="synthetic",
        capability=[8, 0], total_memory=1024, visible_devices=1,
        matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False,
        cudnn_deterministic=True, cublas_workspace_config=":4096:8")
    return payload


def test_cuda_schema_can_be_read_for_cpu_inference(saved, tmp_path):
    payload = cuda_payload(saved)
    path = tmp_path / "cuda.pt"
    torch.save(payload, path)
    before = rng_snapshot()
    loaded = load_inference(path, expected_config=replace(saved["config"], device="cuda:0"))
    assert loaded.config.device == "cpu"
    assert_tree_equal(loaded.model.state_dict(), payload["model_state"])
    assert_rng_equal(before, rng_snapshot())
    with pytest.raises(CheckpointError, match="incompatible model configuration"):
        restore_training(path, expected_config=saved["config"],
            expected_provenance=payload["provenance"], dataset=saved["dataset"], source_id="fixture-source")
    assert_rng_equal(before, rng_snapshot())


@pytest.mark.parametrize("bad", ["v1_cuda", "v2_cpu", "cuda_double", "rng_missing",
    "rng_dtype", "rng_device", "rng_empty", "runtime_missing", "tf32", "benchmark",
    "workspace", "nondeterministic", "warn_only", "foreach", "fused", "visible"])
def test_cuda_metadata_rejects_unsupported_boundaries_on_cpu(saved, tmp_path, bad):
    payload = cuda_payload(saved)
    training = payload["training"]
    if bad == "v1_cuda": payload["version"] = "m06-checkpoint-v1"
    elif bad == "v2_cpu": payload["model_config"]["device"] = "cpu"
    elif bad == "cuda_double": payload["model_config"]["dtype"] = "float64"
    elif bad == "rng_missing": del training["rng"]["cuda"]
    elif bad == "rng_dtype": training["rng"]["cuda"]["state"] = torch.ones(32)
    elif bad == "rng_device": training["rng"]["cuda"]["device"] = "cuda:1"
    elif bad == "rng_empty": training["rng"]["cuda"]["state"] = torch.empty(0, dtype=torch.uint8)
    elif bad == "runtime_missing": del training["cuda_runtime"]["cuda"]
    elif bad == "tf32": training["cuda_runtime"]["matmul_tf32"] = True
    elif bad == "benchmark": training["cuda_runtime"]["cudnn_benchmark"] = True
    elif bad == "workspace": training["cuda_runtime"]["cublas_workspace_config"] = ""
    elif bad == "nondeterministic": training["runtime"]["deterministic"] = False
    elif bad == "warn_only": training["runtime"]["deterministic_warn_only"] = True
    elif bad in ("foreach", "fused"): training["optimizer"]["defaults"][bad] = True
    elif bad == "visible": training["cuda_runtime"]["visible_devices"] = 2
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError): load_inference(path)
    assert_rng_equal(before, rng_snapshot())


@pytest.mark.parametrize("device,dtype", [("cuda", "float32"), ("cuda:1", "float32"),
    ("mps", "float32"), ("cuda:0", "float64"), ("cuda:0", "float16")])
def test_device_contract_is_explicit(device, dtype):
    with pytest.raises(CheckpointError): ModelConfig(device=device, dtype=dtype)


def test_snapshots_own_cpu_storage():
    source = {"a": [torch.ones(3)], "b": (torch.arange(2),)}
    copy = _cpu_tree(source)
    source["a"][0].zero_()
    assert torch.equal(copy["a"][0], torch.ones(3))
    assert copy["b"][0].data_ptr() != source["b"][0].data_ptr()


@pytest.mark.parametrize("count", [0, 2])
def test_no_silent_device_fallback(saved, monkeypatch, count):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="exactly one visible"):
        load_inference(saved["path"], device="cuda:0")
    assert_rng_equal(before, rng_snapshot())
