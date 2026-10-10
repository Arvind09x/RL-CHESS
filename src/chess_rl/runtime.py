"""Explicit single-process runtime setup and relocatable artifact paths."""
import json
import os
from pathlib import Path


def data_paths(config_path):
    path = Path(config_path).resolve()
    value = json.loads(path.read_text())
    fields = {"bundle", "manifest_path", "games_path", "split_audit_path", "prepared", "teacher_evidence"}
    if value.get("version") != "cuda-portable-data-v1" or set(value.get("paths", {})) != fields:
        raise ValueError("data configuration schema mismatch")
    paths = {}
    for name, raw in value["paths"].items():
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"invalid artifact path: {name}")
        paths[name] = (path.parent / raw).resolve()
        if not paths[name].exists():
            raise FileNotFoundError(f"missing {name}: {paths[name]}; stage the artifact bundle first")
    return paths


def configure(device="cpu", *, threads=1):
    """Call once before constructing models; no silent backend fallback."""
    import torch
    from chess_rl.training.checkpoints import ModelConfig, _cuda_available
    ModelConfig(device=device)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    if device == "cuda:0":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch")
        _cuda_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
