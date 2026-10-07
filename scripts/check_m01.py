"""Offline M01 numerical acceptance; optional Git source verification only."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import inspect
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tests.gab_checks import numerical_case, sharing_case, functional_reference, pinned_reference


def source_check(checkout):
    lock = json.loads((ROOT / "docs/upstream-lock.json").read_text())["repositories"]["maia3"]
    data = subprocess.check_output(["git", "-C", str(checkout), "show", lock["commit"] + ":maia3/models.py"])
    checksum = hashlib.sha256(data).hexdigest()
    assert checksum == lock["files"]["maia3/models.py"]["sha256"]
    def method(text, class_name):
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sq_bias")
        return ast.get_source_segment(text, node)
    fixture = (ROOT / "tests/fixtures/gab_reference.py").read_text()
    assert method(data.decode(), "MHA") == method(fixture, "PinnedOperation")
    return {"commit": lock["commit"], "full_file_sha256": checksum, "verbatim_method_matches": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maia3-checkout", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = [numerical_case(batch, dtype, reference)
             for dtype in (torch.float32, torch.float64)
             for batch in (1, 2, 5) for reference in (pinned_reference, functional_reference)]
    report = {
        "task": "M01", "status": "passed", "date": "2026-10-07",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version, "device": "cpu",
                        "cuda_build": torch.version.cuda, "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "installed_distributions": dict(sorted((d.metadata["Name"], d.version)
                                                              for d in importlib.metadata.distributions())),
                        "torch_build_config": torch.__config__.show()},
        "dimensions": {"input": ["B", 64, 256], "heads": 8, "d1": 8, "d2": 32, "d3": 32,
                       "bank": [4096, 32], "output": ["B", 8, 64, 64]},
        "constructor_defaults": {
            "Linear": {"bias": True, "weight_reset": "kaiming_uniform_(a=sqrt(5))",
                       "weight_and_bias_uniform_bound": "1/sqrt(in_features)",
                       "resolved_weight_shapes": [[8, 256], [32, 512], [256, 32]],
                       "reset_source_sha256": hashlib.sha256(inspect.getsource(torch.nn.Linear.reset_parameters).encode()).hexdigest()},
            "LayerNorm": {"eps": 1e-5, "elementwise_affine": True, "bias": True,
                          "weight_initial_value": 1, "bias_initial_value": 0,
                          "normalized_shapes": [[32], [256]],
                          "variance": "biased, correction=0", "second_norm_scope": "all 256 coefficients jointly",
                          "reset_source_sha256": hashlib.sha256(inspect.getsource(torch.nn.LayerNorm.reset_parameters).encode()).hexdigest()},
            "GELU": {"approximate": "none"},
            "bank": {"initializer": "xavier_normal_", "gain": 1.0, "fan_in": 32, "fan_out": 4096,
                     "theoretical_std": (2 / (4096 + 32)) ** 0.5},
            "factory": {"default_device": "cpu", "default_dtype": "torch.float32",
                        "generator_device_and_dtype": "inherited from supplied bank"}},
        "cases": cases, "sharing": [sharing_case(dtype) for dtype in (torch.float32, torch.float64)],
        "source_check": source_check(args.maia3_checkout) if args.maia3_checkout else "not requested; uses frozen offline method",
        "limitations": ["CPU macOS arm64 only tested; no CUDA/MPS or reduced precision acceptance",
                        "No full Maia model imported; no upstream checkpoint or RNG stream equivalence claimed",
                        "M02 onward, attention integration, heads, relative terms, training and search remain pending"]}
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
