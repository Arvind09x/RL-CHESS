"""Offline M02 backbone acceptance; optional pinned Git-object source audit."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tests.backbone_checks import numerical_case, sharing_case, dropout_case, adapter_case, constructor_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maia3-checkout", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    framework = {}
    for name, obj in {
        "MultiheadAttention.__init__": torch.nn.MultiheadAttention.__init__,
        "MultiheadAttention._reset_parameters": torch.nn.MultiheadAttention._reset_parameters,
        "MultiheadAttention.forward": torch.nn.MultiheadAttention.forward,
        "multi_head_attention_forward": torch.nn.functional.multi_head_attention_forward,
        "Linear.reset_parameters": torch.nn.Linear.reset_parameters,
        "LayerNorm.__init__": torch.nn.LayerNorm.__init__,
        "LayerNorm.reset_parameters": torch.nn.LayerNorm.reset_parameters,
        "Dropout.forward": torch.nn.Dropout.forward,
    }.items():
        source = inspect.getsource(obj)
        framework[name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "source": source}
    source_check = "not requested; numerical checks run offline"
    if args.maia3_checkout:
        lock = json.loads((ROOT / "docs/upstream-lock.json").read_text())["repositories"]["maia3"]
        data = subprocess.check_output(["git", "-C", str(args.maia3_checkout), "show", lock["commit"] + ":maia3/models.py"])
        checksum = hashlib.sha256(data).hexdigest()
        assert checksum == lock["files"]["maia3/models.py"]["sha256"]
        source_check = {"commit": lock["commit"], "full_file_sha256": checksum,
                        "read_from_pinned_git_object": True, "full_upstream_module_imported": False}
    report = {
        "task": "M02", "status": "passed", "date": "2026-10-07",
        "review_status": "automated acceptance only; awaiting independent review",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version, "device": "cpu",
                        "cuda_build": torch.version.cuda, "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "installed_distributions": dict(sorted((d.metadata["Name"], d.version)
                                                              for d in importlib.metadata.distributions())),
                        "torch_build_config": torch.__config__.show()},
        "resolved_settings": {
            "input": ["B", 64, 128], "output": ["B", 64, 256], "blocks": 8,
            "width": 256, "heads": 8, "head_depth": 32, "ffn_width": 256,
            "normalization": "Post-LayerNorm plus encoder-final LayerNorm; eps=1e-5, affine and bias enabled, biased variance",
            "activation": "GELU approximate=none",
            "biases": "input projection, packed QKV, attention output, FFN, GAB and LayerNorm all enabled",
            "packed_qkv": {"shape": [768, 256], "initializer": "xavier_uniform_, gain=1, fan_in=256, fan_out=768",
                           "uniform_bound": (6 / 1024) ** 0.5, "bias_initial_value": 0},
            "attention_output": {"weight_initializer": "Linear kaiming_uniform_(a=sqrt(5)); uniform +/-1/16",
                                 "bias_initial_value": 0, "bias_reset": "MHA resets constructor Linear bias to zero"},
            "other_linears": "kaiming_uniform_(a=sqrt(5)); weights and biases uniform +/-1/sqrt(fan_in)",
            "norm_initialization": "weight=1, bias=0", "square_embeddings": "[64,256], zero initialized, additive",
            "bank": "one [4096,32] shared parameter, Xavier normal, gain=1; existing M01 operation unchanged",
            "dropout": {"training_default": 0.1, "attention": "probabilities after softmax, before value contraction",
                        "residual1": "attention output before residual sum/norm1",
                        "ffn": "after GELU(linear1), before linear2",
                        "residual2": "linear2 output before residual sum/norm2",
                        "evaluation": "all disabled", "input_or_position_dropout": False},
            "gab_logits": "q dot k / sqrt(32) + GAB; softmax over keys",
            "model_wide_reset": False, "elo_time_conditioning": False,
            "heads_or_relative_terms": False},
        "constructors": constructor_case(),
        "cases": [numerical_case(batch, dtype) for dtype in (torch.float32, torch.float64) for batch in (1, 2, 5)],
        "sharing": [sharing_case(dtype) for dtype in (torch.float32, torch.float64)],
        "dropout": dropout_case(), "adapter": adapter_case(),
        "pinned_source_check": source_check, "installed_framework_source": framework,
        "limitations": ["CPU macOS arm64 only; CUDA/MPS/reduced precision and other platforms unvalidated",
                        "Upstream does not pin Torch; no upstream framework, RNG-stream or checkpoint equivalence claimed",
                        "3,567,168 parameters is the M02 backbone only, not the final combined model",
                        "M03 onward, heads, relative terms, data, training, evaluator and search unimplemented",
                        "SGD update is an optimizer semantics check, not a training run; no subagents or paid compute"]}
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
