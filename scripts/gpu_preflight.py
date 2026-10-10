"""Inspect a college/server runtime before opening data or starting training."""
import argparse
import json
import platform
import subprocess

from chess_rl.runtime import configure, data_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--data-config")
    args = p.parse_args()
    import torch
    configure(args.device)
    from chess_rl.training.checkpoints import ModelConfig, cuda_runtime_config, runtime_config
    model = ModelConfig(device=args.device).build()
    with torch.inference_mode():
        policy, wdl = model.eval()(torch.zeros(1, 64, 128, device=args.device))
    assert policy.shape == (1, 4352) and wdl.shape == (1, 3)
    assert torch.isfinite(policy).all() and torch.isfinite(wdl).all()
    report = dict(passed=True, device=args.device, runtime=runtime_config(),
        parameters=sum(v.numel() for v in model.parameters()),
        cuda_runtime=cuda_runtime_config() if args.device == "cuda:0" else None)
    if args.device == "cuda:0":
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True)
        report["driver"] = result.stdout.strip() if result.returncode == 0 else "nvidia-smi unavailable"
    if args.data_config:
        paths = data_paths(args.data_config)
        from chess_rl.supervised import open_dataset
        report["train_positions"] = len(open_dataset(**paths, split="train"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
