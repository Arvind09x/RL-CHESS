"""Verify sealed Stage 2 file hashes, optionally a train batch and CPU inference.

Only the production loader may internally validate held-out bundle contents.
No held-out dataset, training update, continuation, or benchmark is requested.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


TRAIN_SHA256 = "e11bcc4f4941f1cb43a88eb72d65c67299a3001829661bcf526d1bf0b80b99fc"
CHECKPOINT = "checkpoints/t03-tiny-overfit-20261009/final.pt"
CHECKPOINT_SHA256 = "d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify_hashes(root, anchor):
    require(identity(root / "inventory.json")["sha256"] == anchor,
            "inventory does not match independently retained SHA-256")
    inventory = json.loads((root / "inventory.json").read_text())
    files = inventory["files"]
    require(inventory["schema"] == "rl-chess-stage2-transfer-manifest-v1",
            "unexpected inventory schema")
    total = 0
    for relative, entry in files.items():
        raw = Path(relative)
        require(not raw.is_absolute() and ".." not in raw.parts, "unsafe inventory path")
        path = root / raw
        path.resolve().relative_to(root)
        require(not path.is_symlink(), f"payload symlink: {relative}")
        expected = {key: entry[key] for key in ("bytes", "sha256")}
        require(identity(path) == expected, f"file identity mismatch: {relative}")
        total += expected["bytes"]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(files) | {"inventory.json", "SHA256SUMS"},
            "unexpected or missing payload files")
    require(len(files) == inventory["file_count"] and total == inventory["total_bytes"],
            "inventory totals mismatch")
    return dict(verified_files=len(files), verified_bytes=total, inventory_sha256=anchor)


def verify_cpu(root):
    import numpy as np
    import torch
    from chess_rl.runtime import configure, data_paths
    from chess_rl.supervised import open_dataset
    from chess_rl.training.checkpoints import ModelConfig, load_inference, runtime_config

    configure("cpu")
    config = root / "data/d08/data-config.json"
    wrapper = json.loads(config.read_text())
    require(all(not Path(raw).is_absolute() for raw in wrapper["paths"].values()),
            "runtime paths must remain relative")
    paths = data_paths(config)
    for path in paths.values():
        path.relative_to(root)
    # Selected references contain membership metadata, not held-out targets.
    references = []
    with (paths["bundle"] / "selected.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train":
                references.append((row["game_id"], row["sample_ply"]))
    require(len(references) == len(set(references)) == 9000, "training membership count")
    membership_hash = hashlib.sha256(
        json.dumps(references, separators=(",", ":")).encode()).hexdigest()
    require(membership_hash == TRAIN_SHA256, "accepted training membership/order mismatch")
    data = open_dataset(**paths, split="train")
    require([data.reference(i) for i in range(len(data))] == references,
            "production loader training references mismatch")
    batch = data.batch([0, 1])
    require(batch.split == "train" and batch.references == tuple(references[:2]), "batch membership")
    require(batch.features.shape == (2, 64, 128) and batch.features.dtype == np.float32,
            "batch feature contract")
    require(batch.legal_mask.shape == (2, 4352) and batch.legal_mask.dtype == np.bool_,
            "batch mask contract")
    require(np.isfinite(batch.features).all() and np.isfinite(batch.wdl).all(), "nonfinite batch")
    require(batch.legal_mask[np.arange(2), batch.policy_classes].all(), "illegal training action")
    checkpoint = root / CHECKPOINT
    require(identity(checkpoint)["sha256"] == CHECKPOINT_SHA256, "T03 checkpoint bytes mismatch")
    loaded = load_inference(checkpoint, expected_config=ModelConfig(dropout=0.0), device="cpu")
    require(loaded.checkpoint_id == "T03-plumbing-step-26" and not loaded.model.training,
            "checkpoint identity or eval mode mismatch")
    with torch.inference_mode():
        policy, wdl = loaded.model(batch.to_torch(device="cpu")["features"])
    require(policy.shape == (2, 4352) and wdl.shape == (2, 3), "inference shapes")
    require(torch.isfinite(policy).all() and torch.isfinite(wdl).all(), "nonfinite inference")
    mask = torch.from_numpy(batch.legal_mask)
    priors = policy.masked_fill(~mask, -torch.inf).softmax(-1)
    probabilities = wdl.softmax(-1)
    require(torch.allclose(priors.sum(-1), torch.ones(2), atol=1e-6, rtol=0)
            and not priors[~mask].any(), "legal policy normalization")
    require(torch.allclose(probabilities.sum(-1), torch.ones(2), atol=1e-6, rtol=0),
            "WDL normalization")
    return dict(environment=runtime_config(), data_config=identity(config),
                dataset_count=len(data), training_membership_exact=True,
                training_membership_sha256=membership_hash, batch_size=2,
                features_shape=list(batch.features.shape), policy_shape=list(policy.shape),
                wdl_shape=list(wdl.shape), finite_inference=True,
                legal_policy_normalized=True, wdl_normalized=True,
                checkpoint_path=CHECKPOINT, checkpoint_id=loaded.checkpoint_id,
                checkpoint_identity=identity(checkpoint), model_config=asdict(loaded.config),
                load_mode="shared load_inference: weights for inference; training state not installed",
                production_loader_full_bundle_integrity_validation=True,
                heldout_dataset_opened=False, training_performed=False,
                training_continuation_verified=False, performance_benchmark=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--location", choices=("local-rehearsal", "backup-destination", "college-restoration"),
                        required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report.resolve()
    require(root != report_path and root not in report_path.parents, "report must be outside sealed payload")
    require(not report_path.exists(), "report already exists; use a new evidence directory")
    repository = Path(__file__).resolve().parents[1]
    report = dict(passed=False, location=args.location, root=str(root),
                  command=sys.orig_argv, cwd=str(Path.cwd()), host=platform.node(),
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(),
                  source_revision=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
                  source_status=subprocess.check_output(
                      ["git", "status", "--porcelain"], cwd=repository, text=True),
                  lock_identity=identity(repository / "uv.lock"), utility_identity=identity(Path(__file__)))
    try:
        report.update(verify_hashes(root, args.inventory_sha256))
        if args.cpu:
            report.update(verify_cpu(root))
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("passed", "location", "source_revision", "verified_files")}))


if __name__ == "__main__":
    main()
