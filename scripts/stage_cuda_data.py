"""Copy verified D07 data bytes to portable paths; never rewrite frozen files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from chess_rl.supervised import BUNDLE_FILES, open_dataset

PATH_FIELDS = {"bundle", "manifest_path", "games_path", "split_audit_path", "prepared", "teacher_evidence"}
PREPARED_FILES = ("samples.jsonl", "overlap-audit.json", "protected-heldout.jsonl",
                  "heldout-games.jsonl", "exclusions.jsonl")


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def stage(config_path, destination, checkpoint=None):
    config_path = config_path.resolve()
    source = json.loads(config_path.read_text())
    if source["version"] != "cuda-portable-data-v1" or set(source["paths"]) != PATH_FIELDS:
        raise ValueError("portable data configuration schema mismatch")
    paths = {k: (config_path.parent / v).resolve() for k, v in source["paths"].items()}
    data = open_dataset(**paths, split="train")  # unchanged authoritative loader
    destination.mkdir(parents=True, exist_ok=False)
    targets = dict(bundle="bundle", manifest_path="manifest.json", games_path="games.jsonl",
        split_audit_path="split-audit.json", prepared="prepared", teacher_evidence="teacher-evidence.json")
    files = [(paths[k], targets[k]) for k in PATH_FIELDS - {"bundle", "prepared"}]
    files += [(paths["bundle"]/name, f"bundle/{name}") for name in (*BUNDLE_FILES, "summary.json")]
    files += [(paths["prepared"]/name, f"prepared/{name}") for name in PREPARED_FILES]
    if checkpoint: files.append((checkpoint.resolve(), "checkpoint.pt"))
    inventory = {}
    for original, relative in sorted(files, key=lambda item: item[1]):
        before = identity(original)
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        with copied.open("xb") as out, original.open("rb") as inp:
            shutil.copyfileobj(inp, out, 1024 * 1024)
        if identity(copied) != before or identity(original) != before:
            raise ValueError(f"staging bytes changed: {original}")
        inventory[relative] = dict(source=str(original), **before)
    portable = dict(version=source["version"], paths=targets)
    (destination/"data-config.json").write_text(json.dumps(portable, indent=2) + "\n")
    staged = {k: destination / v for k, v in targets.items()}
    copied_data = open_dataset(**staged, split="train")
    assert len(copied_data) == len(data)
    assert [copied_data.reference(i) for i in range(len(data))] == [data.reference(i) for i in range(len(data))]
    report = dict(version="cuda-byte-staging-v1", files=inventory,
        train_count=len(data), original_frozen_bytes_unchanged=True, staged_d07_verified=True)
    (destination/"staging-evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    print(json.dumps(stage(a.config, a.destination, a.checkpoint), indent=2))


if __name__ == "__main__": main()
