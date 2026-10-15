"""Loader acceptance for a labelled bundle: open every split through the production loader.

Repeats the acceptance step of ``expand_labels.py finalize`` on another machine and records
the evidence in a sidecar next to the bundle (``loader-acceptance.json``): verification time,
per-split counts and fingerprints, peak RSS, the job cgroup's ``memory.peak``, bundle byte
identities and the code revision. The bundle is verified once and each split selected from that
snapshot; the snapshot is discarded before ``--write-summary`` changes ``summary.json``.

``summary.json`` is part of every dataset's ``source_identity`` and therefore of every training
checkpoint's provenance. ``--write-summary`` fills ``loader_acceptance`` there exactly as finalize
would; use it once, before any training run opens the bundle, never after a checkpoint exists.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from chess_rl.runtime import data_paths
from chess_rl.supervised import verify_bundle

SPLITS = ("train", "validation", "test")


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def cgroup_memory_peak():
    """Peak memory of this process's cgroup v2 (the Slurm step, including descendants)."""
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        return dict(cgroup=str(path), memory_peak_bytes=int((path / "memory.peak").read_text()))
    except (OSError, StopIteration, ValueError) as error:
        return dict(cgroup=None, memory_peak_bytes=None, error=str(error))


def git_revision(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def maxrss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, help="sidecar path; default <bundle>/loader-acceptance.json")
    p.add_argument("--write-summary", action="store_true",
                   help="also record loader_acceptance in the bundle's summary.json (pre-training only)")
    args = p.parse_args()
    paths = data_paths(args.data_config)
    bundle = Path(paths["bundle"])
    summary_path = bundle / "summary.json"
    summary = json.loads(summary_path.read_text())
    import torch
    result = dict(bundle=str(bundle.resolve()), data_config=str(args.data_config.resolve()),
        summary_identity_before=file_identity(summary_path), bundle_identities=summary["output_identities"],
        code_revision=git_revision(Path(__file__).resolve().parents[1]),
        host=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        python=platform.python_version(), torch=torch.__version__, splits={})
    counts = {}
    start = time.perf_counter()
    # One verification pass covers every label of every split; selections only slice it.
    verified = verify_bundle(**paths)
    result["verification_seconds"] = round(time.perf_counter() - start, 3)
    result["verification_maxrss_bytes"] = maxrss_bytes()
    for split in SPLITS:
        t0 = time.perf_counter()
        data = verified.select(split)
        counts[split] = len(data)
        result["splits"][split] = dict(positions=len(data), fingerprint=data._admission_fingerprint,
                                       select_seconds=round(time.perf_counter() - t0, 3))
        del data
    del verified
    gc.collect()
    result.update(loader_acceptance=counts, total_seconds=round(time.perf_counter() - start, 3),
                  maxrss_bytes=maxrss_bytes(), **cgroup_memory_peak())
    expected = summary["by_split"]
    result["matches_summary_by_split"] = counts == expected
    if counts != expected:
        raise SystemExit(f"loaded counts {counts} differ from summary by_split {expected}")
    result["summary_written"] = False
    if args.write_summary:
        current = summary.get("loader_acceptance")
        if current is None:
            summary["loader_acceptance"] = counts
            # Same serialisation and atomic replace as expand_labels.save_json.
            tmp = summary_path.with_name(summary_path.name + ".tmp")
            tmp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
            os.replace(tmp, summary_path)
            result["summary_written"] = True
        elif current != counts:
            raise SystemExit(f"summary already records loader_acceptance {current}; refusing to overwrite")
    # Training's source_identity includes summary.json as it is now, so record the final bytes.
    result["summary_identity"] = file_identity(summary_path)
    output = args.output or bundle / "loader-acceptance.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("loader_acceptance", "verification_seconds", "total_seconds", "maxrss_bytes",
                                              "memory_peak_bytes", "cgroup", "matches_summary_by_split", "summary_written")}, indent=2))


if __name__ == "__main__":
    main()
