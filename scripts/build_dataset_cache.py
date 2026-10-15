"""Build the verified on-disk sample cache for a bundle's splits.

Verifies the bundle once (the summary must already carry loader_acceptance), writes each requested
split during its admission decode pass, reads samples back through the cache reader, and publishes
each split atomically. Records timings, peak memory and the cache manifests in
``<bundle>/cache/build-report.json``. Refuses when free disk is below the projected size.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time

from chess_rl.runtime import data_paths
from chess_rl.supervised import verify_bundle
from chess_rl.dataset_cache import build_split_cache, bytes_per_sample, cache_dir


def cgroup_memory_peak():
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        return dict(cgroup=str(path), memory_peak_bytes=int((path / "memory.peak").read_text()))
    except (OSError, StopIteration, ValueError) as error:
        return dict(cgroup=None, memory_peak_bytes=None, error=str(error))


def maxrss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation", "test"])
    p.add_argument("--sample-check", type=int, default=1000, help="fresh decodes compared against the cache per split")
    args = p.parse_args()
    paths = data_paths(args.data_config)
    bundle = Path(paths["bundle"])
    summary = json.loads((bundle / "summary.json").read_text())
    if summary.get("loader_acceptance") is None:
        raise SystemExit("summary.json has no loader_acceptance; run acceptance with --write-summary first")
    for split in args.splits:
        if cache_dir(bundle, split).exists():
            raise SystemExit(f"cache already exists for {split}: {cache_dir(bundle, split)}")
    needed = sum(summary["by_split"][s] for s in args.splits) * bytes_per_sample()
    free = shutil.disk_usage(bundle).free
    if free < needed * 1.05 + (1 << 30):
        raise SystemExit(f"insufficient disk: need ~{needed / 1e9:.1f} GB plus margin, {free / 1e9:.1f} GB free")
    report = dict(data_config=str(args.data_config.resolve()), bundle=str(bundle.resolve()), host=platform.node(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), projected_bytes=needed, splits={})
    start = time.perf_counter()
    verified = verify_bundle(**paths)
    report["verification_seconds"] = round(time.perf_counter() - start, 3)
    for split in args.splits:
        t0 = time.perf_counter()
        manifest = build_split_cache(verified, split, bundle=bundle, data_paths=paths, sample_check=args.sample_check)
        report["splits"][split] = dict(seconds=round(time.perf_counter() - t0, 3), count=manifest["count"],
                                       bytes=sum(a["bytes"] for a in manifest["arrays"].values()),
                                       admission_fingerprint=manifest["admission_fingerprint"],
                                       maxrss_bytes_after=maxrss_bytes())
        print(json.dumps({split: report["splits"][split]}), flush=True)
    del verified
    report.update(total_seconds=round(time.perf_counter() - start, 3), maxrss_bytes=maxrss_bytes(), **cgroup_memory_peak())
    (bundle / "cache" / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("verification_seconds", "total_seconds", "maxrss_bytes", "memory_peak_bytes")}))


if __name__ == "__main__":
    main()
