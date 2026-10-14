#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=acceptance
#SBATCH --output=artifacts/slurm/%x-%j.out
# Usage: sbatch jobs/b200/acceptance.sbatch <100k|300k|1m>
# CPU/RAM only. --write-summary fills loader_acceptance in summary.json when it is still null;
# run this before any training opens the bundle (summary.json is in checkpoint provenance).
source jobs/b200/common.sh
name=${1:?bundle name: 100k, 300k or 1m}
job_banner
"$PY" scripts/check_bundle_loader.py --data-config "configs/data-d09-${name}.json" --write-summary
report_cgroup_peak
