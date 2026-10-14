#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=acceptance-100k
#SBATCH --output=artifacts/slurm/%x-%j.out
# Loader acceptance for bundle-100k. CPU/RAM only. The guarded sbatch takes no job arguments,
# hence one script per bundle. --write-summary fills loader_acceptance in summary.json when it
# is still null; run before any training opens the bundle (summary.json is in checkpoint provenance).
source jobs/b200/common.sh
job_banner
"$PY" scripts/check_bundle_loader.py --data-config configs/data-d09-100k.json --write-summary
report_cgroup_peak
