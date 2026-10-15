#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=05:00:00
#SBATCH --job-name=cache-1m
#SBATCH --output=artifacts/slurm/%x-%j.out
# Build the verified sample cache for bundle-1m (train + validation, ~34 GB) from one verified
# snapshot. Requires acceptance to have written loader_acceptance into summary.json first.
# CPU/RAM only; the job refuses when free disk is short of the projected size.
source jobs/b200/common.sh
job_banner
df -h "$HOME" | tail -1
"$PY" scripts/build_dataset_cache.py --data-config configs/data-d09-1m.json --splits train validation
report_cgroup_peak
