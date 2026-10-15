#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=cache-100k
#SBATCH --output=artifacts/slurm/%x-%j.out
# Build the bundle-100k cache, then measure training from it: 600 updates with validation every
# 200, reporting steady-state step time, validation cost and GPU memory. The measured numbers,
# not projections, decide what the cache is worth.
source jobs/b200/common.sh
job_banner
assert_gpu
"$PY" scripts/build_dataset_cache.py --data-config configs/data-d09-100k.json --splits train validation
run="artifacts/cache-bench-100k/$SLURM_JOB_ID"
mkdir -p "$(dirname "$run")"
"$PY" scripts/train_supervised.py --device cuda:0 --data-config configs/data-d09-100k.json --data-cache use \
  --output "$run" --steps 600 --batch-size 32 --threads 4 --learning-rate 1e-4 --weight-decay 0.01 \
  --max-gradient-norm 10 --warmup-steps 50 --schedule-total-steps 600 --validation-every 200 --seed 20261010
"$PY" scripts/report_step_timing.py "$run" --project-steps 62500 --project-validation-every 500
report_cgroup_peak
