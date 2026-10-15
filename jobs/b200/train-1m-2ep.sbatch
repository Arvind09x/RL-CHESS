#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --job-name=train-1m-2ep
#SBATCH --output=artifacts/slurm/%x-%j.out
# 1M, 2 epochs (the original frozen plan; optional comparison against the 3-epoch main run): 2 epochs over 1,000,000 positions at batch 32 = 62,500 updates, warmup 500,
# cosine horizon 62,500, validation and checkpoint every 500. Output must not exist.
# Reads both splits from the verified sample cache (jobs/b200/cache-1m.sbatch builds it).
source jobs/b200/common.sh
job_banner
assert_gpu
"$PY" scripts/train_supervised.py --device cuda:0 --data-config configs/data-d09-1m.json --data-cache use \
  --output artifacts/p-1m-2ep --steps 62500 --batch-size 32 --threads 4 \
  --learning-rate 1e-4 --weight-decay 0.01 --max-gradient-norm 10 \
  --warmup-steps 500 --schedule-total-steps 62500 --validation-every 500 --seed 20261010
"$PY" scripts/report_step_timing.py artifacts/p-1m-2ep
report_cgroup_peak
