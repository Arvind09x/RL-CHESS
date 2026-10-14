#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=train-1m
#SBATCH --output=artifacts/slurm/%x-%j.out
# Frozen 1M plan: 2 epochs over 1,000,000 positions at batch 32 = 62,500 updates, warmup 500,
# cosine horizon 62,500, validation and checkpoint every 500. Output must not exist.
source jobs/b200/common.sh
job_banner
assert_gpu
"$PY" scripts/train_supervised.py --device cuda:0 --data-config configs/data-d09-1m.json \
  --output artifacts/p-1m --steps 62500 --batch-size 32 --threads 4 \
  --learning-rate 1e-4 --weight-decay 0.01 --max-gradient-norm 10 \
  --warmup-steps 500 --schedule-total-steps 62500 --validation-every 500 --seed 20261010
"$PY" scripts/report_step_timing.py artifacts/p-1m
report_cgroup_peak
