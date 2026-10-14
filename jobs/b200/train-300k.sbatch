#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train-300k
#SBATCH --output=artifacts/slurm/%x-%j.out
# Frozen 300k plan: 3 epochs over 300,000 positions at batch 32 = 28,125 updates, warmup 500,
# cosine horizon 28,125, validation and checkpoint every 500. Output must not exist.
source jobs/b200/common.sh
job_banner
assert_gpu
"$PY" scripts/train_supervised.py --device cuda:0 --data-config configs/data-d09-300k.json \
  --output artifacts/p-300k --steps 28125 --batch-size 32 --threads 4 \
  --learning-rate 1e-4 --weight-decay 0.01 --max-gradient-norm 10 \
  --warmup-steps 500 --schedule-total-steps 28125 --validation-every 500 --seed 20261010
"$PY" scripts/report_step_timing.py artifacts/p-300k
report_cgroup_peak
