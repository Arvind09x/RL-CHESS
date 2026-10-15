#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --job-name=train-1m
#SBATCH --output=artifacts/slurm/%x-%j.out
# Main 1M run, amended 15 Oct from 2 to 3 epochs once the cache made 1M cheap: 3 epochs over
# 1,000,000 positions at batch 32 = 93,750 updates, warmup 500, cosine horizon 93,750, validation
# and checkpoint every 500. P-candidate = lowest validation loss among checkpoints. Output must not exist.
# Reads both splits from the verified sample cache (jobs/b200/cache-1m.sbatch builds it).
source jobs/b200/common.sh
job_banner
assert_gpu
"$PY" scripts/train_supervised.py --device cuda:0 --data-config configs/data-d09-1m.json --data-cache use \
  --output artifacts/p-1m --steps 93750 --batch-size 32 --threads 4 \
  --learning-rate 1e-4 --weight-decay 0.01 --max-gradient-norm 10 \
  --warmup-steps 500 --schedule-total-steps 93750 --validation-every 500 --seed 20261010
"$PY" scripts/report_step_timing.py artifacts/p-1m
report_cgroup_peak
