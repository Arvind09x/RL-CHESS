#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --job-name=benchmark-3g
#SBATCH --output=artifacts/slurm/%x-%j.out
# Probe: does student-small (MaxTRES gres/gpu=1, no slice type named) get a 3g.90gb MIG slice (about
# 3/7 of a B200, versus the 1g.23gb slice used so far), and how much faster is self-play's work on it?
# Runs only the read-only benchmark on R1-mix (plays and stores nothing). If the job sits pending
# (only four 3g slices exist), cancel it with scancel; it holds your one job slot while pending.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
nvidia-smi -L || true
assert_gpu
"$PY" scripts/benchmark_selfplay.py --checkpoint runs/train/r1-mix/update-002303.pt --device cuda:0 --processes 5 \
  | tee "artifacts/slurm/benchmark-3g-$SLURM_JOB_ID.json"
