#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=selfplay-batching-bench
#SBATCH --output=artifacts/slurm/%x-%j.out
# User submits only AFTER the R2 test match, from a committed isolated checkout.
# Run prepare_batching_source.py on the submission host first. No Git on compute.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf 'job=%s node=%s cpus=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}"
assert_gpu
trap report_cgroup_peak EXIT
# Existing R1-mix artifact. Change BOTH path and SHA deliberately for another checkpoint.
# All four layouts use exactly the same frozen bytes, corpus, seeds and CPU/GPU allocation.
CHECKPOINT=runs/train/r1-mix/update-002303.pt
CHECKPOINT_SHA256=32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2
# Hard bound covers acceptance and search, including a wedged native CUDA operation.
timeout --signal=TERM --kill-after=30s 110m \
  "$PY" scripts/benchmark_selfplay_batching.py \
    --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --source-record artifacts/batching-source.json --device cuda:0 \
    --output "artifacts/selfplay-batching-${SLURM_JOB_ID}" --mode search \
    --batch-sizes 8 16 32 --active-games-per-worker 8 16 32 \
    --positions 160 --simulations 64 --repeats 2 --deadline-seconds 6500
