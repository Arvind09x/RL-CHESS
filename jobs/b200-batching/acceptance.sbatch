#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=match-batching-accept
#SBATCH --output=artifacts/slurm/%x-%j.out
# USER SUBMITS ONLY AFTER THE ROUND-2 TEST MATCH FINISHES, from an isolated checkout.
# Prepare artifacts/batching-source.json on the submission host first; see docs/match-batching.md.
# The guarded scheduler accepts this script path alone:
# sbatch jobs/b200-batching/acceptance.sbatch
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=artifacts/p-1m/step-090000.pt
P_SHA256=cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2
R=runs/train/r1-mix/update-002303.pt
R_SHA256=32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2
# Do not call common.sh's job_banner (it invokes Git). Source/weight identity is
# verified in Python against the submission-host manifest and the pins above.
printf 'job=%s node=%s cpus=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}"
assert_gpu
"$PY" scripts/run_batching_acceptance.py --source-record artifacts/batching-source.json \
  --left "$P" --left-sha256 "$P_SHA256" --right "$R" --right-sha256 "$R_SHA256" \
  --output "artifacts/batching-acceptance-${SLURM_JOB_ID}"
report_cgroup_peak
