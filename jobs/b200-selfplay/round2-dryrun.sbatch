#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=selfplay-round2-dryrun
#SBATCH --output=artifacts/slurm/%x-%j.out
# Dry run of round2.sbatch: the same script end to end (selection read and sha256 check, four halves
# created and played, disk checks, archive, ingest, exact buffer verification, the once-only record)
# with 2 games per half (8 games, ~10-15 min) in separate directories (runs/selfplay/round2-dryrun*)
# and different seeds (20261121-20261124). Its outputs are never read by round-2 training.
# A resubmission after the self-play part finished skips straight to the benchmark (round2.sbatch
# returns instead of exiting when its record exists). Success marker: the last line, printed after the
# benchmark. Submit only when no other job of this user exists, and before round2.sbatch.
export TAG=round2-dryrun GAMES=2 SEED_BASE=20261121
source jobs/b200-selfplay/round2.sbatch
# Then the read-only benchmark on this job's slice (plays and stores nothing): where a simulation's
# time goes (encoding, network at batch 1/8/32/64, tree work) and how 5 concurrent processes contend.
echo "== benchmark (read-only)"
"$PY" scripts/benchmark_selfplay.py --checkpoint "$R" --device cuda:0 --processes 5 \
  | tee "artifacts/slurm/benchmark-$SLURM_JOB_ID.json"
# The only marker that the whole dry run (self-play and benchmark) succeeded; any failure exits earlier.
echo "round2-dryrun complete: self-play and benchmark"
