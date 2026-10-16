#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=selfplay-smoke
#SBATCH --output=artifacts/slurm/%x-%j.out
# Self-play smoke from P at 64 simulations: one game per worker (5 games, 5 workers), then the
# status report (per-move seconds by phase, game lengths), the JSONL archive and a streaming
# ingest into a fresh replay buffer. Not training data. Resubmitting resumes the same games.
# Decide the round-1 budget from status: 200 games / 5 workers x mean plies x seconds per move.
source jobs/b200/common.sh
source jobs/b200-selfplay/finish.sh
export PYTHONUNBUFFERED=1
job_banner
assert_gpu
P=artifacts/p-1m/step-090000.pt
DIR=runs/selfplay/smoke-64
test -f "$P" || { echo "missing $P" >&2; exit 1; }
if test ! -e "$DIR"; then
  "$PY" scripts/run_selfplay_campaign.py create --directory "$DIR" --checkpoint "$P" --games 5 \
    --simulations 64 --seed 20261016 --device cuda:0 \
    --purpose "B200 smoke: timing and archive/ingest check; not training data"
fi
"$PY" scripts/run_selfplay_campaign.py run --directory "$DIR" --workers 5 || echo "run stopped early" >&2
"$PY" scripts/run_selfplay_campaign.py status --directory "$DIR"
finish_campaign "$DIR"
report_cgroup_peak
