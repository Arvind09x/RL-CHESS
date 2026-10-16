#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=selfplay-round1
#SBATCH --output=artifacts/slurm/%x-%j.out
# Self-play round 1: 200 games from P with the frozen settings, 5 workers on one GPU. The first
# submission freezes the plan with SIMS; later submissions resume unfinished games. The
# directory name carries SIMS, so changing it starts a fresh campaign and leaves a stopped one
# intact; the check below refuses a plan whose budget differs from SIMS. When all 200 are
# finished, finish.sh archives them and loads them into the replay buffer.
# Set SIMS from the smoke result before the first submission.
SIMS=64
source jobs/b200/common.sh
source jobs/b200-selfplay/finish.sh
job_banner
assert_gpu
P=artifacts/p-1m/step-090000.pt
DIR=runs/selfplay/round1-sims$SIMS  # switching budget starts a fresh campaign; the old plan stays intact
test -f "$P" || { echo "missing $P" >&2; exit 1; }
if test ! -e "$DIR"; then
  "$PY" scripts/run_selfplay_campaign.py create --directory "$DIR" --checkpoint "$P" --games 200 \
    --simulations "$SIMS" --seed 20261017 --device cuda:0 \
    --purpose "self-play round 1 from P (1M step 90000)"
fi
"$PY" - "$DIR" "$SIMS" <<'PYEOF'
import sys
from chess_rl.selfplay_campaign import load
planned = load(sys.argv[1])["settings"]["simulation_budget"]
assert planned == int(sys.argv[2]), f"plan has {planned} simulations, script has {sys.argv[2]}"
PYEOF
"$PY" scripts/run_selfplay_campaign.py run --directory "$DIR" --workers 5 || echo "run stopped early" >&2
"$PY" scripts/run_selfplay_campaign.py status --directory "$DIR"
finish_campaign "$DIR"
report_cgroup_peak
