#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=selfplay-round1
#SBATCH --output=artifacts/slurm/%x-%j.out
# Self-play round 1 from P: two 200-game halves at SIMS simulations, run in order, 5 workers on
# one GPU. Each half is its own frozen campaign with its own seed.
# Stop rule, fixed on 16 Oct before any round-1 game and enforced here:
#   - Half a has no deadline; the round needs it. Its games load into BUF_A.
#   - Half b plays only until DEADLINE (`run --deadline` stops starting and running games then).
#   - `choose` then records, once, in CHOICE: both halves (BUF_AB) if every half-b game finished
#     by DEADLINE, otherwise half a alone (BUF_A). Archiving and loading after the deadline do
#     not change the choice; only game completion times count. A recorded choice is final.
#   - If half a is still running at DEADLINE, half b never starts and the choice is half a.
# Mixed training reads its replay buffer from CHOICE. Resubmitting resumes whatever is left.
# Directory names carry SIMS, so changing it starts fresh campaigns; the check refuses drift.
SIMS=128
DEADLINE=2026-10-17T10:00:00+05:30
source jobs/b200/common.sh
source jobs/b200-selfplay/finish.sh
export PYTHONUNBUFFERED=1
job_banner
assert_gpu
P=artifacts/p-1m/step-090000.pt
test -f "$P" || { echo "missing $P" >&2; exit 1; }
A=runs/selfplay/round1a-g200-sims$SIMS
B=runs/selfplay/round1b-g200-sims$SIMS
BUF_A=runs/selfplay/round1-a-sims$SIMS
BUF_AB=runs/selfplay/round1-ab-sims$SIMS
CHOICE=runs/selfplay/round1-sims$SIMS-choice.json
CAMPAIGN="$PY scripts/run_selfplay_campaign.py"

prepare() {  # create the half if new, then refuse a plan that differs from this script
  local dir=$1 seed=$2 half=$3
  if test ! -e "$dir"; then
    $CAMPAIGN create --directory "$dir" --checkpoint "$P" --games 200 --simulations "$SIMS" \
      --seed "$seed" --device cuda:0 --purpose "self-play round 1, half $half, from P (1M step 90000)"
  fi
  "$PY" - "$dir" "$SIMS" "$seed" <<'PYEOF'
import sys
from chess_rl.selfplay_campaign import load
plan = load(sys.argv[1])
found = plan["settings"]["simulation_budget"], plan["seed"], len(plan["games"])
expected = int(sys.argv[2]), int(sys.argv[3]), 200
assert found == expected, f"plan has (sims, seed, games) {found}, script expects {expected}"
PYEOF
}

prepare "$A" 20261017 a
$CAMPAIGN run --directory "$A" --workers 5 || echo "half a: run stopped early" >&2
$CAMPAIGN status --directory "$A"
$CAMPAIGN archive --directory "$A"
ingest_archives "$BUF_A" "$A"

if test ! -e "$CHOICE"; then
  prepare "$B" 20261018 b
  $CAMPAIGN run --directory "$B" --workers 5 --deadline "$DEADLINE" || echo "half b: run stopped early" >&2
  $CAMPAIGN status --directory "$B"
fi
$CAMPAIGN choose --output "$CHOICE" --first "$A" --second "$B" --deadline "$DEADLINE" \
  --first-buffer "$BUF_A" --both-buffer "$BUF_AB"
if test "$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["halves"]))' "$CHOICE")" = 2; then
  $CAMPAIGN archive --directory "$B"
  ingest_archives "$BUF_AB" "$A" "$B"
fi
report_cgroup_peak
