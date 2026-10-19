#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=match-test-round2
#SBATCH --output=artifacts/slurm/%x-%j.out
# Test match, bound to the ROUND-2 selection (match-driver plan, round-2 plan decision 5): the network selected in
# runs/matches/selection-round2.json vs P on the 64 reserved test openings, played once, same frozen settings as
# every validation match (128 simulations, 512-ply cap, 5 workers on cuda:0). The plan refuses any R other than
# the selected one, refuses a round-1 selection now that the round-2 record exists, and refuses a second test
# plan (runs/matches/test-plan-claim.json). Claim rule: R improves over P only if the lower bound of R's test
# pair-bootstrap interval is above 0.5; entirely below 0.5 indicates regression; containing 0.5 supports neither.
# Resubmitting resumes the same plan. Submit only when no other job of this user exists, and only after Arvind
# has decided that the round-2 selection is the final model (no further round).
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
RM="$PY scripts/run_match.py --round 2"
M=runs/matches
ID=$("$PY" -c "import json; s=json.load(open('$M/selection-round2.json')); assert s['final'], s['status']; print('test-P-vs-' + s['selected']['name'])") \
  || { echo "no final round-2 selection; nothing planned" >&2; exit 1; }
echo "== test match $ID (round-2 selection)"
test -e $M/$ID/plan.json || $RM plan --kind test --device cuda:0 --workers 5 >/dev/null \
  || { echo "test plan refused; nothing played" >&2; exit 1; }
status=0
$RM run --match-id $ID || status=1
$RM report --match-id $ID || status=1
report_cgroup_peak
echo "match-test-round2 status=$status"
exit $status
