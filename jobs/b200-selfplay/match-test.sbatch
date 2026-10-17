#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=match-test
#SBATCH --output=artifacts/slurm/%x-%j.out
# Test match (match-driver plan): the selected R vs P on the 64 reserved test openings, played once.
# Requires a final runs/matches/selection.json; the plan refuses any R other than the selected one and
# refuses a second test plan (runs/matches/test-plan-claim.json). The claim rule: R improves over P only
# if the lower bound of R's test pair-bootstrap interval is above 0.5; entirely below 0.5 indicates
# regression; containing 0.5 supports neither. Resubmitting resumes the same plan.
# Submit only when no other job of this user exists.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
RM="$PY scripts/run_match.py"
M=runs/matches
ID=$("$PY" -c "import json; s=json.load(open('$M/selection.json')); assert s['final'], s['status']; print('test-P-vs-' + s['selected']['name'])")
test -e $M/$ID/plan.json || $RM plan --kind test --device cuda:0 --workers 5 >/dev/null
status=0
$RM run --match-id $ID || status=1
$RM report --match-id $ID || status=1
report_cgroup_peak
echo "match-test status=$status"
exit $status
