#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=match-validation
#SBATCH --output=artifacts/slurm/%x-%j.out
# Validation (match-driver plan): R1-mix (selectable) and R1-pure-full (report-only, A6) each play P
# on the same 64 frozen validation openings, paired colours, identical settings, 5 workers on cuda:0.
# Needs the smoke job's frozen runs/matches/openings.json and candidates.json. Both plans are frozen
# before any validation game. Order: R1-mix's match, then `select` (once), then R1-pure-full's match,
# so the report-only match can never block, shrink or change the selection. The full schedule is
# always played: nothing stops on observed scores, and scores are reported only for finished matches.
# The final selection needs all 64 of R1-mix's pairs complete and no failed or interrupted game in its
# match; 48-63 complete pairs give a provisional result only (Arvind decides); fewer give no
# comparison. A non-final result goes to selection-status.json, never to selection.json.
# Resubmitting resumes whatever is left.
# Submit only when no other job of this user exists.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
RM="$PY scripts/run_match.py"
M=runs/matches
test -e $M/openings.json -a -e $M/candidates.json || { echo "run match-smoke.sbatch first" >&2; exit 1; }
for C in R1-mix R1-pure-full; do
  test -e $M/validation-P-vs-$C/plan.json || $RM plan --kind validation --candidate $C --device cuda:0 --workers 5 >/dev/null
done
status=0
echo "== run validation-P-vs-R1-mix"
$RM run --match-id validation-P-vs-R1-mix || status=1
if test $status -eq 0 -a ! -e $M/selection.json; then
  echo "== select (decided by the selectable candidate's match only)"
  $RM select || status=1
fi
echo "== run validation-P-vs-R1-pure-full (report-only)"
$RM run --match-id validation-P-vs-R1-pure-full || status=1
for C in R1-mix R1-pure-full; do
  $RM report --match-id validation-P-vs-$C || status=1
done
report_cgroup_peak
echo "match-validation status=$status"
exit $status
