#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=match-validation-round2
#SBATCH --output=artifacts/slurm/%x-%j.out
# Round-2 validation (round-2 plan, decisions 4 and 7; Arvind, 19 Oct 2026): R2-mix (selectable) and R2-pure
# (report-only) each play P on the same 64 frozen validation openings, paired colours, identical settings to
# round 1 (128 simulations, 512-ply cap, 5 workers on cuda:0), serially. R1-mix's sealed round-1 validation
# match is reused by the round-2 selection, never replayed. Same order as round 1: the selectable match, then
# the selection (once), then the report-only match, so R2-pure can never delay or change the selection.
#   1. Freezes runs/matches/candidates-round2.json once (P; R1-mix and R2-mix selectable; R1-pure-full and
#      R2-pure report-only; round-1 entries carried from the sealed round-1 record; every checkpoint's full
#      sha256 and lineage from R1-mix checked). Needs the round-1 openings.json, candidates.json and the final
#      selection.json.
#   2. Freezes both validation plans before any round-2 validation game (sources = the pulled revision; it must
#      not change until the matches and the selection are done).
#   3. Plays R2-mix's match. The full schedule is always played; nothing stops on observed scores.
#   4. Round-2 selection (`run_match.py --round 2 select`): R1-mix's sealed match and R2-mix's new one, under
#      the audited cross-revision rule, which needs runs/matches/evaluation-equivalence-round2.json (made on the
#      Mac by scripts/check_evaluation_equivalence.py and copied here). Without it, or with a non-final outcome,
#      the status goes to selection-round2-status.json and the job goes on; rerun `select` by hand later.
#   5. Plays R2-pure's match (report-only), then reports both matches (scores only for finished schedules).
# Resubmitting resumes whatever is left. No test match is planned. Submit only when no other job of this
# user exists.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
RM="$PY scripts/run_match.py --round 2"
M=runs/matches
test -e $M/openings.json -a -e $M/candidates.json -a -e $M/selection.json \
  || { echo "round-1 openings, candidates and final selection are required" >&2; exit 1; }
echo "== 1. round-2 candidate record"
if test -e $M/candidates-round2.json; then
  echo "already frozen: $M/candidates-round2.json"
else
  $RM candidates >/dev/null || { echo "candidate freeze failed; nothing planned or played" >&2; exit 1; }
fi
echo "== 2. plans"
for C in R2-mix R2-pure; do
  test -e $M/validation-P-vs-$C/plan.json \
    || $RM plan --kind validation --candidate $C --device cuda:0 --workers 5 >/dev/null \
    || { echo "plan for $C failed; nothing played" >&2; exit 1; }
done
status=0
echo "== 3. run validation-P-vs-R2-mix (selectable)"
$RM run --match-id validation-P-vs-R2-mix || status=1
if test $status -eq 0 -a ! -e $M/selection-round2.json; then
  echo "== 4. round-2 selection (R1-mix's sealed match reused; R2-pure not involved)"
  if test -e $M/evaluation-equivalence-round2.json; then
    $RM select || echo "selection not final (see $M/selection-round2-status.json); rerun select by hand" >&2
  else
    echo "no $M/evaluation-equivalence-round2.json yet: copy it from the Mac, then run select by hand" >&2
  fi
fi
echo "== 5. run validation-P-vs-R2-pure (report-only)"
$RM run --match-id validation-P-vs-R2-pure || status=1
echo "== reports"
for C in R2-mix R2-pure; do
  $RM report --match-id validation-P-vs-$C || status=1
done
report_cgroup_peak
echo "match-validation-round2 status=$status"
exit $status
