#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=match-smoke
#SBATCH --output=artifacts/slurm/%x-%j.out
# Smoke match (match-driver plan, decision 8): P vs R1-mix on the two checked smoke openings
# (italian-1, italian-2), 2 pairs = 4 games, at the evaluation settings (128 simulations, c_puct 1.5,
# no noise, maximum-visit moves, 512-ply cap), 5 workers on cuda:0 (4 busy: there are 4 games).
# Before the smoke plays, this job freezes, once, everything validation depends on:
#   - runs/matches/openings.json: 64 validation + 64 test roots from protected-heldout.jsonl
#     (plies 6-12, standard start, full history), plus the two smoke openings;
#   - runs/matches/candidates.json: P, R1-mix (selectable), R1-pure-full (report-only, A6), the
#     excluded two-pass runs (A7), the selection, incomplete-validation, demo and claim rules.
# Both files are never rewritten; a rerun reuses them. The smoke report shows statuses, game lengths
# and timing only (outcomes are stored, not printed). Resubmitting resumes the smoke.
# Submit only when no other job of this user exists (the cluster allows one, srun included).
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
mkdir -p runs/matches
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
RM="$PY scripts/run_match.py"
M=runs/matches
ID=smoke-P-vs-R1-mix
echo "== freeze openings (once)"
test -e $M/openings.json || $RM openings --data-config configs/data-d09-1m.json
echo "== freeze candidates and selection rules (once)"
test -e $M/candidates.json || $RM candidates
"$PY" -c "import json; [print(n, json.load(open(f'$M/{n}'))['sha256']) for n in ('openings.json', 'candidates.json')]"
echo "== plan $ID (once)"
test -e $M/$ID/plan.json || $RM plan --kind smoke --candidate R1-mix --device cuda:0 --workers 5 >/dev/null
echo "== run $ID"
status=0
$RM run --match-id $ID || status=1
echo "== report $ID (statuses, lengths, timing only)"
$RM report --match-id $ID || status=1
report_cgroup_peak
echo "match-smoke status=$status"
exit $status
