#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=train-r1-pure-full
#SBATCH --output=artifacts/slurm/%x-%j.out
# Exploratory R1-pure run to completion without the hard stop (amendment A6, below). The stopped run
# runs/train/r1-pure is left untouched and remains the original-rule result: R1-pure failed the
# forgetting rule at update 250 (delta +0.4957) and its original-rule candidate is P. Finishing it
# tests a new hypothesis (does the early value-head shift persist, and does it change playing
# strength) and does not undo that result.
#   1. Prints the measured free quota, then reopens and verifies ROUND_DATA (never rebuilt).
#   2. Trains into OUT up to update 250 only, then compares OUT with the stopped run at update 250:
#      weights, optimizer, sampler, RNG, progress, training metrics and the update-250 evaluation must
#      be exactly equal (provenance differs by design and is listed, not compared). A difference stops
#      the job before any further training.
#   3. Resumes OUT in a new process to the full 2,303 updates. Every breach of +0.25 is recorded in the
#      sidecars (hard_stop) without stopping (training_stopped false). summary.json reports both the
#      exploratory candidate (final checkpoint) and the original-rule candidate (P), and lists breaches.
# The train command refuses to start unless the quota covers the remaining checkpoints plus 1 GiB.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
mkdir -p runs/train
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
P=artifacts/p-1m/step-090000.pt
P_SHA256=cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2
EXPECT_P_TOTAL=2.2609
DATA_CONFIG=configs/data-d09-1m.json
POINTER=runs/selfplay/round1-sims128-training-buffer.json
ROUND_DATA=runs/train/r1-data
STOPPED=runs/train/r1-pure
OUT=runs/train/r1-pure-full
QUOTA_BYTES=100000000000
AMENDMENT="A6, approved by Arvind on 17 Oct 2026 at 19:55 IST after the stopped R1-pure run's update-250 result (delta +0.4957 > +0.25): an exploratory R1-pure run to completion without the hard stop, every breach recorded, candidate = final checkpoint, reported alongside the original-rule candidate (P); runs/train/r1-pure stays the original-rule result."
TM="$PY scripts/train_mixed.py"
echo "== free quota"
"$PY" -c "import json; from chess_rl.selfplay_rounds import free_quota; f = free_quota($QUOTA_BYTES); print(json.dumps(dict(method=f['method'], free_gib=f['free_bytes'] / 2**30, used_gib=f['used_bytes'] / 2**30, limit_gib=f['limit_bytes'] / 2**30)))"
$TM data --pointer "$POINTER" --data-config "$DATA_CONFIG" --p-checkpoint "$P" --p-sha256 "$P_SHA256" \
  --output "$ROUND_DATA" --expected-branch extended --expected-games 800 --expected-positions 67579 \
  || { echo "round data failed; nothing trained" >&2; report_cgroup_peak; exit 1; }
COMMON=(--recipe pure --data-config "$DATA_CONFIG" --round-data "$ROUND_DATA" --p-checkpoint "$P"
        --p-sha256 "$P_SHA256" --device cuda:0 --threads 4 --expect-p-total "$EXPECT_P_TOTAL" --p-tolerance 0.001
        --quota-bytes "$QUOTA_BYTES" --no-hard-stop --amendment-note "$AMENDMENT" --output "$OUT")
echo "== R1-pure-full to update 250"
$TM train "${COMMON[@]}" --stop-after 250 || { report_cgroup_peak; exit 1; }
echo "== compare with the stopped run at update 250"
$TM compare "$STOPPED" "$OUT" --update 250 --state-only \
  || { echo "R1-pure-full differs from the stopped run at update 250; not continuing" >&2; report_cgroup_peak; exit 1; }
echo "== R1-pure-full resumed to the end"
status=0
$TM train "${COMMON[@]}" || status=1
$TM timings "$OUT" || status=1
report_cgroup_peak
echo "train-r1-pure-full status=$status"
exit $status
