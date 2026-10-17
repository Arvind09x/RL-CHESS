#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=train-r1-dryrun
#SBATCH --output=artifacts/slurm/%x-%j.out
# Dry run of round-1 training (train-mixed-plan, dry-run gate). Show the results before train-r1.sbatch.
#   1. Builds the shared self-play arrays ROUND_DATA once (verify_buffer and a full replay audit,
#      ~20 min, then one streaming pass), or reopens and verifies them if they exist. The full job
#      reuses them; nothing else from this job is reused.
#   2. Measures P alone (no updates) into BASELINE and prints its report: validation metrics,
#      per-position loss spread, the interval of the mean across validation games, and the
#      self-play hold-out. No automatic cutoff; review it with the dry-run results.
#   3. For each recipe, in runs/train/dryrun-<recipe>/: an uninterrupted 4-update run (checkpoints
#      at 2 and 4), and a run stopped after its update-2 checkpoint and resumed in a new process.
#      Every train process first measures P fresh and stops before training unless P's validation
#      total loss reproduces EXPECT_P_TOTAL within 0.001.
#   4. compare: the two runs' update-4 checkpoints must be exactly equal (weights, optimizer, sampler,
#      RNG) with equal training metrics (timing excluded). No tolerance is applied; a difference
#      fails the job and is diagnosed before any full training.
#   5. inference: the resumed run's update-4 checkpoint through load_inference gives exactly the
#      validation metrics its training process recorded.
#   6. timings: data build, cache opens, P measurement, per-update, validation, hold-out, checkpoint
#      save and restore times, plus checkpoint sizes.
#   7. Measures P again in a new process (it must reproduce BASELINE exactly) and prints the
#      baseline report last, next to the results above.
# Every step runs even if an earlier comparison fails, so all measurements are available; the job
# exits nonzero if any step failed. Submit after any other job of the user has ended.
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
BASELINE=runs/train/p-baseline
TM="$PY scripts/train_mixed.py"
# Two full runs plus the 8 dry-run checkpoints must fit with a 1 GiB reserve.
$TM data --pointer "$POINTER" --data-config "$DATA_CONFIG" --p-checkpoint "$P" --p-sha256 "$P_SHA256" \
  --output "$ROUND_DATA" --expected-branch extended --expected-games 800 --expected-positions 67579 \
  --extra-checkpoints 8 || { echo "round data failed; nothing trained" >&2; report_cgroup_peak; exit 1; }
BASE=(--data-config "$DATA_CONFIG" --round-data "$ROUND_DATA" --p-checkpoint "$P" --p-sha256 "$P_SHA256"
      --output "$BASELINE" --device cuda:0 --threads 4 --expect-p-total "$EXPECT_P_TOTAL" --p-tolerance 0.001)
status=0
echo "== P baseline"
$TM baseline "${BASE[@]}" || status=1
for RECIPE in mix pure; do
  A=runs/train/dryrun-$RECIPE/uninterrupted
  B=runs/train/dryrun-$RECIPE/resumed
  COMMON=(--recipe "$RECIPE" --data-config "$DATA_CONFIG" --round-data "$ROUND_DATA" --p-checkpoint "$P"
          --p-sha256 "$P_SHA256" --device cuda:0 --threads 4 --updates 4 --checkpoint-every 2
          --expect-p-total "$EXPECT_P_TOTAL" --p-tolerance 0.001)
  echo "== $RECIPE: uninterrupted"
  $TM train "${COMMON[@]}" --output "$A" || { status=1; continue; }
  echo "== $RECIPE: stopped after update 2"
  $TM train "${COMMON[@]}" --output "$B" --stop-after 2 || { status=1; continue; }
  echo "== $RECIPE: resumed in a new process"
  $TM train "${COMMON[@]}" --output "$B" || { status=1; continue; }
  echo "== $RECIPE: compare"
  $TM compare "$A" "$B" --update 4 || status=1
  echo "== $RECIPE: inference"
  $TM inference "$B/update-000004.pt" --data-config "$DATA_CONFIG" --device cuda:0 --threads 4 || status=1
done
$TM timings --round-data "$ROUND_DATA" runs/train/dryrun-{mix,pure}/{uninterrupted,resumed} || status=1
echo "== P baseline again (must reproduce the recorded one)"
$TM baseline "${BASE[@]}" || status=1
report_cgroup_peak
echo "dry-run status=$status"
exit $status
