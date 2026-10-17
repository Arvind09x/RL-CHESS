#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=train-r1
#SBATCH --output=artifacts/slurm/%x-%j.out
# Round-1 training, both recipes back to back from P (train-mixed-plan; submit only after the dry
# run passed). R1-mix then R1-pure, each ceil(N / 24) updates with a checkpoint and sidecar every
# 250 updates and at the last, into runs/train/r1-<recipe>/. Each train process measures P fresh
# first and stops before training unless P reproduces EXPECT_P_TOTAL within 0.001; every forgetting
# delta is against that fresh measurement. A run stops at the first checkpoint more than 0.25 above
# P (its candidate is then the last checkpoint at or below that, or P). Flag and stop use the measured
# delta only; each checkpoint also records labels on the paired interval across validation games
# (lower end above 0; above +0.05), which never change either rule. summary.json records the
# candidate and why. Submit only after reviewing the dry run and its P baseline report.
# Resumable: resubmit after any stop. A finished run is left alone; an interrupted one resumes from
# its latest complete checkpoint and sidecar pair, setting aside a checkpoint whose sidecar was not
# written and recording superseded metric lines. ROUND_DATA comes from the dry run and is verified,
# not rebuilt. Both runs are attempted even if the first fails; the job exits nonzero if either did.
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
TM="$PY scripts/train_mixed.py"
$TM data --pointer "$POINTER" --data-config "$DATA_CONFIG" --p-checkpoint "$P" --p-sha256 "$P_SHA256" \
  --output "$ROUND_DATA" --expected-branch extended --expected-games 800 --expected-positions 67579 \
  || { echo "round data failed; nothing trained" >&2; report_cgroup_peak; exit 1; }
status=0
for RECIPE in mix pure; do
  echo "== R1-$RECIPE"
  $TM train --recipe "$RECIPE" --data-config "$DATA_CONFIG" --round-data "$ROUND_DATA" --p-checkpoint "$P" \
    --p-sha256 "$P_SHA256" --device cuda:0 --threads 4 --expect-p-total "$EXPECT_P_TOTAL" --p-tolerance 0.001 \
    --output "runs/train/r1-$RECIPE" || status=1
done
$TM timings --round-data "$ROUND_DATA" runs/train/r1-mix runs/train/r1-pure || status=1
report_cgroup_peak
echo "train-r1 status=$status"
exit $status
