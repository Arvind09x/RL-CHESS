#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=train-r1-2pass
#SBATCH --output=artifacts/slurm/%x-%j.out
# Diagnostic second-pass runs on the round-1 games (amendment A7, below): R1-mix and R1-pure-full
# trained for two passes over the same 55,253 self-play positions, ceil(2N / 24) = 4,605 updates, from
# P, into runs/train/r1-mix-2pass and runs/train/r1-pure-full-2pass. Diagnostic only: neither enters
# validation unless that is decided before validation starts. The one-pass runs are not touched.
#   - R1-mix-2pass keeps the hard stop (+0.25), as R1-mix did.
#   - R1-pure-full-2pass runs without it, as R1-pure-full did under A6 (breaches recorded).
# For each: train to update 2250, compare exactly with the one-pass run at update 2250 (weights,
# optimizer, sampler, RNG, progress, metrics, evaluation; provenance differs by design and is listed),
# stop that recipe if it differs, otherwise resume to the end. Every train process measures P fresh
# and refuses to start unless the quota covers its remaining checkpoints plus 1 GiB.
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
QUOTA_BYTES=100000000000
AMENDMENT="A7, approved by Arvind on 17 Oct 2026 at 23:19 IST, after the one-pass round-1 results: diagnostic second-pass runs of R1-mix (hard stop kept) and R1-pure-full (hard stop off, as under A6) for two passes over the round-1 self-play positions (ceil(2N / 24) updates); not validation candidates unless decided before validation starts; the one-pass runs remain the round-1 candidates."
TM="$PY scripts/train_mixed.py"
echo "== free quota"
"$PY" -c "import json; from chess_rl.selfplay_rounds import free_quota; f = free_quota($QUOTA_BYTES); print(json.dumps(dict(method=f['method'], free_gib=f['free_bytes'] / 2**30, used_gib=f['used_bytes'] / 2**30, limit_gib=f['limit_bytes'] / 2**30)))"
$TM data --pointer "$POINTER" --data-config "$DATA_CONFIG" --p-checkpoint "$P" --p-sha256 "$P_SHA256" \
  --output "$ROUND_DATA" --expected-branch extended --expected-games 800 --expected-positions 67579 \
  || { echo "round data failed; nothing trained" >&2; report_cgroup_peak; exit 1; }
status=0
run() {  # run <recipe> <one-pass dir> <output dir> [extra train arguments]
  local recipe=$1 one=$2 out=$3
  shift 3
  local common=(--recipe "$recipe" --data-config "$DATA_CONFIG" --round-data "$ROUND_DATA" --p-checkpoint "$P"
                --p-sha256 "$P_SHA256" --device cuda:0 --threads 4 --expect-p-total "$EXPECT_P_TOTAL"
                --p-tolerance 0.001 --quota-bytes "$QUOTA_BYTES" --passes 2 --amendment-note "$AMENDMENT"
                --output "$out" "$@")
  echo "== $out to update 2250"
  $TM train "${common[@]}" --stop-after 2250 || return 1
  echo "== compare with $one at update 2250"
  $TM compare "$one" "$out" --update 2250 --state-only \
    || { echo "$out differs from $one at update 2250; not continuing" >&2; return 1; }
  echo "== $out resumed to the end"
  $TM train "${common[@]}"
}
run mix runs/train/r1-mix runs/train/r1-mix-2pass || status=1
run pure runs/train/r1-pure-full runs/train/r1-pure-full-2pass --no-hard-stop || status=1
$TM timings runs/train/r1-mix-2pass runs/train/r1-pure-full-2pass || status=1
report_cgroup_peak
echo "train-r1-2pass status=$status"
exit $status
