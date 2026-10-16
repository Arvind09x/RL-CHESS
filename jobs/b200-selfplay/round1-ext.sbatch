#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=selfplay-round1-ext
#SBATCH --output=artifacts/slurm/%x-%j.out
# Round-1 extension (amendment A2.1; deadline moved to 11:00 on 16 Oct, before any c/d game):
# halves c and d, 200 games each, seeds 20261019 and 20261020, same settings as a and b.
# Submit after job 554 has ended, however it ended (the cluster refuses a second submitted job), then:
#   1. Requires the core choice (round1.sbatch's CHOICE) and finishes the core buffer
#      idempotently (no-ops if job 554 did it). No core choice: exit with an error, no pointer.
#   2. Core choice a alone: no extension; writes the training pointer for a.
#   3. Plays c, then d only if c finished by DEADLINE. A half starts only if the projected disk
#      for it and its buffer fits the quota (a scan that cannot be completed counts as not
#      fitting); c not fitting means no extension, d not fitting means c alone. A half is resumed until it finishes or DEADLINE passes, so the choice is
#      always made in this job.
#   4. Records the extension choice once (c counts if finished by DEADLINE; d only if c counts),
#      loads the counted set into its own buffer (a load failing 3 times, any cause, falls back to
#      the core), then writes the training pointer once.
# No pointer is written (exit nonzero; investigate or resubmit) if: the core choice is missing;
# the core buffer fails to ingest or verify; the extension choice record is malformed or names
# another core; a disk check fails in a way other than not fitting; an OSError while verifying;
# or the job is killed.
# The pointer is what mixed training reads, and it is final: a fallback closes the extension.
# The core choice record (round1-sims128-choice.json) is never touched.
SIMS=128
DEADLINE=2026-10-17T11:00:00+05:30
QUOTA_BYTES=100000000000  # 100 GB, for the du fallback when `quota -w` does not parse
LOAD_RETRY_SECONDS=60
source jobs/b200/common.sh
source jobs/b200-selfplay/finish.sh
export PYTHONUNBUFFERED=1
job_banner
mkdir -p runs/selfplay
exec 9>>runs/selfplay/.round1-ext.lock
flock -n 9 || { echo "another round-1 extension job holds runs/selfplay/.round1-ext.lock" >&2; exit 1; }
assert_gpu
P=artifacts/p-1m/step-090000.pt
test -f "$P" || { echo "missing $P" >&2; exit 1; }
A=runs/selfplay/round1a-g200-sims$SIMS
B=runs/selfplay/round1b-g200-sims$SIMS
C=runs/selfplay/round1c-g200-sims$SIMS
D=runs/selfplay/round1d-g200-sims$SIMS
BUF_ABC=runs/selfplay/round1-abc-sims$SIMS
BUF_ABCD=runs/selfplay/round1-abcd-sims$SIMS
CHOICE=runs/selfplay/round1-sims$SIMS-choice.json
EXT_CHOICE=runs/selfplay/round1-ext-sims$SIMS-choice.json
POINTER=runs/selfplay/round1-sims$SIMS-training-buffer.json
CAMPAIGN="$PY scripts/run_selfplay_campaign.py"
EXT="$PY scripts/extend_selfplay_round.py"
DEADLINE_EPOCH=$(date -d "$DEADLINE" +%s)

field() {  # field <json file> <python expression over r>; lists print one item per line
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
v = eval(sys.argv[2])
print("\n".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v))
PYEOF
}

prepare() {  # create the half if new, then refuse a plan that differs from this script
  local dir=$1 seed=$2 half=$3
  if test ! -e "$dir"; then
    $CAMPAIGN create --directory "$dir" --checkpoint "$P" --games 200 --simulations "$SIMS" \
      --seed "$seed" --device cuda:0 \
      --purpose "self-play round 1, extension half $half, from P (1M step 90000)"
  fi
  "$PY" - "$dir" "$SIMS" "$seed" "$P" <<'PYEOF'
import hashlib, sys
from chess_rl.selfplay_campaign import load
plan = load(sys.argv[1])
found = plan["settings"]["simulation_budget"], plan["seed"], len(plan["games"])
expected = int(sys.argv[2]), int(sys.argv[3]), 200
assert found == expected, f"plan has (sims, seed, games) {found}, script expects {expected}"
with open(sys.argv[4], "rb") as stream:
    assert plan["checkpoint"]["sha256"] == hashlib.file_digest(stream, "sha256").hexdigest(), "plan is not from P"
PYEOF
}

play_half() {  # resume until every game finished or the deadline passed
  local dir=$1
  while ! $EXT finished --directory "$dir" --deadline "$DEADLINE" >/dev/null && test "$(date +%s)" -lt "$DEADLINE_EPOCH"; do
    $CAMPAIGN run --directory "$dir" --workers 5 --deadline "$DEADLINE" || echo "$dir: run stopped early" >&2
    $CAMPAIGN status --directory "$dir"
    if ! $EXT finished --directory "$dir" --deadline "$DEADLINE" >/dev/null && test "$(date +%s)" -lt "$DEADLINE_EPOCH"; then
      echo "$dir: unfinished before the deadline; resuming in 5 minutes" >&2
      sleep 300
    fi
  done
}

fits() {  # disk check: 0 fits, 1 does not; any other failure stops the job (no pointer)
  local rc=0
  $EXT disk --core "$CHOICE" --limit-bytes "$QUOTA_BYTES" "$@" || rc=$?
  case $rc in 0) return 0 ;; 10) return 1 ;; *) echo "disk check failed with status $rc" >&2; exit "$rc" ;; esac
}

if test -e "$POINTER"; then
  echo "training pointer already recorded:"; cat "$POINTER"; exit 0
fi
test -e "$CHOICE" || { echo "no core choice $CHOICE: round 1 is not decided; resubmit round1.sbatch" >&2; exit 1; }
mapfile -t CORE_HALVES < <(field "$CHOICE" 'r["halves"]')
CORE_BUFFER=$(field "$CHOICE" 'r["buffer"]')
for half in "${CORE_HALVES[@]}"; do $CAMPAIGN archive --directory "$half"; done
ingest_archives "$CORE_BUFFER" "${CORE_HALVES[@]}"

if test "${#CORE_HALVES[@]}" != 2; then
  $EXT pointer --output "$POINTER" --core "$CHOICE"
  report_cgroup_peak; exit 0
fi

NOTE=""
if test ! -e "$EXT_CHOICE"; then
  if fits --play "$C" --buffer "$BUF_ABC" --buffer-halves 3; then
    prepare "$C" 20261019 c
    play_half "$C"
    SKIP=()
    if $EXT finished --directory "$C" --deadline "$DEADLINE"; then
      if fits --play "$C" --play "$D" --buffer "$BUF_ABCD" --buffer-halves 4; then
        prepare "$D" 20261020 d
        play_half "$D"
        HALVES=(--half "$C" --buffer "$BUF_ABC" --half "$D" --buffer "$BUF_ABCD")
      else
        HALVES=(--half "$C" --buffer "$BUF_ABC")
        SKIP=(--skipped "$D=disk check did not pass before d started (did not fit, or could not be measured; see job log)")
      fi
    else
      HALVES=(--half "$C" --buffer "$BUF_ABC" --half "$D" --buffer "$BUF_ABCD")  # c late: neither counts
    fi
    $EXT choose --output "$EXT_CHOICE" --core "$CHOICE" --deadline "$DEADLINE" "${HALVES[@]}" "${SKIP[@]}"
  else
    NOTE="disk check for half c and its buffer did not pass (did not fit, or could not be measured; see job log); extension not started"
  fi
fi

if test -e "$EXT_CHOICE"; then
  EXT_BUFFER=$(field "$EXT_CHOICE" 'r["buffer"]')
  if test -n "$EXT_BUFFER"; then
    mapfile -t EXT_HALVES < <(field "$EXT_CHOICE" 'r["halves"]')
    # Archiving and loading resume where they stopped, so a failure is retried; one that persists
    # over 3 attempts (any cause, I/O included) is recorded and the pointer falls back to the core.
    # Each attempt runs in its own bash so that errexit holds inside it (it is ignored in the
    # parent's || context).
    for attempt in 1 2 3; do
      status=0
      PY="$PY" bash -euo pipefail -c 'source jobs/b200-selfplay/finish.sh; out=$1; shift
        for half in "$@"; do "$PY" scripts/run_selfplay_campaign.py archive --directory "$half"; done
        ingest_archives "$out" "$@"' _ "$EXT_BUFFER" "${EXT_HALVES[@]}" || status=$?
      test "$status" = 0 && break
      echo "loading the extended set: attempt $attempt exited with status $status" >&2
      test "$attempt" = 3 || sleep "$LOAD_RETRY_SECONDS"
    done
    test "$status" = 0 || NOTE="archiving or loading the extended set failed 3 times (last status $status)"
  fi
  $EXT pointer --output "$POINTER" --core "$CHOICE" --extension "$EXT_CHOICE" ${NOTE:+--note "$NOTE"}
else
  $EXT pointer --output "$POINTER" --core "$CHOICE" ${NOTE:+--note "$NOTE"}
fi
report_cgroup_peak
