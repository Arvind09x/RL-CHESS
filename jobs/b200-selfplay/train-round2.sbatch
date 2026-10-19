#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=train-round2
#SBATCH --output=artifacts/slurm/%x-%j.out
# Round-2 training (round-2 plan, approved decisions of 18 Oct 2026 ~13:00 IST; training plan approved
# 18 Oct ~16:45 IST). Submit after selfplay-round2 has ended (one job per user). Resumable: resubmit
# after any stop; finished steps are verified and skipped.
#   1. Requires runs/selfplay/round2-sims128-buffer.json (written by round2.sbatch).
#   2. Deletes the round-2 dry-run self-play (the halves and buffer named by its own record, 8 games;
#      approved). The record itself is kept. If the record is not the expected one, nothing is deleted.
#   3. If runs/train/r2-data does not exist: disk check (round-2 arrays upper bound + 1 GiB). Only if the
#      measured free quota is below that (exit 10; any other failure stops the job), deletes the
#      intermediate checkpoints of runs/train/r1-mix and r1-pure-full (keeps update-002303.pt of each and
#      every sidecar, metric and summary; approved B.2), then checks again.
#      Builds r2-data (record and selection checks, verify_buffer, full replay audit, every sample played
#      by R1-mix, same filtering and hold-out rule r1-holdout-v1).
#   4. Reopens r2-data (every file hash; record, selection and R1-mix checked; the buffer is not opened),
#      checks the four round-2 archives against their summaries, then deletes the round-2 buffer
#      runs/selfplay/round2-abcd-sims128 (approved B.1; it can be rebuilt from the archives).
#   5. Dry run BOTH resolved recipes (4 updates, checkpoint every 2), uninterrupted and stopped after 2 then
#      resumed in a new process; compare at update 4 and inference must both pass, else the job stops
#      before the full runs. On a pass, writes passed.json and deletes the dry-run checkpoints (B.3).
#   6. R2-mix (original rule: flag +0.05, hard stop +0.25, fallback R1-mix), then R2-pure (no hard
#      stop, report-only). Every train process measures P fresh (must reproduce 2.2609 within 0.001) and
#      R1-mix fresh (total within 0.001 of its sidecar; the difference is recorded). Both attempted.
#   7. Timings and both summaries. Exit nonzero if anything failed.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
mkdir -p runs/train
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
assert_gpu
P=artifacts/p-1m/step-090000.pt
P_SHA256=cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2
START=runs/train/r1-mix/update-002303.pt
START_SHA256=32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2
EXPECT_P_TOTAL=2.2609
DATA_CONFIG=configs/data-d09-1m.json
RECORD=runs/selfplay/round2-sims128-buffer.json
SELECTION=runs/matches/selection.json
BUFFER=runs/selfplay/round2-abcd-sims128
DRY_RECORD=runs/selfplay/round2-dryrun-sims128-buffer.json
R1_DATA=runs/train/r1-data
R2_DATA=runs/train/r2-data
QUOTA_BYTES=100000000000
# Settings come from jobs/b200-selfplay/round2.conf, committed with the job (the guarded sbatch takes no job
# arguments). Keys:
#   R2_SAMPLING_MODE  weighted (default) | legacy
#   R2_UPDATES        positive integer; or "auto" = the smallest budget that gives one full pass over the
#                     round-2 training positions at the mix recipe's round-2 share (computed from
#                     runs/train/r2-data/manifest.json once the arrays exist, so it is exact and the same on a
#                     resubmit); legacy mode may leave it unset for the original ceil(union / 24) budget.
#                     Both recipes get the same budget.
#   R2_MIX_WEIGHTS    default "0.20 0.80";  R2_PURE_WEIGHTS  default "0.00 1.00"  (ordered [round 1, round 2])
CONF=jobs/b200-selfplay/round2.conf
test -f "$CONF" || { echo "write $CONF first (R2_SAMPLING_MODE, R2_UPDATES, R2_MIX_WEIGHTS, R2_PURE_WEIGHTS)" >&2; exit 1; }
echo "== settings ($CONF)"; cat "$CONF"
source "$CONF"
R2_SAMPLING_MODE=${R2_SAMPLING_MODE:-weighted}
R2_UPDATES=${R2_UPDATES:-}
read -r -a MIX_WEIGHTS <<< "${R2_MIX_WEIGHTS:-0.20 0.80}"
read -r -a PURE_WEIGHTS <<< "${R2_PURE_WEIGHTS:-0.00 1.00}"
case "$R2_SAMPLING_MODE:$R2_UPDATES" in
  weighted:auto) ;;
  weighted:) echo "weighted sampling needs R2_UPDATES (an integer or auto)" >&2; exit 1 ;;
  weighted:*|legacy:|legacy:*) ;;
  *) echo "invalid R2_SAMPLING_MODE: $R2_SAMPLING_MODE" >&2; exit 1 ;;
esac
test -z "$R2_UPDATES" -o "$R2_UPDATES" = auto || [[ "$R2_UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid R2_UPDATES: $R2_UPDATES" >&2; exit 1; }
# Validate mode and weights before any cleanup; the budget and the settings tag are fixed after step 4.
"$PY" - "$R2_SAMPLING_MODE" "${MIX_WEIGHTS[*]}" "${PURE_WEIGHTS[*]}" <<'PYEOF' || { echo "invalid R2 sampling settings" >&2; exit 1; }
import sys
from chess_rl.training.round_sampling import weight_units
mode, mix, pure = sys.argv[1:]
assert mode in ("weighted", "legacy"), "invalid R2_SAMPLING_MODE"
if mode == "weighted":
    for v in (mix, pure):
        weight_units(list(map(float, v.split())), 2)
PYEOF
MIX_SAMPLING=(--sampling-mode "$R2_SAMPLING_MODE")
PURE_SAMPLING=(--sampling-mode "$R2_SAMPLING_MODE")
if test "$R2_SAMPLING_MODE" = weighted; then
  MIX_SAMPLING+=(--round-weights "${MIX_WEIGHTS[@]}")
  PURE_SAMPLING+=(--round-weights "${PURE_WEIGHTS[@]}")
fi
AMENDMENT="Round-2 plan, approved decision 3 (Arvind, 18 Oct 2026, ~13:00 IST): R2-pure uses the pure recipe with no hard stop (every breach recorded, as amendment A6), candidate = final checkpoint, report-only, never selected."
T2="$PY scripts/train_round2.py"
TM="$PY scripts/train_mixed.py"
quota_now() {
  "$PY" -c "import json; from chess_rl.selfplay_rounds import free_quota; f = free_quota($QUOTA_BYTES); print(json.dumps(dict(method=f['method'], free_gib=round(f['free_bytes'] / 2**30, 3), used_gib=round(f['used_bytes'] / 2**30, 3))))"
}
remove() {  # remove <path>: report its size, then delete it
  if test -e "$1"; then
    echo "delete $1 ($(du -sh "$1" | cut -f1))"
    rm -rf -- "$1"
  else
    echo "already gone: $1"
  fi
}
fail() { echo "$1" >&2; report_cgroup_peak; exit 1; }

echo "== 1. round-2 buffer record"
test -e "$RECORD" || fail "no $RECORD: round-2 self-play has not finished; nothing done"
quota_now

echo "== 2. round-2 dry-run self-play (approved deletion)"
if test -e "$DRY_RECORD"; then
  DRY_PATHS=$("$PY" - "$DRY_RECORD" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["schema"] == "round2-buffer-v1" and r["games"] == 8, "unexpected dry-run record"
assert r["buffer"] == "runs/selfplay/round2-dryrun-abcd-sims128", r["buffer"]
assert len(r["halves"]) == 4 and all(h.startswith("runs/selfplay/round2-dryrun") and h.endswith("-g2-sims128")
                                     for h in r["halves"]), r["halves"]
print(" ".join(r["halves"] + [r["buffer"]]))
PYEOF
  ) || { echo "dry-run record is not the expected 8-game record; nothing of it deleted" >&2; DRY_PATHS=; }
  for D in $DRY_PATHS; do remove "$D"; done
else
  echo "no $DRY_RECORD; nothing deleted"
fi

echo "== 3. round-2 arrays"
if ! test -e "$R2_DATA"; then
  rc=0; $T2 disk --record "$RECORD" --quota-bytes "$QUOTA_BYTES" || rc=$?
  test $rc -eq 0 -o $rc -eq 10 || fail "disk check failed (exit $rc); nothing deleted, nothing built"
  if test $rc -eq 10; then
    echo "arrays + 1 GiB do not fit: deleting round-1 intermediate checkpoints (approved B.2)"
    for RUN in runs/train/r1-mix runs/train/r1-pure-full; do
      test -e "$RUN/update-002303.pt" -a -e "$RUN/update-002303.json" || fail "$RUN final checkpoint or sidecar missing; nothing deleted"
      for C in "$RUN"/update-*.pt; do test "$C" = "$RUN/update-002303.pt" || remove "$C"; done
    done
    quota_now
    $T2 disk --record "$RECORD" --quota-bytes "$QUOTA_BYTES" || fail "round-2 arrays still do not fit; free space and resubmit"
  fi
  $T2 data --record "$RECORD" --selection "$SELECTION" --data-config "$DATA_CONFIG" --start-checkpoint "$START" \
    --start-sha256 "$START_SHA256" --output "$R2_DATA" --expected-games 800 --quota-bytes "$QUOTA_BYTES" \
    || fail "round-2 data build failed; nothing trained"
fi

echo "== 4. reopen r2-data, then delete the round-2 buffer (approved B.1)"
$T2 data --record "$RECORD" --selection "$SELECTION" --data-config "$DATA_CONFIG" --start-checkpoint "$START" \
  --start-sha256 "$START_SHA256" --output "$R2_DATA" --expected-games 800 --quota-bytes "$QUOTA_BYTES" \
  || fail "r2-data does not reopen intact; buffer kept, nothing trained"
if test -e "$BUFFER"; then
  "$PY" - "$RECORD" "$BUFFER" <<'PYEOF' || fail "archive check failed; buffer kept, nothing trained"
import hashlib, json, sys
from pathlib import Path
record, buffer = json.load(open(sys.argv[1])), sys.argv[2]
assert record["buffer"] == buffer, (record["buffer"], buffer)
for half in map(Path, record["halves"]):
    summary = json.loads((half / "games.summary.json").read_text())
    with (half / "games.jsonl").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == summary["archive_sha256"], f"{half}: archive altered"
    print(json.dumps(dict(archive=str(half), games=summary["games"], intact=True)))
PYEOF
  remove "$BUFFER"
  quota_now
else
  echo "already gone: $BUFFER"
fi

echo "== budget and settings tag"
if test "$R2_UPDATES" = auto; then
  R2_UPDATES=$("$PY" - "$R2_DATA/manifest.json" "${MIX_WEIGHTS[*]}" <<'PYEOF'
import json, math, sys
from chess_rl.training.round_sampling import weight_units
counts = json.load(open(sys.argv[1]))["counts"]
units = weight_units(list(map(float, sys.argv[2].split())), 2)
share = units[1] / sum(units)  #the mix recipe's round-2 share of the 24 self-play draws per update
# the deadline sampler keeps every round within one draw of 24 * share * N, so N = ceil(count / (24 share)) + 1
# gives at least count round-2 draws: one full pass over round 2 for the mix recipe (round 1 gets the rest)
n = math.ceil(counts["training_positions"] / (24 * share)) + 1
print(json.dumps(dict(event="auto-budget", rule="smallest N with >= one full pass over round 2 at the mix round-2 share",
                      round2_training_positions=counts["training_positions"], round2_share=share, updates=n)), file=sys.stderr)
print(n)
PYEOF
  ) || fail "could not compute the automatic budget"
  echo "R2_UPDATES=$R2_UPDATES (auto)"
fi
BUDGET=(); test -z "$R2_UPDATES" || BUDGET=(--updates "$R2_UPDATES")
TAG=$("$PY" - "$R2_SAMPLING_MODE" "$R2_UPDATES" "${MIX_WEIGHTS[*]}" "${PURE_WEIGHTS[*]}" <<'PYEOF'
import sys
from chess_rl.evaluation.matches import digest
from chess_rl.training.round_sampling import VERSION, weight_units
mode, budget, mix, pure = sys.argv[1:]
weights = [list(map(float, v.split())) for v in (mix, pure)] if mode == "weighted" else None
if weights:
    for w in weights:
        weight_units(w, 2)
print(digest([VERSION, mode, int(budget) if budget else None, weights])[:16])
PYEOF
) || fail "invalid R2 sampling settings"
MIX_RUN="runs/train/r2-mix-$TAG"
PURE_RUN="runs/train/r2-pure-$TAG"
echo "mode=$R2_SAMPLING_MODE updates=${R2_UPDATES:-default} mix_weights=${MIX_WEIGHTS[*]} pure_weights=${PURE_WEIGHTS[*]} tag=$TAG"

ROUNDS=(--round-data "$R1_DATA" "$P_SHA256" --round-data "$R2_DATA" "$START_SHA256")
COMMON=(--data-config "$DATA_CONFIG" "${ROUNDS[@]}" --start-checkpoint "$START" --start-sha256 "$START_SHA256"
        --p-checkpoint "$P" --p-sha256 "$P_SHA256" --device cuda:0 --threads 4 --expect-p-total "$EXPECT_P_TOTAL"
        --p-tolerance 0.001 --start-tolerance 0.001 --quota-bytes "$QUOTA_BYTES")

echo "== 5. dry runs (both resolved recipes, 4 updates)"
for RECIPE in mix pure; do
  DRY="runs/train/dryrun-r2-$RECIPE-$TAG"
  if test "$RECIPE" = mix; then SAMPLING=("${MIX_SAMPLING[@]}"); else SAMPLING=("${PURE_SAMPLING[@]}"); fi
  if test -e "$DRY/passed.json"; then
    "$PY" - "$DRY" "$TAG" "$RECIPE" <<'PYEOF' || fail "dry-run marker/settings mismatch"
import json, sys
from pathlib import Path
root, key, recipe = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p = json.loads((root / "passed.json").read_text())
assert p["validation_key"] == key and p["recipe"] == recipe
for name in ("uninterrupted", "resumed"):
    c = json.loads((root / name / "config.json").read_text())["provenance"]
    s = json.loads((root / name / "summary.json").read_text())
    assert c["sampler"] == p["sampler"] and c["total_updates"] == 4 and c["recipe"] == recipe
    assert s["completed"] and s.get("round_sampling") == p["round_sampling"]
print(json.dumps(p))
PYEOF
  else
    DCOMMON=(--recipe "$RECIPE" "${COMMON[@]}" "${SAMPLING[@]}" --updates 4 --checkpoint-every 2)
    if test "$RECIPE" = pure; then DCOMMON+=(--no-hard-stop --amendment-note "$AMENDMENT"); fi
    $T2 train "${DCOMMON[@]}" --output "$DRY/uninterrupted" || fail "dry run (uninterrupted) failed; no full run"
    $T2 train "${DCOMMON[@]}" --output "$DRY/resumed" --stop-after 2 || fail "dry run (stop after 2) failed; no full run"
    $T2 train "${DCOMMON[@]}" --output "$DRY/resumed" || fail "dry run (resume) failed; no full run"
    $TM compare "$DRY/uninterrupted" "$DRY/resumed" --update 4 || fail "dry-run compare found a difference; no full run"
    $TM inference "$DRY/resumed/update-000004.pt" --data-config "$DATA_CONFIG" --device cuda:0 --threads 4 \
      || fail "dry-run inference check failed; no full run"
    $TM timings "$DRY/uninterrupted" "$DRY/resumed" || true
    "$PY" - "$DRY" "$TAG" "$RECIPE" <<'PYEOF' || fail "dry-run report incomplete"
import datetime, json, sys
from pathlib import Path
root, key, recipe = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root / "resumed/config.json").read_text())["provenance"]
s = json.loads((root / "resumed/summary.json").read_text())
assert s["completed"] and s["updates"] == 4
p = dict(validation_key=key, recipe=recipe, sampler=c["sampler"], round_sampling=s.get("round_sampling"),
         passed=["compare --update 4", "inference"], utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (root / "passed.json").open("x") as f:
    json.dump(p, f)
print(json.dumps(p))
PYEOF
    echo "dry run passed; deleting its checkpoints (approved B.3)"
    for C in "$DRY"/uninterrupted/update-*.pt "$DRY"/resumed/update-*.pt; do remove "$C"; done
    quota_now
  fi
done

status=0
echo "== 6a. R2-mix (selectable; hard stop at +0.25, fallback R1-mix)"
$T2 train --recipe mix "${COMMON[@]}" "${MIX_SAMPLING[@]}" ${BUDGET[@]+"${BUDGET[@]}"} --output "$MIX_RUN" || status=1
echo "== 6b. R2-pure (report-only; no hard stop)"
$T2 train --recipe pure "${COMMON[@]}" "${PURE_SAMPLING[@]}" ${BUDGET[@]+"${BUDGET[@]}"} --no-hard-stop --amendment-note "$AMENDMENT" --output "$PURE_RUN" || status=1
echo "== 7. timings and summaries"
$TM timings --round-data "$R2_DATA" "$MIX_RUN" "$PURE_RUN" || status=1
for RUN in "$MIX_RUN" "$PURE_RUN"; do
  test -e "$RUN/summary.json" && "$PY" -c "import json; s = json.load(open('$RUN/summary.json')); print(json.dumps({k: s[k] for k in ('name', 'role', 'completed', 'updates', 'total_updates', 'candidate', 'final_delta_total', 'threshold_breaches')} | dict(round_sampling=s.get('round_sampling')), indent=2))"
done
quota_now
report_cgroup_peak
echo "train-round2 status=$status"
exit $status
