#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=selfplay-round2
#SBATCH --output=artifacts/slurm/%x-%j.out
# Round-2 self-play (round-2 plan, decided by Arvind on 18 Oct 2026, ~13:00 IST): 800 games in four
# halves of 200 (seeds 20261021-20261024) from the round-1 selection (R1-mix, read from
# runs/matches/selection.json, which must be final; sha256 checked), same settings as round 1:
# 128 simulations, c_puct 1.5, temperature 1 to ply 30 then 0, Dirichlet 0.3/0.25, 512-ply cap,
# 5 workers on cuda:0. No deadline: every half is played to the end (Arvind: "as soon as it gets
# done"). Every half, finished or not, is checked against this script's full settings on every run.
# Before each half a disk check reserves every half still to play plus the whole buffer not yet loaded
# (round-1 sizes: ~1.75 GiB played and ~1.8 GiB buffered per 200-game half) plus 1 GiB, and checks
# again before loading; if it does not fit the job stops (resubmit after freeing space).
# After all four halves: archive each, load all four archives into runs/selfplay/round2-abcd-sims128,
# verify the buffer exactly (one checkpoint, every archived game once, full audit) and record the
# result once in runs/selfplay/round2-sims128-buffer.json. Resubmitting resumes whatever is left.
# Submit only when no other job of this user exists.
# round2-dryrun.sbatch sources this script with TAG=round2-dryrun, GAMES=2 and SEED_BASE=20261121.
SIMS=128
TAG=${TAG:-round2}
GAMES=${GAMES:-200}
SEED_BASE=${SEED_BASE:-20261021}
source jobs/b200/common.sh
source jobs/b200-selfplay/finish.sh
export PYTHONUNBUFFERED=1
job_banner
mkdir -p runs/selfplay
# Not runs/.heavy-job.lock here: `run_selfplay_campaign.py run` takes that lock itself for each half.
exec 9>>runs/selfplay/.$TAG.lock
flock -n 9 || { echo "another $TAG job holds runs/selfplay/.$TAG.lock" >&2; exit 1; }
assert_gpu
CAMPAIGN="$PY scripts/run_selfplay_campaign.py"
EXT="$PY scripts/extend_selfplay_round.py"
NO_DEADLINE=2030-01-01T00:00:00+05:30  # "finished" needs a deadline; this one never binds
BUF=runs/selfplay/$TAG-abcd-sims$SIMS
RECORD=runs/selfplay/$TAG-sims$SIMS-buffer.json
read -r R R_SHA < <("$PY" - <<'PYEOF'
import hashlib, json
s = json.load(open("runs/matches/selection.json"))
assert s["final"] and s["status"] == "final", f"round-1 selection is not final: {s['status']}"
path, sha = s["selected"]["path"], s["selected"]["sha256"]
with open(path, "rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == sha, f"{path} differs from the selection"
print(path, sha)
PYEOF
)
test -n "${R_SHA:-}" || { echo "no final round-1 selection with a matching checkpoint; nothing played" >&2; exit 1; }
echo "start network: $R ($R_SHA)"
if test -e "$RECORD"; then
  echo "$TAG buffer already recorded:"; cat "$RECORD"; echo "$TAG self-play done"
  return 0 2>/dev/null || exit 0  # return when sourced (round2-dryrun.sbatch continues), else exit
fi

create_half() {  # create a half that does not exist yet
  local dir=$1 seed=$2 half=$3
  $CAMPAIGN create --directory "$dir" --checkpoint "$R" --games "$GAMES" --simulations "$SIMS" \
    --seed "$seed" --device cuda:0 --purpose "$TAG self-play, half $half, from the round-1 selection R1-mix"
}

validate_half() {  # every half, finished or not: refuse a plan that differs from this script in any setting
  "$PY" - "$1" "$SIMS" "$2" "$R_SHA" "$GAMES" <<'PYEOF'
import json, sys
from dataclasses import asdict
from chess_rl.selfplay_campaign import STANDARD_OPENING, frozen_settings, game_seed, load
directory, sims, seed, sha, games = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
plan = load(directory)
same = lambda value: json.loads(json.dumps(value))
checks = dict(
    settings=plan["settings"] == same(asdict(frozen_settings(sims))),
    opening=plan["opening"] == same(asdict(STANDARD_OPENING)),
    seed=plan["seed"] == seed, device=plan["device"] == "cuda:0",
    checkpoint=plan["checkpoint"]["sha256"] == sha,
    games=[g["seed"] for g in plan["games"]] == [game_seed(seed, i) for i in range(games)])
assert all(checks.values()), f"{directory}: plan differs from this script: {checks}"
print(json.dumps(dict(half=directory, validated=True)))
PYEOF
}

# Disk, estimated from round 1's measured sizes (per 200-game half: journal + archive ~1.75 GiB, buffer
# ~1.8 GiB), scaled by GAMES / 200. Before each half: every half still to play plus the whole buffer not
# yet loaded, plus a 1 GiB reserve. Before loading: the buffer share of every archive not yet loaded,
# plus the reserve. "Not yet loaded" = no ingest report in the buffer directory.
unloaded_halves() {
  local n=0 d
  for d in "${ALL_HALVES[@]}"; do test -e "$BUF/ingest-$(basename "$d").json" || n=$((n + 1)); done
  echo $n
}

fits() {  # fits <halves still to play> <halves not yet loaded>: 0 if the estimate fits the quota
  "$PY" - "$1" "$2" "$GAMES" <<'PYEOF'
import json, sys
from chess_rl.selfplay_rounds import free_quota
play, load, games = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
scale = games / 200
need = (play * 1.75 + load * 1.8) * scale + 1.0
f = free_quota(100_000_000_000)
print(json.dumps(dict(method=f["method"], free_gib=round(f["free_bytes"] / 2**30, 2), need_gib=round(need, 2),
                      halves_to_play=play, halves_to_load=load)))
sys.exit(0 if f["free_bytes"] >= need * 2**30 else 10)
PYEOF
}

finished() { $EXT finished --directory "$1" --deadline "$NO_DEADLINE" >/dev/null; }

ALL_HALVES=()
for i in 0 1 2 3; do
  ALL_HALVES+=("runs/selfplay/$TAG$(echo abcd | cut -c$((i + 1)))-g$GAMES-sims$SIMS")
done
for i in 0 1 2 3; do
  half=$(echo abcd | cut -c$((i + 1))) seed=$((SEED_BASE + i)) dir=${ALL_HALVES[$i]}
  test -e "$dir" && validate_half "$dir" "$seed"
  if test ! -e "$dir" || ! finished "$dir"; then
    to_play=0
    for d in "${ALL_HALVES[@]:$i}"; do { test -e "$d" && finished "$d"; } || to_play=$((to_play + 1)); done
    fits "$to_play" "$(unloaded_halves)" || { echo "half $half: not enough quota for the rest of round 2; free space and resubmit" >&2; report_cgroup_peak; exit 1; }
    test -e "$dir" || { create_half "$dir" "$seed" "$half"; validate_half "$dir" "$seed"; }
    for attempt in 1 2 3; do
      finished "$dir" && break
      $CAMPAIGN run --directory "$dir" --workers 5 || echo "$dir: run stopped early (attempt $attempt)" >&2
      $CAMPAIGN status --directory "$dir"
    done
    finished "$dir" || { echo "$dir: unfinished after 3 runs; resubmit to resume" >&2; report_cgroup_peak; exit 1; }
  fi
  $CAMPAIGN archive --directory "$dir"
done
HALVES=("${ALL_HALVES[@]}")

echo "== disk check before loading"
fits 0 "$(unloaded_halves)" || { echo "not enough quota to load the buffer; free space and resubmit" >&2; report_cgroup_peak; exit 1; }
echo "== load the four archives into $BUF"
ingest_archives "$BUF" "${HALVES[@]}"
echo "== verify and record the round-2 buffer (once)"
"$PY" - "$BUF" "$RECORD" "$R" "$R_SHA" "${HALVES[@]}" <<'PYEOF'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from chess_rl.selfplay_rounds import verify_buffer
buffer, record, start, start_sha, *halves = sys.argv[1:]
checked = verify_buffer(buffer, halves)
assert checked["checkpoint_sha256"] == start_sha, "round-2 games were played by another checkpoint"
value = dict(schema="round2-buffer-v1", decided_utc=datetime.now(timezone.utc).isoformat(),
             start_checkpoint=dict(path=start, sha256=start_sha), buffer=buffer,
             replay=str(Path(buffer) / "replay.sqlite"), halves=halves, games=checked["games"],
             training_positions_before_holdout=checked["training_positions"], checks=checked,
             rule=("four halves played to the end, no deadline (Arvind, 18 Oct 2026); recorded once; "
                   f"{len(halves)} halves of {checked['games'] // len(halves)} games"))
path = Path(record)
fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
with os.fdopen(fd, "w") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.link(tmp, path)
os.unlink(tmp)
print(json.dumps({k: v for k, v in value.items() if k != "checks"}, indent=2))
PYEOF
report_cgroup_peak
echo "$TAG self-play done"
