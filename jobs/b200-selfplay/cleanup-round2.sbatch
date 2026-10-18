#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=cleanup-round2
#SBATCH --output=artifacts/slurm/%x-%j.out
# Frees quota for round 2. Deletes exactly what Arvind approved on 18 Oct 2026 (~13:00 IST):
#   1. runs/selfplay/round1-abcd-sims128 (round-1 replay buffer): training reads the hash-verified
#      arrays in runs/train/r1-data; the buffer can be rebuilt from the four round-1 archives.
#   2. runs/selfplay/round1-a-sims128 and round1-ab-sims128: superseded buffers.
#   3. runs/train/r1-mix-2pass and r1-pure-full-2pass: every update-*.pt except the highest update of
#      each run (diagnostic runs, amendment A7); sidecars, metrics and summaries are kept.
#   4. runs/selfplay/smoke-64 and runs/train/dryrun-*: smoke and dry-run outputs.
# Before deleting anything it checks that what makes the deletions safe is present and intact:
# runs/train/r1-data opens with every file hash verified, and each round-1 half's archive
# (games.jsonl) matches its summary. If any check fails, nothing is deleted. Nothing else is touched;
# any other superseded directory found (for example a round1-abc buffer) is only reported.
# Prints the free quota before and after. Idempotent: rerunning skips what is already gone.
source jobs/b200/common.sh
export PYTHONUNBUFFERED=1
job_banner
exec 9>>runs/.heavy-job.lock
flock -n 9 || { echo "another heavy job holds runs/.heavy-job.lock" >&2; exit 1; }
quota_now() {
  "$PY" -c "import json; from chess_rl.selfplay_rounds import free_quota; f = free_quota(100_000_000_000); print(json.dumps(dict(method=f['method'], free_gib=round(f['free_bytes'] / 2**30, 2), used_gib=round(f['used_bytes'] / 2**30, 2))))"
}
echo "== quota before"; quota_now
echo "== safety checks (nothing is deleted if one fails)"
"$PY" - <<'PYEOF'
import hashlib, json
from pathlib import Path
from chess_rl.training.round_training import RoundData
data = RoundData.open("runs/train/r1-data")
print(json.dumps(dict(r1_data="intact", training=len(data.train), holdout=len(data.holdout))))
for half in "abcd":
    d = Path(f"runs/selfplay/round1{half}-g200-sims128")
    summary = json.loads((d / "games.summary.json").read_text())
    with (d / "games.jsonl").open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert sha == summary["archive_sha256"], f"{d}: archive does not match its summary"
    print(json.dumps(dict(archive=str(d), games=summary["games"], intact=True)))
PYEOF
remove() {  # remove <path>: report its size, then delete it
  if test -e "$1"; then
    echo "delete $1 ($(du -sh "$1" | cut -f1))"
    rm -rf -- "$1"
  else
    echo "already gone: $1"
  fi
}
echo "== 1-2. round-1 buffers"
for B in runs/selfplay/round1-abcd-sims128 runs/selfplay/round1-a-sims128 runs/selfplay/round1-ab-sims128; do remove "$B"; done
echo "== 3. two-pass checkpoints except the last of each run"
for R in runs/train/r1-mix-2pass runs/train/r1-pure-full-2pass; do
  test -d "$R" || { echo "missing $R"; continue; }
  LAST=$(ls "$R"/update-*.pt 2>/dev/null | sort | tail -1)
  echo "keep $LAST"
  for C in "$R"/update-*.pt; do test "$C" = "$LAST" || remove "$C"; done
done
echo "== 4. smoke and dry runs"
remove runs/selfplay/smoke-64
for D in runs/train/dryrun-*; do test -e "$D" && remove "$D"; done
echo "== not approved, reported only"
for D in runs/selfplay/round1-abc-sims128; do test -e "$D" && echo "present, kept: $D ($(du -sh "$D" | cut -f1))"; done
echo "== quota after"; quota_now
echo "cleanup-round2 done"
