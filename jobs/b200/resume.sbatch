#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=resume
#SBATCH --output=artifacts/slurm/%x-%j.out
# The guarded sbatch takes no job arguments. Write artifacts/resume.conf first, then submit:
#   ORIG=artifacts/p-1m            original run directory (its config.json is read)
#   CKPT=artifacts/p-1m/step-030000.pt
#   OUT=artifacts/p-1m-r1          fresh output directory
#   DATA=configs/data-d09-1m.json  the data config the original run used (not in config.json)
# Rebuilds every training flag from the original run's config.json (seed, batch, optimizer,
# schedule horizon, threads, validation cadence) so the runner's exact provenance comparison
# passes. The target step count is the original run's target_steps. --mem/--time above must
# already suit the run; they cannot be overridden at submission.
source jobs/b200/common.sh
test -f artifacts/resume.conf || { echo "write artifacts/resume.conf (ORIG, CKPT, OUT, DATA)" >&2; exit 1; }
source artifacts/resume.conf
orig=${ORIG:?}; ckpt=${CKPT:?}; out=${OUT:?}; data=${DATA:?}
test -f "$orig/config.json" -a -f "$ckpt" -a -f "$data" || { echo "missing input" >&2; exit 1; }
test ! -e "$out" || { echo "output exists: $out" >&2; exit 1; }
job_banner
assert_gpu
mapfile -t flags < <("$PY" - "$orig/config.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); pr = c["provenance"]; o = pr["optimizer"]; s = o["schedule"]
assert o["kind"] == "AdamW" and s["version"] == "warmup-cosine-v1" and c["model"]["device"] == "cuda:0"
for flag, value in [("--seed", pr["seed"]), ("--batch-size", pr["batch_size"]),
        ("--learning-rate", o["learning_rate"]), ("--weight-decay", o["weight_decay"]),
        ("--max-gradient-norm", o["max_gradient_norm"] if o["max_gradient_norm"] is not None else 0),
        ("--warmup-steps", s["warmup_steps"]), ("--schedule-total-steps", s["total_steps"]),
        ("--final-lr-fraction", s["final_fraction"]), ("--threads", c["runtime"]["threads"]),
        ("--validation-every", c["validation_every"]), ("--steps", c["target_steps"])]:
    print(flag); print(repr(value) if isinstance(value, float) else value)
PYEOF
)
echo "resume flags: ${flags[*]}"
"$PY" scripts/train_supervised.py --device cuda:0 --data-config "$data" --output "$out" --resume "$ckpt" "${flags[@]}"
"$PY" scripts/report_step_timing.py "$out"
report_cgroup_peak
