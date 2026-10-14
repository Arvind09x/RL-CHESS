#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=resume
#SBATCH --output=artifacts/slurm/%x-%j.out
# Usage: sbatch [--mem=.. --time=..] jobs/b200/resume.sbatch <original run dir> <checkpoint> <new output dir> <data config>
# Rebuilds every training flag from the original run's config.json (seed, batch, optimizer,
# schedule horizon, threads, validation cadence) so the runner's exact provenance comparison
# passes. The target step count is the original run's target_steps. The data config is not
# recorded in config.json and must be the same one the original run used.
source jobs/b200/common.sh
orig=${1:?original run dir}; ckpt=${2:?checkpoint}; out=${3:?new output dir}; data=${4:?data config}
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
