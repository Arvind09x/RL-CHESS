#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=smoke-100k
#SBATCH --output=artifacts/slurm/%x-%j.out
# Pipeline shakedown on bundle-100k: 300 updates, then a resume to 400 in a fresh directory with
# identical settings (the runner compares provenance exactly). Short warmup so the cosine phase
# runs too. Reports steady-state step time and GPU utilisation for projecting the 1M run.
source jobs/b200/common.sh
job_banner
assert_gpu
run="artifacts/smoke-100k/$SLURM_JOB_ID"
mkdir -p "$run"
common=(--device cuda:0 --data-config configs/data-d09-100k.json --batch-size 32 --threads 4
        --learning-rate 1e-4 --weight-decay 0.01 --max-gradient-norm 10 --warmup-steps 50
        --schedule-total-steps 400 --validation-every 100 --seed 20261010)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$run/gpu-util.csv" &
sampler=$!
"$PY" scripts/train_supervised.py "${common[@]}" --output "$run/a" --steps 300
"$PY" scripts/train_supervised.py "${common[@]}" --output "$run/b" --steps 400 --resume "$run/a/step-000300.pt"
kill "$sampler" 2>/dev/null || true
"$PY" scripts/report_step_timing.py "$run/a" --project-steps 62500 --project-validation-every 500
"$PY" scripts/report_step_timing.py "$run/b"
# nvidia-smi reports utilization.gpu as N/A on MIG instances; memory.used is still available.
"$PY" - "$run" <<'PYEOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/gpu-util.csv")))
def numbers(column):
    out = []
    for r in rows:
        token = r.get(column, "").strip().split(" ")[0]
        try: out.append(float(token))
        except ValueError: pass
    return out
util, mem = numbers(" utilization.gpu [%]"), numbers(" memory.used [MiB]")
print(f"gpu samples={len(rows)} utilisation=" + (f"mean {statistics.fmean(util):.1f}% max {max(util):.0f}%" if util
      else "unavailable (N/A on MIG)") + (f" memory.used max {max(mem):.0f} MiB" if mem else ""))
PYEOF
report_cgroup_peak
