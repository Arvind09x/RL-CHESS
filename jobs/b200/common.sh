# Sourced by every jobs/b200/*.sbatch. Establishes the repository root, the pinned runtime and
# shared checks. Submit from the repository root so SLURM_SUBMIT_DIR is the root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
test -f pyproject.toml -a -d runtime/cuda-cu128 || { echo "not the repository root: $PWD" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
PY=.venv-cuda-cu128/bin/python
test -x "$PY" || { echo "missing $PY; run bash runtime/cuda-cu128/setup.sh first" >&2; exit 1; }

job_banner() {
  echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION account=$SLURM_JOB_ACCOUNT"
  echo "cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB start=$(date -Is)"
  git rev-parse HEAD
}

# The MIG policy injects the GPU; nothing in the directives requests one. Fail fast if it did not.
assert_gpu() {
  "$PY" -c 'import torch; n=torch.cuda.device_count(); a=torch.cuda.get_arch_list(); print("gpu", torch.cuda.get_device_name(0) if n else None, a); assert n == 1 and "sm_100" in a, (n, a)'
}

# Peak memory of this job step's cgroup v2 (covers the step and its descendants).
report_cgroup_peak() {
  local rel; rel=$(awk -F:: '/^0::/{print $2}' /proc/self/cgroup)
  local path="/sys/fs/cgroup${rel}"
  echo "cgroup=$path memory.peak=$(cat "$path/memory.peak" 2>/dev/null || echo unavailable) end=$(date -Is)"
}
