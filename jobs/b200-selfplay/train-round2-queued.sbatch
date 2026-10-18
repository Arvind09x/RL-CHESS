#!/bin/bash
#SBATCH --partition=gpu-small
#SBATCH --account=student
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=train-round2
#SBATCH --chdir=projects/RL-Chess
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --dependency=afterok:@ROUND2_JOB@
#SBATCH --kill-on-invalid-dep=yes
# OPTION, not yet tested on the cluster: run train-round2.sbatch automatically once selfplay-round2
# ends successfully. A template: fill in @ROUND2_JOB@ (the running self-play job id) and @PIN@ (the
# commit that contains train-round2.sbatch), save the result OUTSIDE the repository (for example
# ~/train-round2-queued.sbatch, so a fast-forward never meets an untracked file), and submit it from
# the home directory: --chdir is relative to the submission directory. Works only if the cluster
# accepts a pending second job (the QoS may limit submitted, not only running, jobs) and the guarded
# sbatch accepts this script; a batch job must also be allowed to run git and reach GitHub.
# Fails closed: if the fetch, the fast-forward or the HEAD check fails, nothing runs.
set -euo pipefail
PIN=@PIN@
case "$PIN" in *@*) echo "template not filled in" >&2; exit 1;; esac
echo "job=$SLURM_JOB_ID queued after @ROUND2_JOB@; node=$(hostname) start=$(date -Is) dir=$PWD"
test -f pyproject.toml -a -d .git || { echo "not the repository: $PWD" >&2; exit 1; }
git fetch origin
git merge-base --is-ancestor HEAD "$PIN" || { echo "$PIN is not a fast-forward of $(git rev-parse HEAD)" >&2; exit 1; }
git merge --ff-only "$PIN"
test "$(git rev-parse HEAD)" = "$PIN" || { echo "HEAD $(git rev-parse HEAD) != $PIN" >&2; exit 1; }
git status --porcelain --untracked-files=no | grep -q . && { echo "tracked files are modified" >&2; exit 1; }
test -f jobs/b200-selfplay/train-round2.sbatch || { echo "train-round2.sbatch missing at $PIN" >&2; exit 1; }
# common.sh takes the repository root from SLURM_SUBMIT_DIR.
export SLURM_SUBMIT_DIR="$PWD"
exec bash jobs/b200-selfplay/train-round2.sbatch
