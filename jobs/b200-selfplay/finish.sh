# Sourced by jobs/b200-selfplay/*.sbatch once `run` has finished every game; needs $PY.
# Archive, then load it into the replay buffer. Each step has its own completion marker: the
# archive's summary file, then a readable ingest-report.json (run_r02.py publishes it
# atomically). A resubmission after any stop finishes whatever is left; re-ingesting an
# already loaded game is a recorded no-op.
finish_campaign() {
  local dir=$1 create=""
  "$PY" scripts/run_selfplay_campaign.py archive --directory "$dir"
  local report="$dir/ingest-report.json"
  if test -e "$report" && ! "$PY" -c 'import json, sys; json.load(open(sys.argv[1]))' "$report" 2>/dev/null; then
    mv "$report" "$dir/.ingest-report.json.invalid-$(date +%s)"  # unreadable marker: redo the ingest
  fi
  if test ! -e "$report"; then
    test -e "$dir/replay.sqlite" || create="--create"
    "$PY" scripts/run_r02.py --records "$dir/games.jsonl" --buffer "$dir/replay.sqlite" $create \
      --report "$report"
  fi
}
