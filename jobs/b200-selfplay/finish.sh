# Sourced by jobs/b200-selfplay/*.sbatch once `run` has finished every game; needs $PY.
# Each step has its own completion marker: an archive's summary file, then a readable ingest
# report (run_r02.py publishes reports atomically). A resubmission after any stop finishes
# whatever is left; re-ingesting an already loaded game is a recorded no-op.

# Move an unreadable report aside so the ingest it marks is redone.
_drop_unreadable_report() {
  local report=$1
  if test -e "$report" && ! "$PY" -c 'import json, sys; json.load(open(sys.argv[1]))' "$report" 2>/dev/null; then
    mv "$report" "$(dirname "$report")/.$(basename "$report").invalid-$(date +%s)"
  fi
}

# Load archives into one replay buffer, one archive (and one game) at a time.
# Usage: ingest_archives <buffer directory> <campaign directory>...
ingest_archives() {
  local out=$1 dir report create
  shift
  mkdir -p "$out"
  for dir in "$@"; do
    report="$out/ingest-$(basename "$dir").json"
    _drop_unreadable_report "$report"
    if test ! -e "$report"; then
      create=""
      test -e "$out/replay.sqlite" || create="--create"
      "$PY" scripts/run_r02.py --records "$dir/games.jsonl" --buffer "$out/replay.sqlite" $create \
        --report "$report"
    fi
  done
}

# One campaign: archive it, then load it into its own buffer.
finish_campaign() {
  local dir=$1 create=""
  "$PY" scripts/run_selfplay_campaign.py archive --directory "$dir"
  local report="$dir/ingest-report.json"
  _drop_unreadable_report "$report"
  if test ! -e "$report"; then
    test -e "$dir/replay.sqlite" || create="--create"
    "$PY" scripts/run_r02.py --records "$dir/games.jsonl" --buffer "$dir/replay.sqlite" $create \
      --report "$report"
  fi
}
