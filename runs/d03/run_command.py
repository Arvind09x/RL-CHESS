"""Capture actual argv, exit codes and lossless stdout/stderr byte identities."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def run(name, argv):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True)
    streams = {}
    for label, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
        path = OUT / f"{name}.{label}.txt"
        path.write_bytes(raw)
        streams[label] = dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    record = dict(argv=argv, cwd=str(ROOT), started_at_utc=started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  duration_seconds=time.monotonic()-begin, exit_code=result.returncode,
                  environment_overrides={k: env[k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
                  **streams)
    (OUT / f"{name}.command.json").write_text(json.dumps(record, indent=2)+"\n")
    print(name, result.returncode, result.stdout.decode(errors="replace")[-400:], flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2:]))
