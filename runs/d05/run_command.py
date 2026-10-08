"""Record actual commands without changing historical task evidence."""
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
name, *argv = sys.argv[1:]
overrides = {'PYTHONPATH': str(ROOT/'src') + ':' + str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}
start = datetime.now(timezone.utc).isoformat()
clock = time.monotonic()
with (OUT/f'{name}.stdout.txt').open('wb') as stdout, (OUT/f'{name}.stderr.txt').open('wb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, env={**os.environ, **overrides}, stdout=stdout, stderr=stderr)
record = dict(argv=argv, cwd=str(ROOT), started_at_utc=start,
              finished_at_utc=datetime.now(timezone.utc).isoformat(),
              duration_seconds=time.monotonic()-clock, exit_code=result.returncode,
              environment_overrides=overrides)
for stream in ('stdout', 'stderr'):
    path = OUT/f'{name}.{stream}.txt'
    raw = path.read_bytes()
    record[stream] = dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
(OUT/f'{name}.command.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(name=name, exit_code=result.returncode, duration_seconds=record['duration_seconds'])))
sys.exit(result.returncode)
