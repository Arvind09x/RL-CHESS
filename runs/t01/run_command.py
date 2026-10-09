"""Record exact command timing, environment, source hashes and lossless outputs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
directory = root / 'runs/t01'
name, *command = sys.argv[1:]
env = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root / 'src') + ':' + str(root),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
def sources():
    paths = [root/'src/chess_rl/tensor_ops.py', root/'tests/test_training.py',
             root/'tests/fixtures/loss_reference.py', root/'scripts/check_t01.py', directory/'config.json',
             *sorted((root/'src/chess_rl/training').glob('*.py'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = sources()
started = datetime.now(timezone.utc).isoformat()
tick = time.monotonic()
result = subprocess.run(command, cwd=root, env=os.environ | env, capture_output=True)
elapsed = time.monotonic() - tick
outputs = {}
for label, raw in [('stdout', result.stdout), ('stderr', result.stderr)]:
    path = directory / f'{name}.{label}.txt'
    path.write_bytes(raw)
    outputs[label] = dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
record = dict(argv=command, cwd=str(root), environment_overrides=env, started_at_utc=started,
    finished_at_utc=datetime.now(timezone.utc).isoformat(), duration_seconds=elapsed,
    exit_code=result.returncode, outputs=outputs, tested_source_sha256=before,
    tested_source_unchanged_during_command=before == sources())
(directory / f'{name}.command.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
if before != sources(): raise RuntimeError('sources changed during command')
raise SystemExit(result.returncode)
