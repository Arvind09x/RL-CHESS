"""Lossless command receipts for bounded R03 evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'runs/r03'

def sources():
    paths = [p for directory in ('src','tests','scripts') for p in (ROOT/directory).rglob('*.py')]
    paths += [ROOT/p for p in ('pyproject.toml','uv.lock','runs/r03/frozen-plan.json','runs/r03/run_command.py')]
    return {str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(paths)}

def main():
    name,argv = sys.argv[1],sys.argv[2:]
    receipt_path = OUTPUT/(name+'.command.json')
    if receipt_path.exists(): raise ValueError('refuse replacing command evidence')
    environment = dict(PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    before = sources()
    start = datetime.now(timezone.utc).isoformat(); tic = time.monotonic()
    result = subprocess.run(argv,cwd=ROOT,env=dict(os.environ,**environment),capture_output=True)
    elapsed = time.monotonic()-tic
    stdout = result.stdout.decode(); stderr = result.stderr.decode()
    for label,value in (('stdout',stdout),('stderr',stderr)):
        (OUTPUT/(name+'.'+label+'.json')).write_text(json.dumps(value,ensure_ascii=False)+'\n')
    after = sources()
    receipt = dict(argv=argv,cwd=str(ROOT),environment=environment,started_at_utc=start,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),duration_seconds=elapsed,
        exit_code=result.returncode,source_hashes_before=before,source_hashes_after=after,
        sources_unchanged=before==after,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
    receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(stdout,end=''); print(stderr,end='',file=sys.stderr)
    if before != after: raise ValueError('source drift during evidence command')
    raise SystemExit(result.returncode)

if __name__ == '__main__': main()
