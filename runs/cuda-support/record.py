"""Record exact commands/output without changing accepted historical evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[0] == "--" else a.command
    env = dict(PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1")
    sources = {str(path.relative_to(ROOT)): identity(path) for folder in ("src", "tests", "scripts")
        for path in (ROOT / folder).rglob("*.py")}
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **env), capture_output=True)
    # JSON strings retain exact output, including trailing whitespace/newlines.
    for stream in ("stdout", "stderr"):
        (OUT / f"{a.name}.{stream}.json").write_text(json.dumps(
            getattr(result, stream).decode("utf-8"), ensure_ascii=True) + "\n")
    report = dict(argv=command, cwd=str(ROOT), env=env, started_utc=started,
        seconds=time.monotonic() - begin, exit_code=result.returncode, sources=sources,
        source_unchanged=all(identity(ROOT / name) == info for name, info in sources.items()),
        output={stream: identity(OUT / f"{a.name}.{stream}.json") for stream in ("stdout", "stderr")})
    (OUT / f"{a.name}.command.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(a.name, result.returncode, report["seconds"], result.stdout.decode()[-1000:])
    raise SystemExit(result.returncode)


if __name__ == "__main__": main()
