"""Independent review evidence: exact branch checks and combined regressions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = "/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python"
MODEL = "/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python"

def run(name, argv, cwd=ROOT):
    env = dict(os.environ, PYTHONPATH=f"{cwd}/src:{cwd}", PYTHONDONTWRITEBYTECODE="1")
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    (OUT / f"{name}.stdout.txt").write_text(result.stdout)
    (OUT / f"{name}.stderr.txt").write_text(result.stderr)
    record = dict(argv=argv, cwd=str(cwd), started_at_utc=started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  duration_seconds=time.monotonic()-begin, exit_code=result.returncode,
                  environment_overrides={k: env[k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
                  stdout=f"{name}.stdout.txt", stderr=f"{name}.stderr.txt")
    (OUT / f"{name}.command.json").write_text(json.dumps(record, indent=2)+"\n")
    print(name, result.returncode, result.stdout[-300:], flush=True)
    return record

if __name__ == "__main__":
    jobs = [
        ("d02-exact-focused", [PYTHON, "-m", "pytest", "tests/test_pgn_replay.py", "tests/test_data_manifest.py", "-p", "no:cacheprovider", f"--junitxml={OUT}/d02-exact.xml"], Path("/private/tmp/rl-chess-d02")),
        ("s02-exact-focused", [PYTHON, "-m", "pytest", "tests/test_puct.py", "-p", "no:cacheprovider", f"--junitxml={OUT}/s02-exact.xml"], Path("/private/tmp/rl-chess-s02")),
        ("combined-adapter", [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={OUT}/combined-adapter.xml"], ROOT),
        ("combined-model", [MODEL, "-m", "pytest", "-p", "no:cacheprovider", "-p", "runs.m06.diagnostic_plugin", f"--junitxml={OUT}/combined-model.xml"], ROOT),
        ("archive-output-audit", [PYTHON, "runs/d02/audit_outputs.py", "/private/tmp/rl-chess-d02-accepted-prepared"], ROOT),
        ("replay-adapter", [PYTHON, "scripts/check_adapter.py"], ROOT),
        ("replay-model", [MODEL, "scripts/check_adapter.py"], ROOT),
        ("teacher-bytes", [PYTHON, "runs/d02/teacher_bytes.py"], ROOT),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda job: run(*job), jobs))
    (OUT / "commands.json").write_text(json.dumps(records, indent=2)+"\n")
    assert all(record["exit_code"] == 0 for record in records)
    assert json.loads((OUT/"replay-adapter.stdout.txt").read_text()) == json.loads((OUT/"replay-model.stdout.txt").read_text())
