"""Replay final lock payload and compare retained continuation tensors, no updates."""
import json
from pathlib import Path
import torch
from scripts.check_cuda_support import compare, TOLERANCES
from scripts.stage_cuda_data import identity

ROOT = Path(__file__).resolve().parents[2]
lock = ROOT/"runtime/cuda/requirements.lock"
replay = Path("/private/tmp/rl-chess-cuda-requirements-replay.lock")
def payload(path):
    return "\n".join(line for line in path.read_text().splitlines() if not line.startswith("#"))
assert payload(lock) == payload(replay)
assert "torch==2.8.0+cu126" in lock.read_text() and "chess==1.11.2" in lock.read_text()
reports = {}
for name in ("cuda-support-cpu", "cuda-support-portable-cpu"):
    directory = ROOT.parent/"external-artifacts"/name
    a, b = [torch.load(directory/f"{kind}.pt", weights_only=True) for kind in ("uninterrupted", "resumed")]
    differences = []
    compare(a, b, TOLERANCES["reload"], differences)
    reports[name] = dict(final_comparator_exact=True, maximum=max(differences),
        summary=json.loads((directory/"summary.json").read_text()))
compare(1., 1. + 5e-7, TOLERANCES["continuation"], [])
try: compare(1., 1. + 5e-7, TOLERANCES["reload"], [])
except AssertionError: pass
else: raise AssertionError("exact comparator accepted a changed metric")
result = dict(cuda_lock=identity(lock), replay_payload_identical=True,
    continuation_reports=reports, updates_performed=0)
(ROOT/"runs/cuda-support/lock-report-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print("Final lock payload replay identical; final comparator verified exact CPU continuation tensors; zero updates")
