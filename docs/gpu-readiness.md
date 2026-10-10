# GPU preparation verification

This work reconciles accepted CPU main `62af7ac` and CUDA preparation `a97cf31`. It fixes the ordinary test entry point, removes repeated hot-path validation, adds portable supervised execution and CUDA mixed/self-play support, and keeps bulk evidence outside the active source tree. It does not accept a CUDA run, choose P, or establish playing strength.

## Local checks

On macOS arm64, Python 3.12.12, NumPy 2.5.3, and Torch 2.8.0:

- Full model suite: **1,593 passed, 1 skipped** (the hardware CUDA gate).
- Torch-free suite: **1,028 passed, 17 expected skips**.
- Ordinary historical network checks passed without the diagnostic plugin; M06 production checks passed separately in the full suite.
- Separate-process CPU update/save/resume smoke: passed, continuation maximum absolute error **0.0**.
- Supervised pilot CLI: exact uninterrupted-versus-resumed checkpoint contents verified in independent processes by the operational suite.
- CPU preflight: actual network forward passed; parameter count **5,049,411**.
- Lock consistency checked offline; CUDA setup script syntax checked.
- Portable staging fixture: exact copies and relocation passed, including a targeted rerun after the reference-only staging optimization.

The pinned Stockfish executable also completed two fresh-process searches and two searches through one persistent worker. Reset/health evidence was captured and best-move/WDL targets matched on these two positions. This is a bounded functional smoke, not a production throughput or label-equivalence benchmark.

## Transfer bundle

The accepted production bundle is staged in the project workspace's external `external-artifacts/gpu-readiness-data/` directory. Its `data-config.json` uses relative paths. The staged authoritative files were hash-checked before/after copying and reopened by the production loader. Membership is 9,000 train / 500 validation / 500 reserved test labels. Total copied file bytes: **1,558,826,189**.

[portable-data.json](portable-data.json) retains only portable per-file identities and counts. Private source paths, data, checkpoints, and smoke output remain external or ignored. The Git checkout alone cannot train: the transfer bundle must accompany it.

## Remaining gate and workflow

Follow [gpu-setup.md](gpu-setup.md) on the allocated college machine. Driver compatibility, real CUDA numerical/continuation checks, and GPU throughput/memory are pending. Then measure a bounded supervised pilot, select P on validation, and carry out P03 evaluation before scaling.

Run focused tests while editing and one full integration pass plus CI. Re-audit historical campaigns when a relevant contract changes. Historical scripts such as `run_r03.py`, `run_r04.py`, `check_t02.py`, and D06/D08 acceptance scripts retain their original artifact-bound contracts; they require restored historical evidence and are not the portable pilot entry point. Their old accepted outcomes remain historical evidence rather than new GPU results.

No new independent review was commissioned. These are local implementation checks, with the hardware gate explicitly pending.
