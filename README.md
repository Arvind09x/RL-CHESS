# RL-Chess

A reproducible chess reinforcement-learning implementation with an accepted CPU pipeline and a GPU preparation path. The model has about 5.05 million parameters. We have not selected a pretrained baseline P or demonstrated playing strength yet.

CPU work through R04 and bounded Gate I is accepted: board/action contracts, M06 network and losses, supervised data ingestion, checkpoints, search, self-play, replay, mixed updates, and evaluation plumbing. The accepted label bundle contains 9,000 training, 500 validation, and 500 reserved test positions. Gate I establishes that the pieces connect; it is not evidence that learning improves chess strength.

## Development

Use Python 3.12.12 and the committed `uv.lock`:

```sh
uv sync --locked --extra model
uv run --locked --extra model pytest
```

This is the normal test command. Historical M05 network checks explicitly select their M05 configuration; a hidden pytest plugin is no longer required. Production M06 checks exercise the relative-attention model separately.

For an environment without Torch, use `uv sync --locked` and `uv run --locked pytest`. Model tests skip when Torch is absent. `make test` and `make test-adapter` are shortcuts for these two environments. GitHub Actions runs the CPU model suite.

During implementation, run the affected tests. Run the full suite once at integration and in CI. Repeat expensive historical audits only when their inputs or contracts change. Keep the resulting evidence compact: command, source revision, environment, result, and relevant measurements.

## College GPU machine

Follow [the GPU guide](docs/gpu-setup.md). The supported first target is Linux x86-64 with one visible NVIDIA GPU, PyTorch 2.8.0 / CUDA 12.6, float32, deterministic operations, and standard Adam. Driver compatibility and real CUDA execution must be checked on the machine.

Data and weights travel separately from Git. Copy the verified portable data folder, preserving all protected split and provenance files. Paths resolve relative to the configuration file. New runs belong under ignored `artifacts/`.

After the GPU checks, a bounded supervised pilot can be started with:

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-cuda-cu126/bin/python scripts/train_supervised.py   --device cuda:0 --data-config artifacts/data/data-config.json   --output artifacts/pilot-001 --steps 100 --batch-size 16 --validation-every 50
```

The runner records training/validation losses, validation policy accuracy, throughput, GPU memory, and resumable checkpoints. It never opens the reserved test split or designates P automatically. Choose the actual pilot size and batch size after measuring the college GPU.

## Project map

- `src/chess_rl/adapter/`: board history, features, and actions.
- `src/chess_rl/model/`: network implementation; `src/chess_rl/training/`: losses, supervised/mixed updates, and checkpoint ownership.
- `src/chess_rl/search/`, `selfplay.py`, `replay.py`: legal search, recoverable games, and replay admission.
- `supervised.py`, `stockfish_worker.py`: validated labels and teacher execution.
- `scripts/`: data preparation, annotation, GPU checks, pilot training, and evaluation tools.
- [Current milestones](coding-plan.md), [GPU implementation details](docs/cuda-support.md), and [evidence recovery](docs/evidence-archive.md).

The immediate next work is to verify CUDA on the college machine, measure supervised/search throughput, train and select P using validation, then evaluate mixed-training R against P with a frozen budget. Main training and the final held-out report follow those decisions.

## Evidence and artifacts

Historical run evidence is preserved in Git history and a verified external archive. The active source tree retains small provenance fixtures needed by tests. Bulk datasets, weights, raw logs, and future runs are ignored. Historical documents remain under `docs/history/`; their original milestone decisions are not rewritten as new results.

Use institutional storage for data/checkpoint backups. Ignoring files prevents new accidental additions; it does not remove information already present in Git history.
