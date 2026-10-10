# Current project plan

This is the active plan. The original detailed plan is preserved byte-for-byte in [docs/history/coding-plan.md](docs/history/coding-plan.md).

| Stage | Status | What this establishes |
| --- | --- | --- |
| F / A: foundations and adapter | Accepted on CPU | Legal history, action mapping, and features |
| M / T: network and training primitives | Accepted on CPU | M06 architecture, losses, Adam updates, checkpoint continuation |
| D: data and Stockfish label pipeline | Accepted CPU artifacts | Protected game splits and 9,000 / 500 / 500 labels |
| S / R through R04 | Accepted on CPU | Search, durable self-play, replay, mixed updates, and evaluation plumbing |
| E01 and bounded Gate I | Accepted CPU fixture evidence | End-to-end connectivity; no strength claim |
| GPU preparation and workflow cleanup | Implemented; hardware verification pending | Portable runtime/data, CUDA supervised/mixed support, ordinary tests, compact artifacts |
| P01: college GPU checks and profiling | Pending hardware access | Driver/runtime compatibility, CUDA continuation, practical throughput/memory |
| P02: supervised pretrained P | Pending | Validation-selected and reproducible baseline |
| P03: pilot R versus P | Pending | Bounded learning run and frozen-budget evaluation |
| P04: Gate P | Pending | Evidence-based decision on scaling |
| L / remaining E / Z | Pending Gate P | Main training, final evaluation, report/demo |

## Working sequence

1. Run [college GPU setup and checks](docs/gpu-setup.md). Record the source revision, environment, GPU, results, and failure details if any.
2. Measure supervised batches and search/self-play throughput on that hardware. Choose a batch size and bounded pilot budget from those measurements. Independent-game batching remains future optimization if measurements justify it.
3. Train supervised candidates with `scripts/train_supervised.py`. Select P using validation; keep the test split reserved. Record the exact checkpoint bytes and configuration.
4. Run bounded self-play and mixed training, then evaluate R versus P with frozen openings, seeds, search settings, and game budget. CUDA plumbing does not itself establish a successful pilot.
5. Decide Gate P from learning curves, strength evidence, resource cost, and reproducibility. Scale the main run only after this decision.
6. Use the reserved test split and final evaluation protocol for the report/demo.

## Verification policy

Run focused tests while changing a component. Run the normal full CPU model suite once before integrating code; CI repeats it for the resulting revision. Run the Torch-free suite when shared adapter/data/replay code changes. Run CUDA smoke and CUDA-marked tests on the GPU host after relevant device or checkpoint changes.

Historical accepted evidence remains accepted until a changed contract invalidates it. Do not rerun every historical campaign at each small edit. Create one compact summary per experiment and keep bulk evidence under ignored artifact storage. Never mark a stage accepted merely because its tests pass: pilots also require the stage's actual training/evaluation evidence.
