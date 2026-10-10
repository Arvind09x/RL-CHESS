# GPU preparation and operational changes

This change reconciles CUDA support from `a97cf31` with accepted CPU main `62af7ac`, including R03/R04. The original CUDA preparation record is preserved in [history/cuda-support.md](history/cuda-support.md). CPU verification and hardware CUDA verification are separate; local CPU results do not establish GPU acceptance.

## Runtime and checkpoints

The initial GPU contract is one visible `cuda:0`, float32, deterministic operations, TF32 disabled, no AMP, standard Adam with `foreach=False` and `fused=False`, and a pinned cu126 environment. Models initialize from the existing CPU initializer before transfer. Data, masks, targets, and optimizer moments follow the execution device; Adam scalar steps stay on CPU.

Legacy CPU schemas `m06-checkpoint-v1` and `m06-checkpoint-v2-mixed` retain their contracts. CUDA supervised uses `m06-checkpoint-cuda-v2`; CUDA mixed uses `m06-checkpoint-cuda-v3-mixed`. CUDA saves CPU-owned tensor snapshots plus device RNG/runtime metadata. CPU inference can read these files; training restoration requires matching backend, data, cursor, optimizer settings, and runtime. Restoration installs RNG state only after validation succeeds.

Exact runtime matching remains the default. The opt-in compatible policy relaxes only OS descriptions and Python patch versions, while preserving architecture, numerical-library versions, deterministic settings, and CUDA runtime checks. It warns when exact continuation has not been established.

## Avoiding repeated work

Frozen supervised views own immutable metadata and a bounded decoded-sample cache. References and selected batches no longer encode all previous samples. Mixed sources validate full membership at admission and validate ownership in constant time during updates.

Search copies already-owned board history without replaying it through the public constructor at every node. Public state/node admission stays validated. Board-stack copying still scales with history length.

Replay keeps an owned SQLite connection and reuses completed validation when the database signature is unchanged. External writes or corruption invalidate this cache; WAL use disables it. Newly admitted games still pass complete validation. Export still materializes the requested snapshot.

Self-play CLI sessions append immutable per-move files before atomically publishing the boundary pointer. They avoid rewriting earlier dense targets on each move. Legacy records remain readable and exported replay snapshots retain their existing schema. CUDA training batches aggregate finite checks to reduce device/host synchronization.

## Teacher execution

Fresh Stockfish processes remain the default. An explicit `--engine-policy persistent-clear-hash-v1` on `annotate_d05.py` reuses a sequential worker. Every search starts with a new-game/reset/readiness sequence and clear hash; errors discard the process. D07 admission checks this policy's raw UCI evidence and successful health check. The policy is recorded in label provenance.

This changes the execution policy, so create a new annotation run rather than appending to an accepted historical run. No teacher speedup or label-equivalence benchmark is claimed. Historical D06/D08 acceptance is unchanged.

## Verification and remaining work

Run ordinary CPU tests without a hidden plugin. New operational tests cover immutable ownership, bounded selected-batch work, database external writes, journal crash recovery, portable paths, and fresh-process pilot continuation. Hardware-gated CUDA tests and a separate-process smoke are provided in the [setup guide](gpu-setup.md).

The remaining hardware gate is to run those checks on the college machine, profile training/search throughput and memory, and then execute the planned P01/P02/P03 work. No pretrained P, completed GPU pilot, or playing-strength result is implied by this preparation.
