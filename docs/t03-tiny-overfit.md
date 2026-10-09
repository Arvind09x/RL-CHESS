# T03 tiny fixed training fit — 9 October 2026

Implemented and executed once in the isolated `t03-tiny-overfit` clone on
`codex/t03-tiny-overfit`, based on integrated main
`4fa48d4afef91a3bea125fb5f9f85b4d5159fc28`. T02 commit
`27731e32d8aa64d852b439945f30e7238e87a824` has independent AI review acceptance
and integration evidence in the inherited `docs/t02-integration.md` and primary
integration receipt. No new independent reviewer, human review or subagent is
claimed. T03 automated checks are complete; independent review remains pending.

The exact preexecution [frozen plan](../runs/t03/frozen-plan.json) has SHA-256
`bce4268c83e094fe204c45dd94dc8d9ccb3f7b4c71f483be3ccf4d76aa9ddc09`.
It records complete game/ply references, teacher class/WDL targets, feature/mask
hashes, accepted input identities, tested source identities, runtime, optimizer,
losses, gates, cap and output location. Its timestamp precedes execution-start;
the execution receipt copies the exact plan and identity before constructing the
model. Source and accepted input bytes are checked again before execution and
fresh-process reload. A new plan or existing output directory cannot be overwritten.

## Frozen experiment

The first eight D07 source-order labels from accepted `runs/d05/tiny-final` are
used: game `g:000025602b7ecc9da7677d8baf42cad26dae7815ec1145a18af1c2792cbebb34`
at plies 5, 9, 14, 20, 25, 30 and 41, and game
`g:00012298940f898babde41099e7c8970e735bfc5e7077d23e22cb042375211c3` at ply 14.
All 12 source labels are training labels; four remain unused. D07 verifies the
original accepted bundle and replay provenance, then constructs one owned eight
sample batch. The same batch is repeated without shuffling or augmentation.
No validation/test sample is loaded into a batch or used in fitting, stopping,
measurement or checkpoint selection. D07 does read the accepted held-out
registries to establish exclusion; those are provenance checks, not training data.

The combined M06 network uses all GAB + Q/K/V relative terms, default 128-wide
heads and 5,049,411 unique parameter elements. All dropout probabilities are
zero; shared GAB bank identity is retained. Python/NumPy/Torch seed is 20261009.
Pinned runtime: macOS arm64 CPU float32, Python 3.12.12, Torch 2.8.0, NumPy 2.5.3,
one intra-op and one inter-op thread, deterministic algorithms enabled.

Adam settings are fixed at learning rate 0.0003, betas (0.9, 0.999), epsilon
1e-8, weight decay 0, AMSGrad false, foreach false and fused false. There is no
scheduler, clipping, gradient accumulation or tuning. T01 `supervised_step`
and its shared legal class-policy CE and soft root-side WDL CE are reused with
weights 1:1. Accepted model, loader, tensor helpers, losses/step and checkpoint
implementations are unchanged.

Measure the fixed batch at initialization and after every completed update.
The first update satisfying every gate stops the run:

| Frozen requirement | Threshold |
|---|---|
| Legal best-move accuracy | At least 1.0 (8/8) |
| Mean KL(teacher WDL || model WDL) | At most 0.01 nats |
| Maximum per-position teacher-WDL KL | At most 0.03 nats |
| Mean CE versus entropy + KL identity error | At most 0.000002 nats |
| Minimum KL numerical tolerance | At least -0.000002 nats |
| Completed optimizer update cap | 1,000 |
| Fresh-process checkpoint reload tolerance | Exactly zero |

Legal argmax uses the shared masked normalization; vocabulary order breaks ties.
Metrics compute entropy and KL in float64 using the D07 float32 targets exactly
as supplied. Zero teacher mass contributes zero; targets are not hardened or
renormalized and KL is not clamped. WDL CE remains the accepted T01 computation.
The numerical identity tolerance is separate from the fit thresholds.

The runner requires at least one update, writes and flushes every measured row,
and retains a final checkpoint even on `failed_step_cap`. It never adjusts a
gate after execution. Unit checks exercise cap failure, pass at the cap, first
pass stopping, each mandatory threshold, NaNs, illegal targets, zero teacher
entries, independent NumPy entropy/KL/CE and nonzero teacher entropy at perfect
WDL fit.

## Actual results and saved artifacts

The single experiment passed at update 26, before the cap. Step 25 failed both
KL gates, so step 26 is independently checked as the first passing update.
Fit/update measurement took 8.920 seconds; final checkpoint and expected-state
publication brought the recorded training phase to 9.231 seconds. The complete
run including D07 opening and separate-process reload took 40.160 seconds.
These timings are local execution measurements, not throughput projections.

| Metric | Baseline | Step 26 |
|---|---:|---:|
| Legal best-move accuracy | 1/8 (12.5%) | 8/8 (100%) |
| Policy CE | 3.246093273 | 0.001303019 |
| Mean teacher-WDL KL, nats | 0.817264293 | 0.009595421 |
| Maximum teacher-WDL KL, nats | 1.037374971 | 0.021492420 |
| WDL cross-entropy, nats | 1.172104001 | 0.364435166 |
| Teacher entropy floor, nats | 0.354839762 | 0.354839762 |
| WDL CE minus entropy, nats | 0.817264239 | 0.009595404 |

Final entropy identity error is `1.7557722831185174e-08` nats. WDL loss is compared
to the teacher entropy floor, not to zero. All per-position KL values, legal
predicted IDs, post-update metrics and pre-update T01 step losses are retained
at baseline and every update: 27 rows in [JSONL](../runs/t03/learning-curves.jsonl),
[CSV](../runs/t03/learning-curves.csv) and a [static curve plot](../runs/t03/learning-curves.svg).
The plot is generated from the raw rows with a dependency-free renderer; its
SVG was rasterized and visually checked. The bundled runtime lacks matplotlib;
no dependency pins were changed or packages installed.

T02 saves the full M06 model, standard Adam, RNG, frozen provenance, loss weights
and `Progress(step=26, epoch=0, samples_seen=208)`. DataOrder binds the entire
12-label D07 training source, source-order permutation and cursor 8. Epoch zero
is retained because this is repeated fixed-batch fitting, not an epoch sweep;
the eight-label repetition schedule and repetition count are explicit in
provenance. Generic dataset-epoch resumption is not claimed for this experiment.

The 60,883,331-byte checkpoint is external to Git at
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/t03-tiny-overfit-20261009/final.pt`.
SHA-256: `d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`.
[External inventory](../runs/t03/external-artifacts.json) hashes every retained
artifact, including execution-start, expected model/Adam hashes and raw curves.
The external directory is within the user workspace, rather than temporary storage.

A separate Python process scrambles all three RNGs before T02 inference loading
and training restoration. [Reload verification](../runs/t03/reload-verification.json)
checks exact model tensor hashes, Adam state hashes, final metrics, construction
configuration, expected provenance, loss weights, data membership/order,
completed progress and shared GAB identity. No additional update is executed.
Checkpoint bytes remain unchanged. `result.json` preserves the at-publication
`fresh_process_reload: pending` field; the separate reload record and final
[summary](../runs/t03/summary.json) record the successful completed verification.

## Relevant checks, preservation and review boundary

| Check | Result |
|---|---|
| T03/D07/training/checkpoint focused tests | 299 passed, including 20 new T03 cases |
| Full model regressions | 1,364 passed |
| Torch-free adapter regressions | 1,023 passed; 10 expected optional-Torch skips |
| Accepted historical review probes | 84 passed |
| Frozen adapter replay | All 13 fixtures byte-identical to accepted T02 integration |
| M06 numerical checker | Passed |
| Both pinned environments, offline dependency compatibility | Passed |
| Fresh-process inference/training reload | Exact, zero tolerance |

Full model checks use the inherited `runs.m06.diagnostic_plugin` for historical
M05 constructor tests, as documented in accepted M06/T02 evidence. The real T03
fit uses the production combined relative-enabled architecture. Exact argv,
environment overrides, timing, tested source hashes, exit codes and lossless
stdout/stderr containers are in `runs/t03/*.command.json`; JUnit records are
retained. There was exactly one T03 fitting execution and no threshold revision.

The [preservation records](../runs/t03/preservation-after.json) compare all 1,858
inherited files, external canonical/prepared trust roots and frozen plan. Only
dated README/completion-log appends are allowed. All accepted implementation,
historical evidence, manifests, shared interface, strategy and protected
held-out registries remain exact. Primary main and its original uncommitted
D06/D07/E01 plan edits remain untouched. No applicable AGENTS.md was found.

To inspect or reproduce, use the pinned model interpreter with `PYTHONPATH=src:.`
from this checkout. `scripts/run_t03.py freeze --directory <new-path>` publishes
the plan once, `scripts/run_t03.py run` executes once, and `verify` reads the
saved checkpoint. Reproduction requires a fresh checkout/output identity so the
existing frozen plan and artifacts remain intact. Relevant test commands and
original successful execution commands are retained in the command records.

T03 establishes plumbing on eight fixed training roots. This checkpoint is not
immutable pretrained P and establishes neither generalization nor playing
strength. T03 remains unchecked until independent review. Stop after this
isolated commit: T04, pilot/pretraining, model-guided games and RL were not begun.
