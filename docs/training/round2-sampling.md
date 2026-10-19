# R2 self-play sampling

New R2 runs default to proportional self-play sampling in the ordered
`--round-data` inputs `[R1, R2]`:

| Recipe | R1 | R2 |
| --- | ---: | ---: |
| mix | 0.20 | 0.80 |
| pure | 0.00 | 1.00 |

`--round-weights 1 3` overrides these with 25% / 75%. Supply exactly one finite,
nonnegative weight per ordered input; weights are normalized, need not sum to
one, and cannot all be zero. Defaults require exactly two inputs; explicit
weights support additional rounds without adding orchestration.
`--sampling-mode legacy` explicitly uses the original uniform union sampler.
It cannot be combined with weights. This is sampling with reuse, not data filtering.
All existing source, duplicate, checkpoint and train/holdout checks still apply,
including to zero-weight rounds. Evaluation still measures the same complete
holdout union and per-round slices; candidate rules are unchanged.

Only self-play sampling changes. Mix still uses 24 self-play and 8 Stockfish
positions per update, with 75% / 25% loss weighting. Pure still trains on the
24 self-play positions alone and discards its 8 Stockfish indices. Matching
weights, seeds and budgets give mix and pure identical self-play draws; the
new differing defaults intentionally do not.

## Allocation and budget

Sampler `round-deadline-cycles-v1` selects the round with the smallest next
virtual deadline `(round_draws + 1) / round_weight`, breaking ties by input order.
The implementation uses integer ratios of the supplied decimal float values,
so allocation has no floating-point accumulation drift. Counters never reset at
batch or checkpoint boundaries. With two active rounds, each cumulative count
is less than one draw from its target at every draw prefix. More generally, the
absolute discrepancy is strictly less than `max(1, active_rounds - 1)` draws;
one active round is exact. Arbitrary proportions are not exact in every
24-position batch. This deadline method can put larger-weight rounds first;
it does not randomize round selection.

Each round has its own shuffled permutation and RNG, seeded with
`round-deadline-cycles-v1:<sampling_seed>:<zero_based_input_index>`. Draw without
replacement until exhaustion, then reshuffle deterministically. A round may
wrap within a batch. Zero-weight rounds contribute no loss positions.

New weighted runs require explicit positive `--updates N`: at most N optimizer
updates and 24N self-play draws, subject to the existing hard stop. No weighted
union epoch is inferred. Use the same N for both recipes and for uniform
comparisons; N is not R1's old update count. Legacy mode without N retains the
old `ceil(union_training_positions / 24)` budget.

The existing MixedState runs alongside round allocation to preserve the exact
legacy Stockfish sequence, including its shared RNG's union reshuffles. Its
self-play permutation is bookkeeping only; those shadow draws never enter the
loss. This adds one round-local permutation per input alongside the existing
union permutation, plus round RNG states. No extra model work is performed.

## Resuming and reporting

Omitted sampling flags on resume inherit the saved configuration, including
legacy runs without any `round_sampling` field. Omitted N inherits the saved
explicit budget. Explicit requests must match saved settings. Changing mode,
weights, ordered round identities, seed or budget requires a distinct output
directory; completed directories are checked too. Equivalent but differently
expressed weights (for example `[1,3]` versus `[.25,.75]`) are rejected on resume
because the original requested weights are part of provenance. Unknown sampler
versions fail closed. Legacy state and provenance are not rewritten.

Checkpoint state includes local permutations, cursors, RNG states, allocation
counts, the shadow legacy sampler and versioned configuration. Provenance,
metrics, sidecars and summaries record requested/normalized weights, ordered
round replay identities and the total budget. Per-round reports give actual
draws, unique positions seen, reused draws, effective passes and oversampling.
These are exact for the without-replacement cycles. `trained_positions` remains
separate from checkpoint `samples_seen` bookkeeping (32 per update even for
pure). Weighted reports use null for union epochs and the legacy `seen_twice`
field; `reused_draws` covers arbitrary reuse instead.

## Reviewable configurations (not launched)

`jobs/b200-selfplay/train-round2.sbatch` requires `R2_UPDATES` to be set to a
reviewed positive integer. The job gives both recipes that same budget. It uses
the defaults above unless overridden by `R2_MIX_WEIGHTS`, `R2_PURE_WEIGHTS` or
`R2_SAMPLING_MODE=legacy`. For a 25% R1 / 75% R2 experiment in both recipes:

```bash
# Set R2_UPDATES to the agreed total N before running the job.
export R2_SAMPLING_MODE=weighted
export R2_MIX_WEIGHTS='0.25 0.75'
export R2_PURE_WEIGHTS='0.25 0.75'
```

For the equal-budget baseline, retain the identical `R2_UPDATES` and set:

```bash
export R2_SAMPLING_MODE=legacy
```

The job uses distinct `r2-mix-<settings-hash>` / `r2-pure-<settings-hash>` output
directories; the hash includes sampler version, mode, both requested weight
vectors and N. Existing `r2-mix` / `r2-pure` directories are not reused. Before
full training, both recipes must pass their own four-update CUDA interruption/
resume comparison and inference check. Their dry-run directories and pass
markers use the same settings hash, so a legacy `passed.json` cannot satisfy
the weighted gate. Markers record each recipe's resolved sampler and draw report.

For direct CLI use, keep the checkpoint, source and device arguments from the
job's `COMMON` array and use these argument tails:

```bash
# Proportional experiment:
train --recipe mix "${COMMON[@]}" --round-weights 0.25 0.75 \
  --updates "$R2_UPDATES" --output "runs/train/r2-mix-w25-75-n$R2_UPDATES"
# Equal-budget original sampler:
train --recipe mix "${COMMON[@]}" --sampling-mode legacy \
  --updates "$R2_UPDATES" --output "runs/train/r2-mix-uniform-n$R2_UPDATES"
```

These tails follow `python scripts/train_round2.py`. Do not source the job just
to obtain `COMMON`: it performs data preparation and training. The job retains
pure's existing report-only/no-hard-stop amendment. CUDA validation is pending;
CPU fixtures check continuation, allocation and the unchanged loss composition.

## Focused CPU validation

Executed from the isolated checkout with the existing local Python environment
and `PYTHONPATH=src` (so imports resolve to this checkout). Exact final selection:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -m pytest -q \
  tests/test_round_sampling.py \
  tests/test_round2_training.py \
  tests/test_round_training.py::test_mix_step_is_bit_identical_to_the_library_mixed_step \
  tests/test_round_training.py::test_pure_and_mix_draw_identical_selfplay_sequences_across_wraps \
  tests/test_round_training.py::test_failed_input_does_not_advance_the_sampler \
  tests/test_round_training.py::test_restore_round_restores_everything_and_installs_rng_last \
  tests/test_round_training.py::test_restore_round_refuses_mismatches_and_leaves_rng_unchanged \
  tests/test_round_training.py::test_round_checkpoints_load_for_inference \
  tests/test_round_training.py::test_uninterrupted_resumed_and_crashed_runs_are_bit_identical \
  tests/test_mixed_update.py::test_sampler_crosses_both_boundaries_deterministically_without_global_rng \
  tests/test_mixed_update.py::test_mixed_resume_rejects_drift_before_rng_install \
  tests/test_mixed_update.py::test_mixed_inference_preserves_rng_and_legacy_cursor_not_returned \
  tests/test_mixed_update.py::test_exact_cpu_fresh_process_next_update \
  tests/test_checkpoints.py::test_inference_predictions_metadata_and_rng_unchanged \
  tests/test_checkpoints.py::test_next_dropout_update_gradients_adam_and_random_draws_are_exact \
  tests/test_checkpoints.py::test_malformed_checkpoints_fail_clearly_without_installing_rng \
  tests/test_checkpoints.py::test_unsupported_save_settings_fail
bash -n jobs/b200-selfplay/train-round2.sbatch
git diff --check
```

The job's sampling-configuration block was also executed alone (no cleanup,
training, GPU work or submission): both defaults, 25/75 overrides, legacy mode,
distinct output paths when settings/budget differ, and rejection of missing/zero
budgets and invalid weights passed.

Final result (18 October 2026): **112 passed in 179.72 seconds**. Shell syntax,
configuration smoke checks and `git diff --check` passed. No full repository
suite, unrelated match/self-play tests, long training or GPU workload was run.
The earlier focused sampler/R2 pass was 27 passed in 104.50 seconds, followed
by 25 sampler checks in 0.77 seconds; the final selection above includes all
new/changed tests after the last code edits. CUDA validation and the choice of
full-run N remain pending. All changes are uncommitted, based on `c5636c1`.
