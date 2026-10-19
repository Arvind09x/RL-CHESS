# Round-2 match audit: reusing R1-mix's validation match across code revisions

Decided by Arvind, 19 Oct 2026 (15:50 and 16:23 IST): round-2 selection compares R1-mix's sealed round-1
validation match with R2-mix's new one under a narrowly scoped, audited compatibility exception. This document
is the audit. The machine-checked form is `CROSS_REVISION_EXCEPTION` in
`src/chess_rl/evaluation/match_campaign.py`; `run_match.py --round 2 select` enforces it and writes the
result into `runs/matches/selection-round2.json`.

## What differs between the two plans

R1-mix's plan (`runs/matches/validation-P-vs-R1-mix/plan.json`, frozen 17 Oct 2026 20:32 UTC) pins the sha256
of 43 evaluation sources at commit `1bacbfe`. `git archive 1bacbfe` reproduces every one of the 43 (checked by
`scripts/check_evaluation_equivalence.py`, which refuses otherwise). Against the working tree of 19 Oct:

| File | Round-1 hash (pinned) | Changed by | What changed |
|---|---|---|---|
| `scripts/run_match.py` | `e2a6fee2…` | 74ac501, round-2 work | command-line parsing and dispatch (`--kind policy`, `--no-search`, `--round`); no evaluation code |
| `src/chess_rl/evaluation/match_campaign.py` | `8d81888d…` | 74ac501, head-to-head edits, round-2 work | new match kinds (policy, head-to-head), round-2 records, this selection rule, reports that read finished matches across revisions. `git diff 1bacbfe` touches `_check_protocol`, `create_plan`, `verify_inputs`, `match_report`, `status`, `decide`, `select`, `compare_on_common_pairs` and the new round-2 section only. **Unchanged:** `jobs` (schedule and seeds), `validate_game`, `publish`, `read_results`, `load_players`, `initialize_worker`, the worker loop in `run`, `search_player`, `opponent_player`. |
| `src/chess_rl/training/checkpoints.py` | `f5dd6ff0…` | 5bb1e10 | `_validate_payload` parses the mixed-sampler state through `round_sampling.sampling_state_from_dict`, which returns `RoundSamplingState` only for `version == round-deadline-cycles-v1` and otherwise `MixedState.from_dict` as before; `save_checkpoint` accepts the new state type; `restore_training` refuses it. `load_inference` of P (DataOrder) and R1-mix (MixedState) follows the unchanged branch and loads the same tensors. |

Every other pinned file hashes identically, and `git log 1bacbfe..HEAD` shows no commit touching
`src/chess_rl/adapter`, `search`, `model`, `evaluation/{matches,players,replay,pilot,__init__}.py`,
`runtime.py`, `contracts.py`, `tensor_ops.py`, `position_sampling.py`, `supervised.py`, `stockfish_worker.py`,
`training/{losses,trainer,__init__}.py`, `canonical_games.py`, `pgn_replay.py`, `data_manifest.py`, `__init__.py`.

Six files exist in the evaluation source directories now that did not at `1bacbfe`; a round-2 plan lists them
and the round-1 plan cannot. None is imported by the play path (they are the standalone batching and
development-match tools of ca6e668 and 83e8f41):

| Added file | Hash admitted |
|---|---|
| `src/chess_rl/evaluation/batching_acceptance.py` | `410f9f34…` |
| `src/chess_rl/evaluation/benchmark_layouts.py` | `404af82e…` |
| `src/chess_rl/evaluation/concurrent_matches.py` | `9414d58b…` |
| `src/chess_rl/evaluation/development_matches.py` | `ef80166b…` |
| `src/chess_rl/evaluation/execution.py` | `20d335be…` |
| `src/chess_rl/search/batched_evaluator.py` | `2092ed24…` |

No pinned file may be missing from a round-2 plan.

## The rule the selection enforces

For every validation plan in the round-2 selection, compared with the newest round-2 plan:
1. every file present in both must hash identically, except the three files above, and only when the older
   plan pins exactly the round-1 hash listed;
2. a file present only in the newer plan must be one of the six added files at exactly the listed hash;
3. a file present only in the older plan is never admitted;
4. the older plans must all be at R1-mix's exact revision (R1-pure-full's round-1 match shares it);
5. any admitted difference requires `runs/matches/evaluation-equivalence-round2.json` (below), bound to this
   exact pair of revisions.

Any other difference stops the reuse with "rerun this validation match under the current revision". All other
protocol checks are unchanged: same openings and set hash, settings, ply cap, protocol, seed, runtime,
device, workers; P's sha256 on both sides; R1-mix's plan sha256 equals the one recorded in
`candidates-round2.json`; a carried-over plan pins the round-1 candidates file, a round-2 plan the round-2 one.
Playing and resuming any match still requires its own sources to match exactly (`load_plan(check_sources=True)`).

## Supporting evidence (not proof of B200 equivalence)

`scripts/check_evaluation_equivalence.py` runs on CPU, so it supports the code audit above; it does not show
that the B200 would reproduce the stored games. It extracts the old revision with `git archive`, verifies all 43
pinned hashes, then runs P and R1-mix under the old tree and the working tree in separate isolated
interpreters (`python -I`, explicit `sys.path`) on all 64 validation roots and the positions reached by
alternating 128-simulation decisions (P White, R1-mix Black, `--plies` per root). Legal policy, WDL, value, the
chosen move and every visited edge's visits and value sum must be byte-identical at every position.

The sealed record carries: old and new hashes of the three excepted files and of all 43 pinned files, the
checkpoint sha256s, the reference plan's sha256, the openings file identity and set hash, settings (roots,
plies, simulations, seed, colours, device, threads), torch/numpy/python versions, and the digest of both result
sets. `select` accepts it only if all of these match the two plans and the round-2 record, coverage is complete
(64 roots, plies ≥ 2, 128 simulations, at least 64 decisions, both networks at every position, not a `--smoke`
run), and there are no mismatches. Cost: CPU only, tens of minutes for `--plies 3`.

## If the audit or the evidence fails

Cross-revision reuse stops. R1-mix's validation match is replayed under the frozen round-2 revision
(`validation-P-vs-R1-mix` cannot be rewritten; a new plan at the new revision is needed, ~3.9 h on the B200),
and the round-2 selection then compares plans at one revision with no exception.

## Pinning

The B200 checkout is pulled once before `candidates-round2.json` is frozen and stays at that revision through
R2-mix's match, the selection and R2-pure's match. Test planning and execution are deferred; batching stays
unapplied in production.
