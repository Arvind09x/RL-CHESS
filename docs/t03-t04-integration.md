# T03/T04 supplied AI review acceptance and integration — 9 October 2026

T03 `1a9b00dd799af7578476624e4471073eaa3450eb` and T04
`197bf821bfe13f2fab1dc5a8a54da2d8d13779e8` are accepted within their documented
CPU correctness scope. The user supplied an independent AI review finding no
blocking issue, freshly passing 20 T03 and 34 T04 tests, verifying the actual
T03 checkpoint's exact model hashes and final metrics, confirming step 26 was
the first passing update, and exercising that checkpoint through T04 and bounded
search. This is supplied AI review, not human review or a newly commissioned
review. No reviewer or subagent was commissioned by this integration.

Primary `/Users/arvindsrinivass/Desktop/RL-Chess` was on `main` at
`4fa48d4afef91a3bea125fb5f9f85b4d5159fc28`. Its only unrelated working-tree
changes were D06, D07 and E01 checkbox edits in `coding-plan.md`; its index was
clean. Both reviewed isolated checkouts were clean at the exact supplied heads.
No applicable AGENTS.md was found in the repositories or inspected ancestors.
The shared interface, execution plan, task scopes, accepted T02 integration
workflow and M06 diagnostic instructions governed this integration.

Preparation used isolated `t03-t04-integration` on `codex/t03-t04-integration`.
Merge commits `c708168` and `9aee646` retain both reviewed commits and their
history. The only conflicts were append-only README and coding-plan records.
Resolution retained the entire base prefix followed by each exact T03 and T04
completion append. Acceptance is added separately and the T03/T04 task checkboxes
are marked complete. Historical review-pending statements, prior failed-test
logs and all original evidence remain intact; this dated acceptance supersedes
their status. No production code fixes were required.

Fresh checks use the accepted primary `.venv` and Torch-free `.venv-acceptance`
interpreters with the integration checkout's explicit PYTHONPATH, disabled
bytecode/cache writes and single-thread numerical settings. Pins verified:
Python 3.12.12, NumPy 2.5.3, python-chess 1.999/chess 1.11.2, pytest 9.1.1, and
Torch 2.8.0 in the model environment. No dependencies were installed or updated.
Full model regressions use `runs.m06.diagnostic_plugin` for the documented
historical M05 constructor helper; production M06 and T04 inference retain GAB
and all Q/K/V relative terms.

| Fresh combined check | Result |
|---|---|
| Full model regression | 1,398 passed, no skips |
| Full Torch-free adapter regression | 1,023 passed, 11 expected optional-Torch skips |
| Checkpoint/training/T03/T04 focused cases | 206 passed |
| Accepted historical review probes | 84 passed in each environment |
| Frozen adapter replay | All 13 fixtures byte-identical to accepted T02 output in both environments |
| M06 numerical checker | Passed |
| Pinned environments, offline dependency checks, upstream source audit | Passed |
| Actual T03 checkpoint through T04 loader | All saved model tensor hashes and final metrics exact |
| Actual T03 checkpoint through T04 public evaluator/search | Passed mixed-batch, oracle, legality, normalization, history/RNG and bounded-search checks |

The unchanged actual checkpoint is
`external-artifacts/t03-tiny-overfit-20261009/final.pt`, 60,883,331 bytes, SHA-256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`.
T04's shared T02 loader preserves its checkpoint ID, configuration, frozen
provenance, all model tensor hashes and shared GAB bank. The original fixed
eight training samples reproduce the final metrics exactly: 8/8 legal move fit,
mean teacher-WDL KL `0.009595421440313435`, maximum KL
`0.021492420032182824`, and WDL CE `0.3644351661205292` against teacher entropy
`0.35483976223793856`. Reading the 27 original curve rows confirms only step 26
passes every original frozen gate. No original overfit update is rerun.

The separate public T04 audit supplies 31 mixed inputs, with exactly one
`[16,64,128]` forward for the 16 nonterminal entries. Empty and all-terminal
inputs bypass inference. Illegal probability mass is exactly zero; legal priors
and WDL normalize within the original tolerances and `v=P(win)-P(loss)`.
Maximum batch-versus-individual errors are `2.682209014892578e-7` for priors,
`1.1920928955078125e-7` for WDL, and `1.7881393432617188e-7` for value.
Independent float64 NumPy conversion errors are at most
`1.2001197713384926e-7`. Eight roots spanning both players and special moves each
complete exactly three simulations, producing legal moves, legal visit support
and valid normalized training policies. Twelve terminal searches complete zero
simulations and bypass inference. Input histories, caller RNG and checkpoint
bytes remain unchanged. These are bounded correctness probes, without games or
strength/performance claims.

T03's frozen source identity still belongs to its original isolated checkout.
Every original frozen source is verified there and against the corresponding
unchanged integration file. Added T04 files do not extend or rewrite that freeze.
The integration uses its own `verify_t03.py` and evidence directory, without
calling the original runner's freeze/run/verify modes or rewriting its outputs.
The preserved `check_t04.py` report retains its original review-pending scope
string; the separate integration summary records current acceptance.

[Summary](../runs/t03-t04-integration/summary.json),
[exact command index](../runs/t03-t04-integration/commands.json),
[checkpoint verification](../runs/t03-t04-integration/t03-checkpoint-identity-metrics.json)
and [public T04 audit](../runs/t03-t04-integration/t03-through-t04.json) retain
source identities, argv, environment overrides, timings, exits, lossless output
containers and JUnit results. All 15 fresh commands passed. Complete base and
reviewed trees are verified byte-for-byte except the two documented append and
checkbox changes. All indexed external trust roots, protected held-out
registries and 76 retained external artifact files are hashed before and after.
The original primary patch is preserved losslessly in setup.json.

Whole-history diff checking reports only four trailing-whitespace lines in
T04's original failed-test stdout/XML. Those exact reviewed blobs remain intact;
[the diagnostic](../runs/t03-t04-integration/historical-whitespace-check.json)
records this distinction. New staged integration evidence and documentation
must pass whitespace checks before committing.

The application script refuses drift, verifies the committed isolated result,
rehashes external artifacts, saves a recovery stash of primary's user edits,
fast-forwards `main`, and restores exactly the original three uncommitted line
changes. It checks the entire resulting tree and external inventory, commits
only a new primary integration receipt, then removes only its own recovery
stash. Existing stash entries are preserved. The
[primary receipt](../runs/t03-t04-integration/main-integration.json) records the
actual verified result commit and preservation outcome after application.

Acceptance remains limited to pinned macOS arm64 CPU correctness. T03's eight
training roots and plumbing checkpoint establish neither generalization nor
playing strength and do not select immutable pretrained P. Stop after
integration: no R01 implementation, GPU work or training campaign is begun.
