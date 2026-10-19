# Integration note for the round-2 match owner

The standalone match tools are integrated on top of Desktop main 5bb1e10.
The common evaluator and acceptance helpers already came from ca6e668's
self-play batching work; its strengthened source inventory is retained.
No production match hooks are applied. The pre-existing head-to-head changes
remain owned by the match chat, and the training/depth-1 files are untouched.

Round-2 match candidate/selection support is still absent in that snapshot.
The deferred `batching-match-hooks.patch` was based on be7453e (the preserved
head-to-head edits above 85b1d66); rebase and review it only after the match
owner supplies the round-2 match commit. Local integration of the standalone
tools does not authorize deploying them to the running B200 checkout.

The standalone runner is independently runnable for synthetic development tests.
The hook patch is deliberately unapplied, is based on the head-to-head snapshot,
and does not change `_check_validation_plans`. Candidate freezing, selection and
test-plan binding remain entirely the match owner's responsibility.

When adding execution compatibility, compare interpreted execution via
`execution_for(plan)`, not raw `plan.get('execution')`. A historical serial plan
has no execution field; a new serial plan has a dictionary. Comparing those raw
values would wrongly reject serial compatibility. `Execution.validation_identity()`
is supplied for the comparison: it treats serial B/K as 1, preserves process
count, and ignores batching-only flush/version settings in serial mode. Batched
identity retains every frozen field. Resume still compares the full execution
object and sealed plan; this comparison helper must not weaken resume checks.

Keep source identities, runtime, device, workers, settings, seeds, openings and
checkpoint identities independently enforced. Interpreting both plans as serial
does not establish compatibility when sources/runtime differ. No historic plan,
digest or result should be rewritten and no R1-mix reuse approval is implied.

Production remains serial for round-2 validation and its test match. The user
submits acceptance only after that test match. Later batching is limited to a
separately recorded report-only head-to-head proposal; no selection or claim may
use it. The isolated execution parser rejects batched production/test plans.

Adding modules changes the inventory used for newly frozen match plans and
batched self-play source identities. Existing serial match resume checks hash
the paths listed in their original plan; a new plan may still have a different
source dictionary. Do not rewrite historical plans or source manifests to force
compatibility. Keep the B200 checkout pinned through the round-2 test match.
No cluster job was submitted as part of this local integration.
