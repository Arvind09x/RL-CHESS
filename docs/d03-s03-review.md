# Independent D03 and S03 review — 8 October 2026

D03 `7eaba97d14d8e4d769b5db7e8a4e5a3931940d1f` and S03
`e4a1e8b8374d463b3389f503ba5bae29be85ef0e` are accepted for their scoped
milestones through independent AI review by the coordinating Codex agent at
the user's request. There are no blocking findings and no production fixes.
This is AI review, not human teammate review.

Both original commits have exact parent
`1301d033b81ca88e51c66ba27c1035a80c50fe54`. They were combined without conflicts
on `codex/d03-s03-review` in `/private/tmp/rl-chess-d03-s03-review` and reviewed
there. This isolated checkout is the source of the next accepted base;
the original primary checkout and implementation checkouts were not updated.
Nothing was pushed. Preserve this checkout and external data for subsequent tasks.

## Findings and evidence

D03's identity includes the normalized six-field legal-en-passant starting FEN,
including counters, and the complete actual-coordinate mainline. Consolidation
retains every source occurrence, original root, header and result, including
multiplicity and contradictory results. No source result becomes a label.
The integer-threshold split procedure is independent of order and stable under
expansion. Intake hashes and record counts are checked before completed output
is attested. Hash/payload conflicts abort. These behaviors meet D03's contract.

The fresh full audit independently spells identity and split calculations,
compares every game and source occurrence to D02, and verifies output bytes.
The audit runner is the reviewed D03 audit with its report destination changed
to the review directory; it imports no production D03 identity/split helpers.
It matched 289,776 source occurrences, 289,427 games, 349 additional duplicate
occurrences in 112 groups, and 13 groups retaining conflicting completed results.
Actual game assignments are 260,517 train, 14,480 validation and 14,430 test.
All 290 sampled games replayed through the accepted adapter. The two existing
full preparations still have identical games and audit hashes.

S03 negates the leaf value before each reverse-path edge update, retaining
parent-player N/W/Q semantics. A successful nonempty backup counts one simulation;
terminal-root empty paths count zero. At-most-once tickets cover shallow aliases;
distinct pending paths sharing edges accumulate correctly. Validation stages all
updates before publication; injected publication failures in the implementation
tests restore N/W and completion. Accepted formula, defaults, ties, evaluator
and full-history traversal behavior remain intact.

Fresh reviewer probes independently derive expected edge values by comparing
parent and leaf player colors over both colors and depths 1, 2, 3, 7 and 12.
They complete pending paths in reverse order, check aliases, compare 2,000 split
assignments to independent integer arithmetic, and replay six canonical games
into search without input mutation. The 26 accepted D02/S02 reviewer cases
were rerun alongside these 27 new cases.

| Fresh combined check | Result |
|---|---|
| Complete adapter regression | 562 passed, six expected optional-model skips |
| Complete model regression | 730 passed |
| Reviewer cases, adapter environment | 53 passed |
| Reviewer cases, model environment | 53 passed |
| Frozen adapter replay | Both 13-fixture reports identical |
| Full D03 artifact audit | All game/source occurrences matched; 290 sampled replays |
| Preservation | 513 of 518 accepted-base files unchanged; five scoped implementation changes |
| Annotation readiness | Exit 1 as required; four D04 blockers remain |

Actual commands, UTC times, exit codes and raw output hashes live in
[runs/d03-s03-review](../runs/d03-s03-review/acceptance.json).
The model regression uses the accepted `runs.m06.diagnostic_plugin` for historical
M05 behavior. The initial reviewer invocation failed during pytest collection
because two evidence directories contain `test_review_cases.py`; the corrected
invocation uses `--import-mode=importlib`. Both results are preserved. No product
change was needed. Prior task code/evidence and original clean checkouts are
preserved. Only D03's two manifest preparation decisions differ from the base.

## Acceptance boundary and next work

D03 and S03 are accepted on the existing CPU macOS arm64/Python 3.12.12
environments. Full 25-million-ply legal replay relies on accepted D02 evidence;
this review reran the full identity/provenance audit and sampled legal replay.
No fresh full preparation, engine annotation, benchmark, training, real evaluator,
GPU execution or playing-strength assessment was performed. Sequential search
ownership is required; concurrent writes, reconstructed traversal tickets and
crash durability are unsupported.

D04 now follows D03: record phases, deterministic sampling, cross-split rule
position overlap removal, protected held-out registries and stable expansion.
Its four manifest decisions remain unresolved. Game splits alone do not prove
position separation or dataset readiness. Conflicting PGN results stay provenance
and must not become labels.

S04 now follows S03: implement completed-simulation budgeting, raw root visits,
legal normalized training policy, and explicit training/evaluation action
selection. Root preparation and failures must not count visits; terminal roots
return no action; zero visits cannot yield a fabricated training policy.
S05 remains pending, so **Gate S is not accepted**. D04 and S04 can proceed in
separate isolated checkouts from this reviewed base; annotation starts only in D05
after D04 review acceptance.
