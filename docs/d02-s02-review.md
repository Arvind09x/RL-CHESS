# Independent D02/S02 AI review — 8 October 2026

**Accepted for their scoped milestones, with no blocking findings.** This is
a fresh review by the coordinating Codex agent at the user's request, separate
from either implementation session. It is AI review, not human teammate review.

Reviewed exact commits:

- D02: `cb37b0df79f85ee7615d89778efbad75c1b19576`.
- S02: `88e62a418bfd8cce54bc8c7e90bae60bb5719c6a`.
- Common base: `0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`.

Both original worktrees were clean. Review covered the complete production
changes, tests/fixtures, manifest changes, contracts, milestone documentation
and saved evidence. Both commits merge without conflicts into the isolated
review checkout. Production code and dependencies needed no corrections.

## Findings and acceptance scope

D02 preserves the genuine source root, full legal mainline and declared PGN
metadata. Its strict lexer rejects skipped text and parser errors rather than
accepting a partial prefix. Rejection spans, uncertain comment boundaries,
ZIP identity, all-member accounting and failed-run completion behavior match
the documented scope. Source replay deliberately remains independent of
self-play draw adjudication. Unfinished sources and result discrepancies do
not create targets. Canonical IDs, splits, sampling and labels remain absent.

S02 implements the recorded formula, exact-score/smallest-ID ties and legal
zero-prior edges. State copies preserve retained histories; distinct paths
are not merged by FEN. Terminal adjudication precedes evaluation. Expansion
publishes no partial edges on failure; failed traversal retains a retryable
child without changing N/W. The value convention is leaf-player results and
parent-player edge statistics. Root preparation and traversal contribute no
completed visits. Custom evaluators and search-owned records must honor the
documented read-only/sequential ownership contract.

No blocking findings or production fixes were required. Review added only
independent checks, evidence and acceptance/status documentation. One reviewer
fixture initially selected an already-moved pawn at the second White turn;
the fixture was corrected to select `g1f3`. The independent raw-audit script
also needed a Path-construction correction and an assertion correction for
python-chess's fabricated default Seven Tag Roster fields. D02 correctly keeps
only source-declared headers. Failed reviewer runs remain saved; none was a
production defect.

Raw stdout and the initial failed-test XML preserve tool-emitted whitespace.
The evidence directory's `.gitattributes` excludes only trailing/EOF whitespace
checks for those generated files; code and documentation retain ordinary checks.

## Fresh verification

Actual commands, UTC times, durations, exit codes, raw output and JUnit files
are under [runs/d02-s02-review](../runs/d02-s02-review/commands.json).
Each `*.command.json` is authoritative for its check; `commands.json` contains
the initial parallel batch. [Structured acceptance](../runs/d02-s02-review/acceptance.json)
indexes all checks and records preservation results.

| Check | Fresh result |
|---|---|
| D02 exact original commit: replay + manifest tests | 155 passed |
| S02 exact original commit: PUCT tests | 95 passed |
| Combined model suite, existing M06 diagnostic plugin | 570 passed |
| Combined adapter suite | 402 passed, six expected optional-model skips |
| Reviewer-authored PGN/search/failure probes | 26 passed |
| Frozen adapter replay in each environment | 13 fixtures each, identical reports |
| M06 checker | Passed |
| Pinned upstream source/notice audit | Passed |
| Teacher binary/package/NNUE byte audit and idle UCI | Passed; no search requested |
| Annotation-readiness gate | Rejected as required; six D03/D04 blockers |
| D02 stored-output audit over all source records | Passed |
| Independent raw archive framing and source hash/location audit | All 289,776 sources matched |
| Independent python-chess parsing + adapter replay sample | 291 games; first, last, every 1,000th ordinal |
| Three saved preparation passes | Replay, rejection and summary bytes identical |

The independent archive audit uses raw Event-header boundaries for this
selected archive rather than D02's framing function. It verifies archive and
member hashes, every source byte/line span and source hash, all output records,
the 25,429,063 stored-ply total, and sampled mainlines/roots/end boards against
a fresh python-chess parse and public-adapter replay. A fresh complete
25,429,063-ply legal replay was **not** rerun in this review; original full-pass
evidence, full raw/output audits and fresh sampled replays support acceptance.
All three original output directories and their checksums were compared.

CPU macOS arm64/Python 3.12.12 is the verified environment. No GPU, training,
annotation, matches, strength measurement or concurrent traversal was performed.

## Integration and next work

D02 and S02 are accepted and integrated together. Their original commits,
branches, implementation evidence and historical pending-review statements
remain intact. This dated acceptance supersedes those historical statements.
The coding plan now records both accepted milestones.

D03 may implement canonicalization/consolidation and deterministic 90/5/5
game splitting from the accepted pre-split bulk output. S03 may implement
transactional, at-most-once backup under S02's signs and counting convention.
Each must stop after its own committed deliverable for independent review.
Annotation remains blocked until D03/D04 decisions and audits are resolved;
**Gate S remains unaccepted**, requiring S03–S05 and controlled backup checks.
