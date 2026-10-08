# Independent D05 and S05 review — 8 October 2026

D05 `9f4f038fefd2a8957970ba763485f49b5cf856c3` and S05
`8b3cf88482897f12251bbe679d5a96525a373e04` share accepted parent
`ad8438acfbd859f734daaebc6115afdab0367b65`. The coordinating Codex agent
performed independent AI review at the user's request, inspected production
code and implementation evidence, and freshly checked a combined committed
snapshot. This is not human teammate review. The user subsequently authorized
merging both branches into local main. Both histories are preserved by merges.

## Findings and acceptance

No blocking findings or production fixes were required. D05, S05 and Gate S
are accepted for the tested CPU correctness/plumbing scope. D05 replays full
history, verifies audited inputs and pinned teacher bytes/settings, clears the
hash before searches, checks completed principal responses and root-perspective
WDL, and rejects interrupted/incomplete analyses. It retains attempt ledgers,
legal move IDs and compact replay references. Both preserved label bundles
use the same 12 roots; review reran record replay rather than new teacher searches.

S05 owns the game history, delegates search and rules to accepted APIs, records
partial completed simulations, and adjudicates terminal outcomes before the
ply cap. Failed/interrupted/truncated records receive no outcome. Candidate
move application cannot publish a partially changed game. Dummy games and
both-color immediate mates establish legal search plumbing, not strength.

## Fresh combined verification

| Check | Result |
|---|---|
| Model regressions, accepted M06 diagnostic plugin | 984 passed |
| Adapter regressions | 816 passed; six expected optional-model skips |
| Prior D02/S02, D03/S03, D04/S04 reviewer cases | 70 passed per environment |
| New reviewer special-move cases | 14 passed per environment |
| D05 label replay | All 12 final and all 12 initial labels pass |
| S05 game replay | All 15 records pass |
| Frozen adapter reports | Both 13-fixture reports match |

New probes independently calculate move IDs and WDL perspective, then verify
teacher responses and bounded games for all four promotions, both castling
sides and en passant in both colors. They check final board/promoted bits,
input ownership, isolated-FEN provenance and unfinished outcomes. The probe
evaluator is controlled; it does not imply teacher/search runtime integration.

[Structured acceptance](../runs/d05-s05-review/acceptance.json) records
byte equality for every merged tracked source/test/script/config/manifest and
dependency file against the freshly tested combined snapshot. Therefore full
regression evidence is reused without repeating identical checks. The durable
review copies preserve original XML/log bytes and reports.
[Original review summary](../runs/d05-s05-review/review-summary.json) is a
historical pre-integration snapshot; its pending-integration fields are
superseded by this acceptance. Tests initially ran in
`/private/tmp/rl-chess-d05-s05-review-ukekilzz`, using the existing primary
`.venv` and `.venv-acceptance`, `PYTHONPATH=src:.` and bytecode/cache disabled.
Source bytes were merged without conflicts in an isolated local clone; only
acceptance documentation and reviewer evidence were subsequently added.

## Limits and next work

CPU macOS arm64/Python 3.12.12 only. Throughput, scale, concurrency, process
crash recovery, training, GPU performance and playing strength remain unaccepted.
D06 benchmarks 1,000 positions separately; D07 implements loading/batching;
E01 implements paired match protocols. D08 depends on D06/D07; T01 depends on
D07. None of these later tasks ran during review/integration. Original branch
evidence, historical manifests, protected registry artifacts and bulk inputs
remain unchanged.
