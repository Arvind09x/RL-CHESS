# Scoped D06/D07/E01 review acceptance — 8 October 2026

The user supplied independent review acceptance of the exact D06, D07 and E01
commits below, reporting no blocking findings. This integration applies that
acceptance and runs fresh combined checks; it makes no new independent-review
or human-review claim.

| Milestone | Accepted implementation commit | Independent source results supplied by the user |
|---|---|---|
| D06 | `ac424ecfafc3770023fba196b6158e65d943f199` | 992 model / 824 adapter passes; all 1,000 labels replayed |
| D07 | `b6e3d0eec171aa52e8fbc3ddedda9ec128daf330` | 1,111 model / 942 adapter passes; all 12 D05 roots replayed |
| E01 | `ccea776af7ee74814305e09a5df24578b18511e1` | 1,046 model / 878 adapter passes; both 48-game bundles replayed |

The clean starting local main was
`4184387c7e4b67559355d6b212a2a300c86c5ce6`. Each implementation commit has that
exact parent. Three merge commits preserve the original source histories.
The appended `coding-plan.md` conflict retains the complete D06 and D07
completion entries verbatim. Earlier pending statuses and all historical
success/failure evidence remain intact; this dated acceptance supersedes the
pending status only for these three milestones.

Fresh combined verification in the staged integration clone:

| Check | Model | Adapter |
|---|---|---|
| Full regression suite | 1,181 passed | 1,012 passed; seven expected Torch skips |
| Accepted reviewer probes | 84 passed | 84 passed |
| Frozen adapter replay | 13 fixtures, exact accepted match | 13 fixtures, exact accepted match |
| D07 independent board/history/target/batch audit | All 12 roots pass | All 12 roots pass |
| Accepted S05 replay | All 15 records pass | All 15 records pass |
| Both saved E01 bundles | 48 + 48 games pass | 48 + 48 games pass |

The independent D06 audit also replays all 1,000 labels, proves selection
membership/held-out exclusion and checks raw teacher commands, PVs, WDL and
shared move IDs. The accepted M06 checker and manifest/teacher identity gate
pass. [Exact commands and outputs](../runs/d06-d07-e01-integration/commands.json),
[combined summary](../runs/d06-d07-e01-integration/summary.json),
[preservation hashes](../runs/d06-d07-e01-integration/preservation.json) and
[structured acceptance](../runs/d06-d07-e01-integration/acceptance.json)
record the results and original commit identities.

All accepted base files except the two append-only status documents remain
byte-identical, as do all imported implementation, test, documentation and
historical evidence files. Original external data, held-out/protected registries,
candidate database, benchmark artifacts and Stockfish bytes match their saved
identities. Manifests and annotation teacher settings remain unchanged:
100,000 nodes, Threads=1, Hash=128 MiB, MultiPV=1, WDL enabled, no tablebases.
E01's separately recorded evaluation settings remain part of its exact source.
New D06 per-label audit output stays outside Git, alongside the external bulk
inputs; only integration, verification and acceptance records are added here.

Acceptance is scoped to CPU correctness and the original milestone boundaries.
D06 cost projections remain conditional; D07 uses the existing 12 real roots;
E01 controlled/dummy fixtures establish match plumbing, not playing strength.
No new teacher search, D08, T01, later implementation, training, GPU work,
production checkpoint selection or evaluation campaign was started.
