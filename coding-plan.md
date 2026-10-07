# Coding plan: small tasks from interface to playable agent

Version 2 — 7 October 2026. F01, F02 and A01–A06 are implemented and verified. A07 implementation and automated checks pass; second-person review is pending. This copy is the execution record for `/Users/arvindsrinivass/Desktop/RL-Chess`. This revision fixes GAB + Q/K/V relative attention, replaces the attention comparison with combined-model profiling, requires P-versus-R evaluation and supervised replay mixing, adopts actual-occurrence draws and the batched evaluator, and opens staged data expansion beyond 50,000 labels.

This is the execution companion to [plan1.md](plan1.md), not a replacement strategy. [interface.md](interface.md) defines the shared behavior. The intended agent is a Chessformer network with Leela-inspired MCTS and self-play, based on AlphaZero's learning principle and initialized with Stockfish supervision.

The adapter is largely implementation against a settled contract. Later stages still require recorded configuration choices and pilot measurements. Small tasks and explicit checks reduce mistakes; they do not guarantee error-free code.

## How to use this plan

Pick an unchecked task whose prerequisites are complete. Finish its deliverable and acceptance check before starting another dependent task. Each row is intended to be a small, reviewable unit of work; split a row further if it grows beyond one focused coding session. Time-consuming annotation/training runs are separate tasks from writing their runners.

Owners follow the existing division: **P1** owns data/model/training; **P2** owns adapter/search/game execution; **Both** review shared contracts and integration. These are responsibilities for the two project members, not instructions to launch additional agents.

For each completed task, record the changed files, the actual command or experiment used to check it, its result, and any remaining limitation in the completion log. Commit a coherent change when repository work permits; do not bundle unrelated features. A checkbox means the check passed, not simply that code exists. If a check fails, fix or revise that task before passing its gate. Mock data and dummy evaluators verify plumbing only.

Use one implementation for encoding, legality and rule outcomes. Data loading, search and evaluation must call it. Do not reimplement the contract in those consumers. If a real semantic gap appears, clarify `interface.md` first and note whether it requires a new encoding/model/data version.

## Suggested code boundaries

These are proposed paths to create during coding, not files already implemented. Keep them small; add files only when their tasks need them.

| Area | Proposed location | Responsibility |
|---|---|---|
| Package and environment | `pyproject.toml`, `src/chess_rl/` | Installable package and reproducible dependency environment |
| Shared adapter | `adapter/state.py`, `adapter/moves.py`, `adapter/encoding.py`, `adapter/outcomes.py` | Exactly the API and semantics in `interface.md` |
| Configurations and records | `config.py`, `records.py`, `configs/` | Validated settings, versions and record schemas |
| Data preparation | `data/pgn.py`, `data/sampling.py`, `data/teacher.py`, `data/dataset.py` | Replay, split, sample, annotate, load |
| Network | `model/attention.py`, `model/backbone.py`, `model/heads.py`, `model/network.py` | Fixed GAB + Q/K/V relative operations, policy/WDL outputs |
| Search | `search/evaluator.py`, `search/puct.py` | Evaluator adapter and tree search |
| Training and persistence | `training/losses.py`, `training/trainer.py`, `training/checkpoints.py` | Supervised/RL updates and resumable checkpoints |
| Game execution and evaluation | `selfplay/runner.py`, `selfplay/buffer.py`, `evaluation/matches.py` | Games, finalized targets and controlled matches |
| Entry points and checks | `scripts/`, `tests/`, `tests/fixtures/` | Small commands and meaningful correctness/regression checks |
| Run artifacts | `runs/<run_id>/`, `data/manifests/` | Configurations, provenance, metrics and outputs; keep large binaries out of Git |

Package-relative paths in this table live under `src/chess_rl/`. CPU adapter code uses NumPy; model/training code converts batches to PyTorch. Do not require a GPU or Stockfish merely to run adapter tests. Record source commits and attribution before porting upstream model operations.

## Organize the code around one owner for each behavior

Create the smallest working package first and add modules when a task needs them. The proposed tree is a map of responsibilities, not a request to create empty files or a class for every function. Keep closely related logic together until separation makes its responsibilities clearer. Prefer straightforward functions and small typed records to deep class hierarchies, plugin systems or a generic game framework.

| Behavior | Single owner | Other components do |
|---|---|---|
| Coordinates, move IDs, legal masks, features and rule outcomes | `adapter/` | Call the adapter; never maintain another conversion/encoding implementation |
| Encoding/vocabulary versions and WDL order | Shared contract constants | Import the same constants; checkpoints/manifests copy their values as metadata |
| Runtime settings | Validated configuration for the relevant component | Receive config explicitly; no hidden globals or repeated defaults in scripts |
| Combined attention | One network/block implementation with GAB and all accepted Q/K/V relative terms | Use the same architecture for supervised and RL training; diagnostic disabling serves correctness checks only |
| Legal policy normalization and numerical loss handling | Shared tensor helpers and `training/losses.py` | Use the same normalization semantics, with deliberate adapters for class labels versus visit distributions |
| Tensor/device preparation and model inference for search | The real evaluator | Give it states; search does not access model layers or teacher internals |
| Checkpoint versions, loading and resumption | `training/checkpoints.py` | Call the same save/load functions from pretraining, RL, evaluation and demo |
| Rule-ending policy | Adapter outcomes | Episode runner adds interruption/ply-limit statuses; neither changes rule outcomes |
| Game execution and target finalization | Self-play runner and buffer | Training consumes finalized records; it does not independently invent outcomes |
| Match protocol and results | Evaluation runner | Validation, final evaluation and demonstration reuse the relevant underlying functions |

Expose a small public adapter API matching `interface.md`; internal helper placement can change without breaking consumers. Keep CPU/NumPy adapter logic independent of PyTorch, Stockfish and search. Model code consumes feature tensors and configuration, not PGN parsing or game execution. Data code uses the adapter and teacher; search uses the adapter and evaluator; training consumes batches and shared losses. The evaluator is the explicit bridge between states and the network. Avoid reverse imports and circular dependencies.

Use one canonical replayable record for each sample/game. Encoded features and masks are derived values or explicitly versioned caches, not competing sources of truth. Preserve source provenance and reject incompatible cache/checkpoint versions. Do not repeatedly serialize dense one-hot teacher policies or duplicate the full game history inside every position record when a game reference plus ply can represent it.

Keep command-line scripts thin: parse a configuration, call package functions, save outputs. Algorithms belong in the package, so notebooks, tests, evaluation and the demo call the same code. Separate run configuration/metrics from large data and checkpoint files; name artifacts by run/checkpoint identity and never overwrite immutable P.

Share real repeated behavior, not vaguely similar code. Add an abstraction when an actual second caller needs the same semantics; allow short explicit task-specific code where factoring it out would obscure the behavior. Do not duplicate the policy head, loss masking, replay rules or checkpoint loader while waiting to refactor later. Numerical reference checks may intentionally use a small independent calculation so tests do not merely repeat the implementation.

At each task review, check that its imports follow these boundaries, that no second source of truth was introduced, and that the public call is understandable without reading the entire pipeline. Keep documentation and acceptance checks focused on externally observable behavior. Adopt the batched evaluator signature from the start and implement batch-of-one first. Measure batching across independent games during the pilot; broader caching, multiprocessing and concurrent traversal within one tree follow measured need.

## 1. Establish the coding environment

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | F01 / Both | None | Create a minimal package skeleton, dependency environment, test entry point and a short README with real setup commands. Pin the resolved Python/dependency versions; defer modules until their tasks need them. | A fresh environment can import the package and run one smoke check. No dataset/engine download is required for unit checks. |
| [x] | F02 / Both | F01 | Define only the adapter state/result types and version constants needed next, plus minimal configuration validation. Add domain record/configuration types in their later tasks rather than designing every future schema now. | Invalid configurations fail clearly; a sample typed adapter result survives serialization with versions and array shapes preserved. |
| [ ] | F03 / P1 | F01 | Record exact upstream commits, source files and attribution for Maia-3 and the earlier relative-attention implementation. | Every operation to be ported points to immutable source. No reference to a moving branch substitutes for the recorded commit. |

## 2. Implement the shared adapter first

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | A01 / P2 | F02 | Implement `ChessState`, `make_state`, initial-FEN provenance, legal UCI replay and copies retaining the move stack. | Standard start, declared setup and isolated FEN behave as specified. Invalid positions/variants/null or illegal moves fail; known prefixes replay identically. |
| [x] | A02 / P2 | A01 | Implement actual/network square transforms and both move-ID conversions. | White/Black round trips preserve UCI and promotion piece. Check documented IDs, both castling sides, en passant and all four promotions; reject illegal/out-of-range IDs. |
| [x] | A03 / P2 | A02 | Implement the legal mask using the shared move encoder. | Set bits correspond exactly to enumerated legal moves, with no duplicate IDs or ordinary pawn-promotion slots. Terminal handling does not alter the meaning of chess legality. |
| [x] | A04 / P2 | A01 | Collect eight newest-first real snapshots, compute prefix repetition and earliest-position padding with provenance. | Check start, two plies, eight or more positions, isolated FEN and FEN plus moves. Later moves never change an earlier flag; padding never creates a repetition. Inspection restores the original state. |
| [x] | A05 / P2 | A04 | Fill the complete 128 channels, including current metadata and engineered features. | Hand-checked fixtures verify every channel group, ownership and scales for both colors. Include promoted material, legal/pinned en passant, checkers and opposite-colored bishops. Features are finite `float32 [64,128]`. |
| [x] | A06 / P2 | A02 | Implement ending policy `actual-occurrence-v1`, perspective-correct WDL/value and `apply_move` with atomic failure. First use `outcome(claim_draw=False)`, then actual third repetition or reached fifty-move condition as specified in `interface.md`. | Checkmate, stalemate, insufficient material, actual repetition/halfmove draws and terminal precedence pass. Intended-move-only repetition/fifty-move claims remain nonterminal; reached draw conditions with legal moves terminate. Failed application leaves state/history unchanged. |
| [ ] | A07 / Both | A03, A05, A06 | Review the assembled adapter and add deterministic replay/regression fixtures, including legal sequences longer than the network history window. | All adapter checks pass. Each member can replay the same record and obtain identical features, mask, move IDs and terminal result. No GPU or teacher engine is needed. |

**Gate A — automated checks passed; second-person review pending:** A07 must pass before another component depends on real encoded states. This is the first coding milestone. Freeze the accepted behavior, not the internals of its implementation.

## 3. Prepare supervised data

After Gate A, the data, model and search tracks can proceed independently. They meet again at integration gates.

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | D01 / P1 | F02 | Create the dataset manifest and decide the archive, sampling/split seeds, game-ID convention, storage format and ending-policy ID. Plan an initial 10,000 labels and measurement-based expansion to 100,000, several hundred thousand and potentially 1 million. Record the fixed teacher configuration and resolve the binary/version/checksum. | Required provenance fields are explicit; pending archive checksum is filled in D02. Archive coverage is audited before fixing the main scale. No annotation starts with an incomplete manifest; 1 million labels is not a mandatory allocation. |
| [ ] | D02 / P1 | D01, A07 | Fetch the selected monthly archive, record its exact URL/checksum, and implement standard PGN parsing/replay with rejected-game accounting. | Valid games replay with full history; corrupted, variant and unreplayable games are rejected with counts/reasons. A small fixture requires no network access. |
| [ ] | D03 / P1 | D02 | Canonicalize duplicate games and assign deterministic 90/5/5 splits by game, before position sampling. | Repeated preparation produces the same assignments. Duplicate games cannot land in separate splits. |
| [ ] | D04 / P1 | D03 | Record phase boundaries and sampling algorithm; select at most eight distinct nonterminal positions per game and remove position overlap across splits. Define expansion with stable game assignments and protected held-out sets. | A split/coverage audit passes. Identity includes rule-relevant board state and legal en-passant availability, independent of network-history differences. Report phase counts per split before/after removal. Protected held-out splits remain separate; removals, expansion and refill rules are deterministic. |
| [ ] | D05 / P1 | D04 | Implement the Stockfish worker and annotate a tiny valid sample with the locked settings, hash resets, failure/retry records and root-perspective targets. | Every retained best move is legal and matches its ID; WDL counts/probabilities are valid. Missing WDL/interrupted analyses are never accepted as targets. Replaying a saved record reproduces its root. |
| [ ] | D06 / P1 | D05 | Run the 1,000-position annotation benchmark separately from worker implementation. | Save actual throughput, failures, settings and returned search information. Project annotation/replay/encoding/storage costs for 10,000, 100,000, several hundred thousand and 1 million labels, identifying measured rates and scaling assumptions. Choose expansion using these costs and subsequent learning curves. |
| [ ] | D07 / P1 | D05 | Implement record loading, shared-adapter tensor construction, batching and optional versioned cache validation. | A saved record yields the same state/mask/label after reload. Reject incompatible versions and invalid labels. A teacher class ID remains a class ID, not a dense stored one-hot policy. |
| [ ] | D08 / P1 | D06, D07 | Produce the initial 10,000 successfully labeled positions using the measured worker and audited splits. | Manifest, success counts, split overlap and replay/target audits pass. Failures/exclusions are accounted for rather than counted as completed labels. |

## 4. Build the network against the interface

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | M01 / P1 | F02, F03 | Port compact GAB and its shared template bank. | Match the pinned reference operations numerically on synthetic nontrivial inputs. Confirm d1=8, d2=32, d3=32 and that the template parameter is shared, not copied per layer. |
| [ ] | M02 / P1 | M01, A07 | Assemble the projection, additive square embeddings and specified Post-LayerNorm encoder blocks. | Correct `[B,64,256]` outputs, finite forward/backward values and working train/eval dropout behavior for several batch sizes. Record resolved normalization/bias/initialization settings. |
| [ ] | M03 / P1 | M02 | Implement the pair policy head, `d_policy=128`, promotion adjustment and exact flattening order. | Compare selected ordinary/promotion logits to an independent scalar calculation/reference. Output is `[B,4352]`; queen and underpromotions occupy their own correct slots. |
| [ ] | M04 / P1 | M02 | Implement mean pooling and the specified WDL head. | Output is `[B,3]` logits in WDL order. Pool across squares; one known synthetic case independently checks the head path. |
| [ ] | M05 / Both | M03, M04 | Assemble the backbone and heads with GAB as an intermediate correctness stage, before adding relative terms in M06. | End-to-end adapter batches produce finite policy/WDL logits and gradients. Record intermediate parameters/memory; this stage is not the architecture for supervised or RL runs. |
| [ ] | M06 / P1 | M05, F03 | Port all implemented Q/K/V relative terms and finalize the fixed combined architecture. Keep operations inspectable and a diagnostic disable path for correctness checks. | Match reference outputs and gradients with nonzero relative tables and asymmetric inputs. Disabling terms recovers GAB-only behavior; all three tables receive gradients when enabled. No omitted position-only cross term is added. Check finite combined-model forward/backward results and count full parameters/memory; resolve a material mismatch with the provisional 6–8M target before freezing a full run. |

M06 is required before supervised training, the real evaluator and the combined-model pilot. It remains on the training critical path by the accepted architecture choice. M05 is an intermediate implementation check, not a competing training track; dummy search and data preparation can proceed independently of M06.

## 5. Build search using a dummy evaluator

Leela is the practical chess-search/self-play reference; AlphaZero supplies the underlying policy-improvement loop. Use our agreed WDL, move vocabulary and ending policy throughout.

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | S01 / P2 | A07, F02 | Implement a uniform-legal-prior, zero-value dummy evaluator with `evaluate(states) -> list[Evaluation \| TerminalResult]` in input order. Use a batch-of-one call from search initially. | Empty, batch-of-one and mixed-terminal batches preserve order/input histories. Priors sum to one over legal moves only. Terminal states return exact outcomes and never enter policy normalization. |
| [ ] | S02 / P2 | S01 | Define search configuration and implement tree edges, selection, expansion and legal traversal. Record the PUCT formula/constants, tie handling and simulation-count convention. | Fixed seed/config yields repeatable traversals; input state/history is unchanged. Distinct histories are not merged by FEN-only identity. Root expansion versus counted edge visits is explicit. |
| [ ] | S03 / P2 | S02 | Implement backup with values stored from each parent player's perspective. | Tiny controlled trees independently verify one-ply and two-ply signs, visit totals and terminal scores. A terminal child is handled without evaluator invocation. |
| [ ] | S04 / P2 | S03 | Return raw root visits, normalized training policy and a legal selected move; separate training exploration from evaluation selection. | Positive training visit count is required. Returned policy has legal support and unit sum. Terminal roots return no action. Budget exhaustion cannot fabricate a policy. |
| [ ] | S05 / P2 | S04 | Implement a minimal match runner and search smoke checks with known chess positions. | Dummy search completes legal games under the cap/ending policy; a sufficient recorded budget recognizes simple terminal wins. Truncated games are counted explicitly, not scored as draws. |

**Gate S — search accepted:** S05 and controlled backup checks pass before we interpret results from model-guided games. Completing games with a dummy evaluator does not demonstrate playing strength.

## 6. Implement learning, persistence and the real evaluator

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | T01 / P1 | M06, D07 | Implement legal policy cross-entropy, soft WDL cross-entropy and a minimal supervised training step for the combined architecture. Choose and record optimizer, loss weighting, seed and small-run batch settings. | Independent calculations check both losses. Illegal policy logits receive zero loss gradient; teacher distributions remain soft. No `0 * -inf`, terminal sample or nonfinite loss passes silently. |
| [ ] | T02 / P1 | T01 | Implement versioned checkpoint save/load with model/config, optimizer, progress and random states. | Save/load preserves evaluation predictions. A deterministic continuation check reproduces the next update on a supported backend; record backend nondeterminism where applicable. Reject incompatible encodings/configurations. |
| [ ] | T03 / P1 | T02, D05 | Overfit a tiny fixed training-only labeled subset with dropout disabled. Set a step cap and acceptance thresholds before running. | Best-move fit becomes near-perfect and teacher-WDL KL becomes small under the recorded thresholds; compare WDL loss to the teacher entropy floor, not to zero. No held-out samples are used. Save metrics/checkpoint. |
| [ ] | T04 / Both | T02, A07 | Implement the real batched evaluator: checkpoint loading, terminal partitioning, encoding/batching of nonterminal states, inference mode, legal normalization and WDL-to-value conversion, with results restored to input order. | Empty, batch-of-one and mixed-terminal batches preserve order and input histories. Terminal states skip encoding/model inference. Batched nonterminal predictions agree with individual evaluation within recorded numerical tolerance. Legal priors normalize and `v=P(win)-P(loss)` matches an independently checked result. |

## 7. Complete a tiny end-to-end cycle locally

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | R01 / P2 | S05, T04 | Implement model-guided self-play execution, move-boundary adjudication, completed/truncated/interrupted statuses and resumable game records using the batched evaluator. Record a deliberate temperature/root-noise schedule and opening source/seed/prefix. Initially run batch-of-one; permit independent games to contribute one pending leaf each for pilot batching. | Terminal outcomes take precedence over the ply cap. Forced interruption/reload preserves initial board, moves, ending policy, checkpoint/search settings, opening prefix and RNG. Search may rebuild; history cannot disappear. Log duplicate completed games and completion/truncation/interruption rates. |
| [ ] | R02 / Both | R01 | Finalize completed-game records into the replay buffer with per-position legal masks, root visits and exact final WDL. | Winner/loser labels flip correctly across players. Incomplete games remain unlabeled and excluded. Raw visits and normalized targets are consistent. |
| [ ] | R03 / P1 | R02, T01, T02 | Implement an RL update mixing completed-game search policy/WDL with training-split Stockfish samples. Before running, record supervised sample fraction strictly between 0 and 1, source, conservative learning rate, loss weights and replay/update schedule. | Fixed supervised and completed-game fixtures produce a finite mixed update with contributions from both sources and survive checkpoint resume with settings preserved. Verify supervised class/soft-WDL targets and self-play distribution/exact-WDL targets remain distinct; sample counts/losses are logged by source. No held-out or unfinished-game samples enter replay. |
| [ ] | R04 / Both | R03, T03 | Run a few model-guided games, finalize any completed records, update, reload and run a short match. If all games truncate, diagnose or use a known valid near-terminal fixture for the plumbing check. | Saved artifacts show the complete position → search → game → target → update → checkpoint → evaluation path. Never relabel a truncation to pass. The overfit/plumbing checkpoint is not the project pretraining checkpoint P. |

**Gate I — local integration accepted:** R04 passes before spending the main training allocation. The tiny cycle establishes functioning plumbing, not improvement or strength.

## 8. Run the pilot before fixing the main budget

| Done | ID / owner | Prerequisites | Build/run | Acceptance check |
|---|---|---|---|---|
| [ ] | P01 / Both | R04, M06, D08 | Profile and briefly train the fixed GAB + Q/K/V relative-attention model with recorded seeds/configuration. Measure batch-of-one and realistic batches across independent games. | Measure validation policy/WDL quality, training speed, encoding/inference speed, end-to-end game throughput, memory and total parameters. Choose practical batching/concurrency from measurements. Keep test data reserved; no attention ablation or architecture selection is required. |
| [ ] | P02 / P1 | P01 | Run pilot supervised training with the fixed combined architecture and validation-based stopping/checkpoint selection. | Save an immutable checkpoint P with provenance, configuration, optimizer state and training/validation learning curves. Tiny overfit success does not substitute for held-out validation. |
| [ ] | P03 / Both | P02, E01 | Run pilot self-play at the measured initial budget, starting around 16–32 simulations per move and targeting approximately 100–300 games, followed by supervised-mixed updates, mandatory P-versus-R validation matches and weaker fixed-opponent diagnostics. | Save R checkpoints, per-source losses, completion/truncation/interruption and duplicate-game rates, usable positions, match results and a real interrupted-run/resume check. Diagnose high truncation/duplication before scaling; randomized replayed book openings are an available diversity adjustment. Do not require a positive RL gain to call the pilot complete. |
| [ ] | P04 / Both | P03, D06 | Write pilot measurements and projected main-run/evaluation costs. Choose staged dataset expansion and refine supervised replay settings, search budgets and batching. | Record annotation/replay/encoding/storage costs, held-out learning curves, GPU memory, training/inference speed, complete games/hour, average length and match latency including CPU work. Evaluate 100,000, several hundred thousand and potentially 1 million labels without a 50,000 ceiling. Projection fits the allocation and reserves compute/time for required P-versus-R evaluation and reporting. |

**Gate P — measured budget accepted:** P04 precedes the main run. The combined attention architecture is already fixed and numerically verified. Dataset size, batch sizes/concurrency, supervised replay settings and main search budgets follow evidence. Gumbel search, KL anchoring, Q-blended value targets, playout-cap randomization and additional auxiliary heads remain later options, not prerequisites for completing this project.

## 9. Main training and final evaluation

| Done | ID / owner | Prerequisites | Build/run | Acceptance check |
|---|---|---|---|---|
| [ ] | E01 / P2 | S05 | Specify and implement required P-versus-R and fixed-opponent matches, including random legal moves, one-ply material-greedy and pinned Stockfish with small node budgets. Use paired openings with colors reversed, separate validation/test openings and no exploration noise. | Fixtures preserve pairing, checkpoint identity and ending policy. P and R use identical search settings. Record opponent versions/settings, seeds, greedy scoring/ties and failures/truncations. Allocate actual game counts across comparisons from measured cost; 200–400 final games is an aim if feasible, not a mandatory strength claim or guaranteed decisive result. |
| [ ] | L01 / Both | P04, E01 | Freeze a versioned main-run configuration and schedule, including combined architecture, dataset size, loss/optimizer, nonzero supervised replay fraction/source, replay/update settings, opening/exploration settings, batching/concurrency, search budgets and required P-versus-R evaluation cadence. | Every required choice below is resolved; measured projected cost fits the allocation, reserves match compute and a stop date leaves time for evaluation/reporting. |
| [ ] | L02 / P1 | L01 | Prepare the chosen main dataset/checkpoint. If expanding labels or supervised training, rerun the existing audited pipeline; otherwise use P. | Label/split/version and phase-composition audits pass; expansion preserves game assignments and protected held-out sets. Any new supervised P has a distinct immutable ID; record which P initializes RL and supplies the frozen baseline. |
| [ ] | L03 / Both | L02 | Run resumable self-play with supervised-mixed training, periodic P-versus-R validation matches and saved R checkpoints. | Full configs, per-source losses, game/status/duplicate counts and rates, usable positions, compute and validation results are retained. Failed jobs resume from recorded state; P is never overwritten. |
| [ ] | E02 / Both | L03, E01 | Freeze the R comparison checkpoint and playable checkpoint using validation, then run reserved P-versus-R and final fixed-opponent test matches once under the recorded protocol. | Record both selections before examining test results. The playable checkpoint may be P, but R-versus-P test results are still required. Save game traces, paired openings, W/D/L, failures/truncations and actual resource cost for each comparison. Test results are not used to tune checkpoints afterward. |
| [ ] | E03 / Both | E02 | Compute score and uncertainty respecting opening pairs; assess P-versus-R change and assemble final tactical/legal-play and latency results. | Independently verify `(wins + 0.5*draws)/completed_games`, pairing-aware uncertainty and denominators per comparison. Report improvement, degradation or inconclusive evidence honestly; do not infer a general causal advantage from one run. |

## 10. Deliver the project

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [ ] | Z01 / Both | E03 | Write reproducibility instructions and the report using actual artifacts, configurations, training curves, match results and measured compute. | A documented clean setup loads the selected checkpoint and reproduces a short evaluation. Report distinguishes project extensions from borrowed methods and measurements from proposals. |
| [ ] | Z02 / P2 | E02, T04 | Provide a small playable demonstration, initially a CLI or minimal existing chess interface using the same adapter/model/search. | A user can play legal moves against the selected agent; special moves, terminal outcomes and checkpoint loading work. No separate encoder or rules implementation is introduced for the demo. |

**Completion:** Z01 and Z02 are done, the selected model and run metadata are recoverable, mandatory P-versus-R results are reported, and limitations are documented. A particular Elo, positive RL gain or broader causal superiority claim is not required.

## Choices to settle at their task, not silently while coding

| Choice | When | What gets recorded |
|---|---|---|
| Python/dependencies and record storage | F01–F02, D01 | Exact versions, schema/storage choice, supported device environment |
| Upstream implementation | F03 | Immutable commits, source paths, attribution and resolved model configuration |
| Dataset identity and sampling | D01–D04 | Monthly archive/checksum, seeds, duplicate identity, phase boundaries/composition, overlap removal/refill and stable expansion policy |
| Teacher binary | D01, verified D05 | Version/checksum, all UCI settings, requested/returned search information |
| Initial PUCT and exploration | S02–S04, before R01 | Formula/constants, unvisited-edge value, ties, simulation counting, root noise, temperature and move selection |
| RL safeguards and opening diversity | Before R03/R01 respectively; refined P03–P04 | Nonzero supervised replay fraction/source, conservative learning rate and update-to-data ratio, per-source losses; root noise, temperature, opening source/seed/prefix and duplicate/truncation metrics |
| Small-run training | T01–T03, before R03 | Optimizer/loss weights, batch/seed, overfit thresholds/step cap, replay/update schedule |
| Practical budgets for fixed combined model | P01–P04, L01 | Quality/cost measurements, dataset size, batching/concurrency, simulations and reserved training/match allocation |
| Opponents, openings and uncertainty | E01, before P03 | Versions/settings, required P-versus-R protocol, weaker rungs, paired validation/test sets, scoring/failure policy and uncertainty method |

Already locked: 128-channel encoding, eight-snapshot padding, Black rank reflection, 4,352 IDs, WDL order/perspective, initial teacher limits/settings, GAB + implemented Q/K/V relative attention, actual-occurrence ending policy, batched evaluator API, supervised replay mixing, required P-versus-R evaluation and unfinished-game treatment. Read their exact definitions in `interface.md`; this roadmap does not redefine them. Encoding/vocabulary versions stay unchanged; the ending-policy ID distinguishes this revision.

## Critical path and immediate next task

Environment/types → adapter Gate A → data + combined network (including verified relative port) + dummy search → tiny overfit and real batched evaluator → local cycle Gate I → combined-model/GPU pilot and P-versus-R assessment Gate P → main run → reserved evaluation → report and playable agent.

F01, F02 and A01–A06 are complete in this implementation. A07 automation is complete; obtain the second-person review before marking A07/Gate A accepted. The next implementation task is **F03**. F03 can proceed while adapter coding continues. After Gate A, P1 works on the D/M tracks and P2 on S; integrate at T04/R04. Complete E01 after S05 and before P03 so checkpoint evaluation is available for the pilot. Z02 can be developed once a selected checkpoint is available, alongside the report.

Keep the relative schedule in `plan1.md`: foundations in days 1–3, pilot in days 4–7, main self-play in days 8–14, final evaluation in days 15–17, and report/demo in days 18–21. These are planning windows, not promises that all tasks take equal time. If measurements make them unrealistic, reduce optional scale and comparisons while preserving correctness gates and final deliverables.

## Completion log

Append evidence here as tasks finish. Do not prefill commands as if they have already run.

| Task | Date | Files / commit | Check command or run ID | Result / remaining limitation |
|---|---|---|---|---|
| F01 | 2026-10-07 | `pyproject.toml`, `.python-version`, `uv.lock`, package, `README.md`, foundation test | `uv sync --locked`; `uv run --locked pytest tests/test_foundation.py`; fresh-env commands in README | Initial smoke: 1 passed. Fresh environment imported package and passed the final 119-test suite. Python 3.12.12; NumPy 2.5.3; python-chess 1.999 / chess 1.11.2; pytest 9.1.1. CPU macOS arm64 tested. |
| F02 | 2026-10-07 | `contracts.py`, adapter state/result types, `tests/test_types.py` | `.venv/bin/python -m pytest tests/test_types.py` | 7 passed: configuration/version rejection, NPZ+JSON serialization preserves versions, shapes, float32 and provenance; result validation. |
| A01 | 2026-10-07 | `adapter/state.py`, `tests/test_state.py` | `.venv/bin/python -m pytest tests/test_state.py`; final suite | Initial 14 passed; final 17. Legal replay, complete/default/incomplete setup provenance, full-stack copies, invalid boards/variants/nulls and corrupted direct state histories rejected. Explicit complete setup is caller-attested, not independently proven. |
| A02 | 2026-10-07 | `adapter/coordinates.py`, `adapter/moves.py`, `tests/test_moves.py` | `.venv/bin/python -m pytest tests/test_moves.py`; final suite | Initial 27 passed; final 32. All squares/vocabulary slots checked; documented IDs, both castling flanks/colors, EP, queen/underpromotions and round trips pass. Exhaustive A03 check exposed library castling aliases; shared exact-generated-move validation fixed them and adds regressions. |
| A03 | 2026-10-07 | `adapter/moves.py`, `tests/test_legality.py` | `.venv/bin/python -m pytest tests/test_state.py tests/test_moves.py tests/test_legality.py` | 52 passed after canonical-castling fix; legality file 9 passed. Masks match enumerated moves and every decodable slot; impossible promotions and ordinary promotion slots excluded; drawn-state chess legality preserved. |
| A04 | 2026-10-07 | `adapter/history.py`, `tests/test_history.py` | `.venv/bin/python -m pytest tests/test_history.py` | 5 passed: start/two-ply/long history, current-player perspective throughout, historical-prefix flags, padding, FEN+known moves and preservation. Captures/pawn moves do not truncate network history. |
| A05 | 2026-10-07 | `adapter/encoding.py`, `tests/test_encoding.py` | `.venv/bin/python -m pytest tests/test_encoding.py` | 18 passed: every channel group, both perspectives, finite float32 shape, castling order, unclipped scales/promoted counts, bishop condition with other material, checkers, legal/pinned EP and returned-array isolation. |
| A06 | 2026-10-07 | `adapter/outcomes.py`, `tests/test_outcomes.py` | `.venv/bin/python -m pytest tests/test_outcomes.py` | 19 passed: mate/stalemate/material, actual threefold/fifty, automatic fivefold/75, precedence, repetition-before-fifty tie, intended-move-only nontermination, clocks, exact WDL and atomic failures. Incomplete provenance retained. |
| A07 — automated portion only | 2026-10-07 | `tests/test_regression.py`, 13 frozen replay records, `scripts/check_adapter.py`, `runs/milestone-1/` | `uv run --locked pytest --junitxml=runs/milestone-1/pytest.xml`; same command in fresh `.venv-acceptance`; `uv run --locked python scripts/check_adapter.py`; `uv pip check` | Final 119 passed in both environments (1.92s / 2.33s); 11 assembled regression tests include 3 seeded legal playouts, 16-ply repetition beyond input window, rule-equivalence EP/castling/counters, every-prefix preservation and CPU-only imports. All 13 records agree in separate processes/both environments. JSON tuple/list normalization in the review command was fixed before these passes. Second-person Both-owner review is outstanding; A07 checkbox intentionally remains open. |

All commands above actually ran in RL-Chess. Final XML reports have 119 tests,
zero failures and zero errors. `runs/milestone-1/environment.json` records installed
versions; `replay-check.json` records feature/mask fingerprints, legal IDs and
terminal results; `source-plan-checksums.json` verifies originals were preserved
and copies matched before editing this log. No applicable AGENTS.md was found in
either project or the checked ancestor paths. No subagents, annotation, training,
teacher engine, GPU or paid compute were launched.

**Outstanding review:** the other project member should independently run the
README acceptance commands, inspect the contract fixtures and confirm identical
results. The automation establishes reproducibility, not second-person approval.
After that sign-off, mark A07 and Gate A accepted. F03 is the next coding task;
D01/S01 and model work remain pending. No upstream operation has been ported yet.
