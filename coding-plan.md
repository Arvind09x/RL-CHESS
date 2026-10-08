# Coding plan: small tasks from interface to playable agent

Version 2 — 7 October 2026. F01, F02 and A01–A06 are implemented and verified. A07/Gate A and F03 are accepted through independent AI review; M01 is accepted through independent AI review; M02 is accepted through independent AI review; M03 is accepted through independent AI review; M04 and M05 are accepted through independent AI review; M06 is accepted for local CPU correctness through user-supplied independent AI review on 8 October 2026. D01 manifest preparation and S01 CPU evaluator plumbing are independently AI review accepted and integrated in merged base 0347ceb. D02 source replay and S02 CPU PUCT selection/expansion/traversal are accepted through fresh independent AI review on 8 October 2026 and integrated together; D03–D08 and S03 onward remain unchecked. Annotation remains blocked and Gate S is not accepted. See the dated entries below. This copy is the execution record for `/Users/arvindsrinivass/Desktop/RL-Chess`. This revision fixes GAB + Q/K/V relative attention, replaces the attention comparison with combined-model profiling, requires P-versus-R evaluation and supervised replay mixing, adopts actual-occurrence draws and the batched evaluator, and opens staged data expansion beyond 50,000 labels.

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
| [x] | F03 / P1 | F01 | Record exact upstream commits, source files and attribution for Maia-3 and the earlier relative-attention implementation. | Every operation to be ported points to immutable source. No reference to a moving branch substitutes for the recorded commit. |

## 2. Implement the shared adapter first

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | A01 / P2 | F02 | Implement `ChessState`, `make_state`, initial-FEN provenance, legal UCI replay and copies retaining the move stack. | Standard start, declared setup and isolated FEN behave as specified. Invalid positions/variants/null or illegal moves fail; known prefixes replay identically. |
| [x] | A02 / P2 | A01 | Implement actual/network square transforms and both move-ID conversions. | White/Black round trips preserve UCI and promotion piece. Check documented IDs, both castling sides, en passant and all four promotions; reject illegal/out-of-range IDs. |
| [x] | A03 / P2 | A02 | Implement the legal mask using the shared move encoder. | Set bits correspond exactly to enumerated legal moves, with no duplicate IDs or ordinary pawn-promotion slots. Terminal handling does not alter the meaning of chess legality. |
| [x] | A04 / P2 | A01 | Collect eight newest-first real snapshots, compute prefix repetition and earliest-position padding with provenance. | Check start, two plies, eight or more positions, isolated FEN and FEN plus moves. Later moves never change an earlier flag; padding never creates a repetition. Inspection restores the original state. |
| [x] | A05 / P2 | A04 | Fill the complete 128 channels, including current metadata and engineered features. | Hand-checked fixtures verify every channel group, ownership and scales for both colors. Include promoted material, legal/pinned en passant, checkers and opposite-colored bishops. Features are finite `float32 [64,128]`. |
| [x] | A06 / P2 | A02 | Implement ending policy `actual-occurrence-v1`, perspective-correct WDL/value and `apply_move` with atomic failure. First use `outcome(claim_draw=False)`, then actual third repetition or reached fifty-move condition as specified in `interface.md`. | Checkmate, stalemate, insufficient material, actual repetition/halfmove draws and terminal precedence pass. Intended-move-only repetition/fifty-move claims remain nonterminal; reached draw conditions with legal moves terminate. Failed application leaves state/history unchanged. |
| [x] | A07 / Both | A03, A05, A06 | Review the assembled adapter and add deterministic replay/regression fixtures, including legal sequences longer than the network history window. | All adapter checks pass. Each member can replay the same record and obtain identical features, mask, move IDs and terminal result. No GPU or teacher engine is needed. |

**Gate A — accepted 2026-10-07 through independent AI review:** A07 must pass before another component depends on real encoded states. This is the first coding milestone. Freeze the accepted behavior, not the internals of its implementation.

## 3. Prepare supervised data

After Gate A, the data, model and search tracks can proceed independently. They meet again at integration gates.

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | D01 / P1 | F02 | Create the dataset manifest and decide the archive, sampling/split seeds, game-ID convention, storage format and ending-policy ID. Plan an initial 10,000 labels and measurement-based expansion to 100,000, several hundred thousand and potentially 1 million. Record the fixed teacher configuration and resolve the binary/version/checksum. | Required provenance fields are explicit; pending archive checksum is filled in D02. Archive coverage is audited before fixing the main scale. No annotation starts with an incomplete manifest; 1 million labels is not a mandatory allocation. |
| [x] | D02 / P1 | D01, A07 | Fetch the selected monthly archive, record its exact URL/checksum, and implement standard PGN parsing/replay with rejected-game accounting. | Valid games replay with full history; corrupted, variant and unreplayable games are rejected with counts/reasons. A small fixture requires no network access. |
| [ ] | D03 / P1 | D02 | Canonicalize duplicate games and assign deterministic 90/5/5 splits by game, before position sampling. | Repeated preparation produces the same assignments. Duplicate games cannot land in separate splits. |
| [ ] | D04 / P1 | D03 | Record phase boundaries and sampling algorithm; select at most eight distinct nonterminal positions per game and remove position overlap across splits. Define expansion with stable game assignments and protected held-out sets. | A split/coverage audit passes. Identity includes rule-relevant board state and legal en-passant availability, independent of network-history differences. Report phase counts per split before/after removal. Protected held-out splits remain separate; removals, expansion and refill rules are deterministic. |
| [ ] | D05 / P1 | D04 | Implement the Stockfish worker and annotate a tiny valid sample with the locked settings, hash resets, failure/retry records and root-perspective targets. | Every retained best move is legal and matches its ID; WDL counts/probabilities are valid. Missing WDL/interrupted analyses are never accepted as targets. Replaying a saved record reproduces its root. |
| [ ] | D06 / P1 | D05 | Run the 1,000-position annotation benchmark separately from worker implementation. | Save actual throughput, failures, settings and returned search information. Project annotation/replay/encoding/storage costs for 10,000, 100,000, several hundred thousand and 1 million labels, identifying measured rates and scaling assumptions. Choose expansion using these costs and subsequent learning curves. |
| [ ] | D07 / P1 | D05 | Implement record loading, shared-adapter tensor construction, batching and optional versioned cache validation. | A saved record yields the same state/mask/label after reload. Reject incompatible versions and invalid labels. A teacher class ID remains a class ID, not a dense stored one-hot policy. |
| [ ] | D08 / P1 | D06, D07 | Produce the initial 10,000 successfully labeled positions using the measured worker and audited splits. | Manifest, success counts, split overlap and replay/target audits pass. Failures/exclusions are accounted for rather than counted as completed labels. |

## 4. Build the network against the interface

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | M01 / P1 | F02, F03 | Port compact GAB and its shared template bank. | Match the pinned reference operations numerically on synthetic nontrivial inputs. Confirm d1=8, d2=32, d3=32 and that the template parameter is shared, not copied per layer. |
| [x] | M02 / P1 | M01, A07 | Assemble the projection, additive square embeddings and specified Post-LayerNorm encoder blocks. | Correct `[B,64,256]` outputs, finite forward/backward values and working train/eval dropout behavior for several batch sizes. Record resolved normalization/bias/initialization settings. |
| [x] | M03 / P1 | M02 | Implement the pair policy head, `d_policy=128`, promotion adjustment and exact flattening order. | Compare selected ordinary/promotion logits to an independent scalar calculation/reference. Output is `[B,4352]`; queen and underpromotions occupy their own correct slots. |
| [x] | M04 / P1 | M02 | Implement mean pooling and the specified WDL head. | Output is `[B,3]` logits in WDL order. Pool across squares; one known synthetic case independently checks the head path. |
| [x] | M05 / Both | M03, M04 | Assemble the backbone and heads with GAB as an intermediate correctness stage, before adding relative terms in M06. | End-to-end adapter batches produce finite policy/WDL logits and gradients. Record intermediate parameters/memory; this stage is not the architecture for supervised or RL runs. |
| [x] | M06 / P1 | M05, F03 | Port all implemented Q/K/V relative terms and finalize the fixed combined architecture. Keep operations inspectable and a diagnostic disable path for correctness checks. | Match reference outputs and gradients with nonzero relative tables and asymmetric inputs. Disabling terms recovers GAB-only behavior; all three tables receive gradients when enabled. No omitted position-only cross term is added. Check finite combined-model forward/backward results and count full parameters/memory; resolve a material mismatch with the provisional 6–8M target before freezing a full run. |

M06 is required before supervised training, the real evaluator and the combined-model pilot. It remains on the training critical path by the accepted architecture choice. M05 is an intermediate implementation check, not a competing training track; dummy search and data preparation can proceed independently of M06.

## 5. Build search using a dummy evaluator

Leela is the practical chess-search/self-play reference; AlphaZero supplies the underlying policy-improvement loop. Use our agreed WDL, move vocabulary and ending policy throughout.

| Done | ID / owner | Prerequisites | Build | Acceptance check |
|---|---|---|---|---|
| [x] | S01 / P2 | A07, F02 | Implement a uniform-legal-prior, zero-value dummy evaluator with `evaluate(states) -> list[Evaluation \| TerminalResult]` in input order. Use a batch-of-one call from search initially. | Empty, batch-of-one and mixed-terminal batches preserve order/input histories. Priors sum to one over legal moves only. Terminal states return exact outcomes and never enter policy normalization. |
| [x] | S02 / P2 | S01 | Define search configuration and implement tree edges, selection, expansion and legal traversal. Record the PUCT formula/constants, tie handling and simulation-count convention. | Fixed seed/config yields repeatable traversals; input state/history is unchanged. Distinct histories are not merged by FEN-only identity. Root expansion versus counted edge visits is explicit. |
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

**Historical pre-review sequencing (superseded by the dated acceptance entries below):** F01, F02 and A01–A06 are complete in this implementation. A07 automation is complete; obtain the second-person review before marking A07/Gate A accepted. The next implementation task is **F03**. F03 can proceed while adapter coding continues. After Gate A, P1 works on the D/M tracks and P2 on S; integrate at T04/R04. Complete E01 after S05 and before P03 so checkpoint evaluation is available for the pilot. Z02 can be developed once a selected checkpoint is available, alongside the report.

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
| F03 — complete | 2026-10-07 | `docs/upstream-sources.md`, `docs/upstream-lock.json`, `docs/notices/`, `scripts/check_upstream.py`, `runs/f03/source-check.json`, README guidance | `python3 /private/tmp/rl-chess-f03-change/scripts/check_upstream.py --maia3-checkout /private/tmp/rl-chess-f03-maia3 --lczero-checkout /private/tmp/rl-chess-f03-lczero`; `python3 /private/tmp/rl-chess-f03-change/scripts/check_upstream.py` (HTTPS); `.venv/bin/python scripts/check_upstream.py --maia3-checkout /private/tmp/rl-chess-f03-maia3 --lczero-checkout /private/tmp/rl-chess-f03-lczero` (repository copy) | Acceptance passed: 2 verified full commit objects/repository identities, 10 full-file SHA-256 hashes, 18 required entries, 43 immutable symbol/range references, 2 verbatim notices. Independent HTTPS and Git bytes agree; five isolated corruptions rejected. WDL/history/preset differences, depth-before-head relative tables, promotion scaling and constructor-delegated initialization recorded. No numerical port validation, model dependency or model implementation. Original planning files preserved. A07/Gate A still awaits the other member's replay/sign-off. |

F03 source retrieval was prepared in temporary directories and repeated against
the repository copy using the command above. `runs/f03/source-check.json` saves
the exact discovery/retrieval commands and passing Git/HTTPS reports. Initial
network access failed in the sandbox and the HTTPS check passed with approved
network access. No source operation was executed. The five corruption checks
used isolated temporary copies of the source lock/notices; all were rejected.
The original planning hashes still match the saved foundation hashes, and
repository `plan1.md`/`interface.md` remain byte-identical. Only this completion
log is updated in the repository planning files; the task roadmap is preserved.

All milestone-1 commands above actually ran in RL-Chess. Final XML reports have 119 tests,
zero failures and zero errors. `runs/milestone-1/environment.json` records installed
versions; `replay-check.json` records feature/mask fingerprints, legal IDs and
terminal results; `source-plan-checksums.json` verifies originals were preserved
and copies matched before editing this log. No applicable AGENTS.md was found in
either project or the checked ancestor paths. No subagents, annotation, training,
teacher engine, GPU or paid compute were launched.

**Historical pending-review status (superseded by the 2026-10-07 acceptance below):** the other project member should independently run the
README acceptance commands, inspect the contract fixtures and confirm identical
results. The automation establishes reproducibility, not second-person approval.
After that sign-off, mark A07 and Gate A accepted. The reviewer independently
verified commit `868371c`, 119 tests in both existing environments and all 13
replay records; this does not supply the other member's acceptance. F03 is now
complete. Proposed next implementation task after this commit's reviewer
assessment: M01, compact GAB with numerical reference checks. D01/S01 and all
model work remain pending. No upstream operation has been ported yet.


### Review acceptance — 2026-10-07

The user's reviewer decisions supersede the historical pending statuses above.
A07/Gate A is accepted through independent AI review of
`868371c278e8d96d73ea1d6f00e8e5c1736281b8`: 119 tests passed in each of the two
existing environments and all 13 replay fixtures matched. This was AI review,
not a human teammate's replay/sign-off. The original check results remain intact.

F03 commit `5958fd638fce0b6cf98a6631bacf743ce89551b8` passes independent review,
including the source checker and all five corruption cases. F03 and A07 are now
checked; Gate A is accepted. M01 only is authorized next. No later implementation
or paid compute is authorized by this entry. Changed files: `coding-plan.md`,
`README.md`, `docs/upstream-sources.md`; documentation-only correction checked
with `git diff --check`. The original planning directory remains untouched.


### M01 completion — 2026-10-07

Compact GAB only is complete after numerical acceptance. Changed files:
`pyproject.toml`, `uv.lock`, `src/chess_rl/model/{__init__,gab}.py`,
`tests/{gab_checks,test_gab}.py`, `tests/fixtures/gab_reference.py`,
`scripts/check_m01.py`, `docs/m01-gab.md`, README/port map/this log and
`runs/m01/` evidence. PyTorch 2.8.0 is an optional pinned dependency; its resolved
dependencies/artifact hashes are locked. Adapter imports and default installation
remain independent of PyTorch. CPU macOS 14.5 arm64 / Python 3.12.12 was tested.

Actual checks: `.venv/bin/python scripts/check_m01.py --maia3-checkout
/private/tmp/rl-chess-f03-maia3 --output runs/m01/numerical-acceptance.json`;
`.venv/bin/python -m pytest --junitxml=runs/m01/pytest-model.xml`;
`.venv-acceptance/bin/python -m pytest --junitxml=runs/m01/pytest-adapter-only.xml`;
`scripts/check_adapter.py` in both environments; `uv pip check --python` against
each environment; F03's offline Git-object source check; `git diff --check`.
All passed: 139 model-environment tests; 119 adapter-only tests and one optional
model-module skip; all 13 replay fixtures matched in both environments.
The adapter-only environment has no installed PyTorch; its import boundary passed.
Exact commands/outputs and original-planning-file hashes are in `runs/m01/`.

Numerical acceptance used batches 1/2/5, float32 and float64, nonzero weights
and biases (including both norms), asymmetric noncontiguous inputs, all input
and parameter gradients, and a nonuniform scalar objective. The frozen pinned
method matched exactly. Independent functional maximum output/gradient errors:
float32 `1.0728836059570312e-6` / `9.238719940185547e-7` at
`atol=2e-5, rtol=2e-4`; float64 `1.5543122344752192e-15` /
`1.7763568394002505e-15` at `atol=1e-11, rtol=1e-10`.
Two generators retain identical bank parameter identity, have disjoint sets of
10 private parameter tensors each, accumulate both nonzero bank gradients with
zero error, and put the bank into the optimizer once (21 unique tensors,
186,064 elements total). A single SGD update also matched the expected update.

Limitations: CPU only; CUDA/MPS/reduced precision and other platforms are
unvalidated. Upstream did not pin PyTorch, so no original random-stream or
checkpoint compatibility is asserted. Full model counts/configuration belong
to later tasks. M02, attention integration, heads, relative terms, data
annotation, training and search remain unstarted. No subagents or paid compute.
Stop for review of the focused M01 commit; this completion is numerical
acceptance, not a new claim of independent reviewer sign-off on M01.


### M01 review acceptance — 2026-10-07

The user supplied independent **AI review acceptance** of M01 commit
`66eea36ba74ba811298ce8347a5153f562c474be`. The reviewer inspected the
implementation and reran 139 model tests, 119 adapter-only tests with one
expected skip, all 12 numerical cases, sharing checks, both 13-fixture adapter
replays, pinned-source checks and dependency checks. Original planning hashes
and the clean working tree were verified. No blocking findings were identified.
This is AI review acceptance, not human teammate review. The historical M01
completion and `runs/m01/` remain intact; this entry supersedes their pending
review status. Structured record: `runs/m02/m01-review-acceptance.json`.

### M02 completion — 2026-10-07

M02 only is implemented and passes automated acceptance; **stop for review**.
Changed files: `src/chess_rl/model/{attention,backbone,__init__}.py`,
`tests/{backbone_checks,test_backbone}.py`, `scripts/check_m02.py`,
`docs/m02-backbone.md`, appended M01 review record, README/port map/this status
log and `runs/m02/`. `EncoderBackbone` returns `[B,64,256]` from the full
`[B,64,128]` adapter input, retaining zero-initialized additive square embeddings,
eight independent 256-wide Post-LayerNorm blocks, packed biased Q/K/V,
eight heads, FFN width 256, existing M01 GAB and one shared template bank,
training dropout 0.1 at all four required sites and encoder-final LayerNorm.
No input/position dropout or Elo/time conditioning is present.

Actual commands and raw results are saved in `runs/m02/commands.json`; staging
paths and the verification runner are archived there. All passed: six
independent reference cases (batches 1/2/5 × float32/float64, zero dropout,
nonzero parameters/position embeddings, asymmetric noncontiguous inputs), all
182 parameter tensors plus input gradients, train/eval and each isolated
dropout site, sharing/gradient accumulation/optimizer deduplication across eight
blocks, all 13 real adapter fixtures with full history/state preservation,
M01's 12 numerical cases and sharing/source-byte audit, **157 full model tests**,
**119 adapter-only tests with two expected optional-model module skips**, both
13-fixture adapter replays, import boundary, dependency checks and pinned
source checks. Preservation checks confirm 45 protected files and the three
original planning hashes. The M01 extra skip count remains historically one.

Backbone only: **3,567,168** parameter elements, 182 unique tensors. Constructor
epsilon is 1e-5 with affine LayerNorm biases; all retained projection/FFN
biases enabled. Packed `[768,256]` Q/K/V is Xavier uniform; attention output
weight retains Linear constructor reset, and MHA zeros its biases. Other
Linears retain their constructors; norms initialize weight=1/bias=0.
`docs/m02-backbone.md` and the report resolve settings from the pinned source
and installed PyTorch 2.8.0. No final combined-model count is claimed.

Current execution status: F01/F02, A01–A07/Gate A, F03 and M01 accepted; M02
automated acceptance complete, independent review pending. M03 onward, heads,
relative terms, data, training and search remain unimplemented. CPU macOS arm64
only; CUDA/MPS, reduced precision, other platform builds, upstream framework
RNG/checkpoint compatibility and playing strength are unvalidated. No subagents
or paid compute. Original planning directory, adapter semantics, source
pins/notices and all earlier run evidence are preserved.


### M02 review acceptance — 2026-10-07

The user supplied independent **AI review acceptance** of M02 commit
`bc2066d57203459be18d60013eb874c88f4bc5c3`. The reviewer inspected the
implementation and reproduced 157 model tests, 119 adapter-only tests with
two expected skips, six backbone output/gradient cases, dropout/sharing checks,
all 12 M01 numerical cases, both 13-fixture replays, source/dependency checks
and preservation audits. The 3,567,168 backbone parameter count and clean
working tree were verified. No blocking findings were identified. This is AI
review acceptance, not human teammate review. Historical M02 evidence is
unchanged; this entry supersedes its pending status. Structured record:
`runs/m03/m02-review-acceptance.json`.


### M03 completion — 2026-10-08

M03 only passes automated acceptance; **stop for independent review**.
Changed files: `src/chess_rl/model/{policy,__init__}.py`,
`tests/{policy_checks,test_policy}.py`, `tests/fixtures/policy_reference.py`,
`scripts/check_m03.py`, `docs/m03-policy.md`, appended M02 AI acceptance,
README/port-map/this execution status, and `runs/m03/` evidence.
`PairPolicyHead` validates the public positive-integer `d_policy=128` field,
consumes `[B,64,256]` contextual tokens and returns raw `[B,4352]` logits.
Three bias-free Linears retain constructor initialization, ordinary
`q dot k / sqrt(d_policy)`, rank8 projected destination-key adjustments times
`sqrt(d_policy)`, and exact ordinary/promotion IDs and q/r/b/n order.
All slots remain; no legality, softmax or terminal handling is added.
Head count: **66,048 elements in three tensors** (`516*d_policy`).

Actual commands/outputs/tolerances are under `runs/m03/`; the verification
runner saves cwd, argv, environment, UTC start times, exit codes and durations.
All passed: 12 independent numerical cases (batches 1/2/5 × float32/float64 ×
widths 7/128), scalar checks of eight ordinary pairs and all 256 promotion
slots, all input/parameter gradients, exact full-slot flattening and piece
order, destination-key dependence, both scaling factors, constructor parity,
known IDs through adapter APIs and 13-fixture accepted-backbone diagnostic.
At atol/rtol 2e-5/2e-4 (float32), maximum output/scalar/gradient errors were
3.814697265625e-6 / 4.76837158203125e-6 / 5.7220458984375e-6;
float64 2e-12/2e-11 gave 3.552713678800501e-15 / 1.0658141036401503e-14 /
7.105427357601002e-15. Pinned policy-body output errors were zero.

Regressions: **187 model tests**, **119 adapter-only tests/three expected skips**,
M01's 12 cases, M02's six cases/dropout/sharing/constructors, both 13-fixture
replays, adapter import boundary, dependency and pinned-source checks passed.
Preservation matched 90 protected repository files and three original plans;
all prior runs, adapter, M01/M02 implementation/tests, source pins/notices,
interface/plan1 and dependency pins are unchanged. M02 historical document
is retained as an exact prefix with appended acceptance. Original planning
directory remains untouched. The prior one/two-skip reports are historical.

Current status: M01/M02 AI review accepted; M03 automated acceptance complete,
independent review pending. CPU macOS arm64 float32/float64 only; CUDA/MPS,
reduced precision, other builds, upstream framework/RNG/checkpoint equivalence,
performance and playing strength are unvalidated. No M04 onward, WDL, assembled
network, relative terms, data, training, search, subagents or paid compute.


### M03 review acceptance — 2026-10-08

The user supplied independent **AI review acceptance** of M03 commit
`83d8ac1bbaffe54bbf5cf3d02daf542982c03736`. The reviewer reproduced 187 model
tests, 119 adapter-only tests/three expected skips, all 12 M03 numerical cases,
M01/M02 acceptance, both 13-fixture replays, source/dependency checks and
preservation audits. The 66,048 head parameter count and clean working tree
were verified; no blocking findings. This is AI review acceptance, not human
teammate review. Historical evidence remains unchanged; this entry supersedes
M03's pending status. Record: `runs/m04/m03-review-acceptance.json`.

### M04 completion — 2026-10-08

M04 standalone WDL only passes automated acceptance; **stop for review**.
Changed files: `src/chess_rl/model/{wdl,__init__}.py`,
`tests/{wdl_checks,test_wdl}.py`, `tests/fixtures/wdl_reference.py`,
`scripts/check_m04.py`, `docs/m04-wdl.md`, appended M03 AI acceptance,
README/port map/this execution record and `runs/m04/` evidence.
`WDLHead(d_value=128)` consumes encoder-final `[B,64,256]` tokens, averages
64 squares, applies separate biased affine LayerNorm(256, eps=1e-5), biased
Linear(256,128), ReLU and biased Linear(128,3). Returns raw `[B,3]` in shared
WDL_ORDER `[win,draw,loss]`, from the input position's side to move. Newly
trained rows/targets follow our contract; upstream loss/draw/win interpretation
is not imported. No checkpoint compatibility is claimed. `d_value` is
independently named from `d_policy`; PyTorch 2.8.0 constructor resets remain.
Head-only count: **33,795 elements, six tensors** (`515+260*d_value`).

Twelve independent NumPy float64 forward/manual-backward cases cover batches
1/2/5, CPU float32/float64, widths 7/128, asymmetric noncontiguous inputs,
nontrivial weights/biases, active/inactive ReLU, output/input/all-parameter
gradients, square permutation, finite values and input/parameter preservation.
The known closed-form case checks pooling, affine normalization, ReLU and
three distinct contract rows. Seeded constructor parity, constant inputs and
13 real adapter fixtures through the accepted backbone also pass, preserving
board identity, every retained prefix, provenance, features and tokens.
At float32 atol/rtol 2e-5/2e-4, maximum output/gradient/permutation errors are
5.960464477539062e-7 / 1.6689300537109375e-6 / 7.152557373046875e-7.
At float64 2e-12/2e-11, maxima are 1.3322676295501878e-15 /
6.217248937900877e-15 / 8.881784197001252e-16. Pinned body errors are zero.
Regressions passed: **216 model tests**, **119 adapter-only tests/four expected
skips**, M01–M03 acceptance, both 13-fixture replays, adapter import boundary,
dependency/source checks and preservation of 118 baseline files plus three
original plans. Exact commands, raw results, constructor sources/hashes,
tolerances and preservation evidence are saved under `runs/m04/`; see `docs/m04-wdl.md`.

Current status: M01–M03 AI review accepted; M04 automated acceptance complete,
independent review pending. Softmax, value conversion, terminal handling and
outcome labels remain consumers. M05 onward, assembled network, relative
terms, data, training, evaluator and search remain unimplemented. CPU macOS
arm64 float32/float64 only; CUDA/MPS, reduced precision, other builds,
checkpoint/RNG equivalence, throughput and strength are unvalidated.
No ponder/auxiliary heads, subagents or paid compute. Original planning files,
shared contracts, accepted implementations, source pins/notices and all prior
run evidence remain unchanged.


### M04/M05 acceptance synchronization — 2026-10-08

The already-recorded user-supplied independent AI acceptance of M04 commit
`0fdbd1eaf5e4d1701a2627d55f87b2150a4f252a` and M05 commit
`2e6316dd0c49b0c22ff10ef72b55793856189a72` now updates this shared status.
M04's reviewer reproduced 216 model tests, 119 adapter tests/four expected
skips, all 12 WDL cases, synthetic/permutation/constructor checks, M01–M03
acceptance, both 13-fixture replays and source/dependency/preservation checks;
33,795 elements, WDL order, change manifest and clean tree were verified.
Source of this attribution: `runs/m05/m04-review-acceptance.json` and appended
`docs/m04-wdl.md`, already present in the D01 base commit.
M05's reviewer reproduced 251 model tests, 119 adapter tests/five expected
skips, M01–M05 acceptance, both 13-fixture replays and source/dependency,
manifest/preservation and parameter/memory audits; clean tree verified.
Source: existing `runs/m06/m05-review-acceptance.json` and appended
`docs/m05-network.md`. No blocking findings were recorded. These were AI
reviews, not human teammate review. Historical completion entries remain intact.

### M06 review acceptance — 2026-10-08

The user reports that the planning agent independently reviewed and accepts
M06 commit `4765a8cb3d6ae9cf50619be283176ab9ac17cca6` for **local CPU
correctness**. That reviewer reproduced **287 model tests**, **119 adapter
tests with six expected skips**, the complete M06 numerical checker, both
13-fixture adapter replays and **191 protected-file hashes**. The working tree
was clean. This evidence is supplied by the user in the D01 request; it is
not a new reviewer run performed by the D01 implementer. This is independent
AI review, not human review. GPU correctness and performance remain unverified.
Record: `runs/d01/m06-review-acceptance.json`; appended `docs/m06-relative.md`.
The historical M06 automation and pending-review statements remain preserved;
this dated entry supersedes the pending status.

### D01 manifest preparation — 2026-10-08

D01 only passes automated preparation checks and **stops for independent review**;
it is not independently reviewed. Changed files: `src/chess_rl/data_manifest.py`,
`data/manifests/elite-2025-01-v1.json`, `scripts/check_d01.py`,
`tests/test_data_manifest.py`, `docs/d01-data-manifest.md`, appended M06 review,
README/shared status and `runs/d01/` evidence. The selected January 2025 archive
URL is index/HEAD verified; no archive body was fetched. Archive ZIP SHA-256,
retrieval time/byte count remain null pending D02. Split/sampling seeds are
20261008/20261009; replay JSONL stores authoritative games once and sample
references by game/ply. D03/D04 algorithms, phase boundaries and split/overlap/
protected-heldout evidence remain pending. Annotation readiness fails closed.
Initial target: 10,000 successful labels; separate benchmark: 1,000 positions;
expansion and archive coverage/costs require measurements.

Stockfish 17.1 official source release was locally built for Apple silicon
using Apple clang 15.0.0 and the non-PGO build target after the official binary
asset host repeatedly reset connections. The pinned source/tag, source-package,
NNUE and executable hashes plus full build/idle-UCI evidence are in `runs/d01/`.
Executable SHA-256:
`3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`.
The teacher contract stays 100,000 nodes, Threads=1, Hash=128 MB, MultiPV=1,
UCI_ShowWDL=true, full-history roots, per-annotation Clear Hash and root WDL;
failed/missing/interrupted outputs require retry or explicit exclusion.
No annotation searches, benchmark, training, search, paid compute or subagents.

Actual command/exit-code evidence: `runs/d01/commands.json`; focused acceptance
and limitations: `docs/d01-data-manifest.md`. Plan1/interface semantics, accepted
adapter/model bytes, dependency/source pins and every previous run remain
preserved. S01 is not marked complete; its concurrent work is outside this
worktree. D02–D08 are unimplemented. GPU correctness/performance remains
unverified. D01 preparation readiness does not authorize annotation.

D01 check results: 83 focused tests; full model environment 370 passed
(287 historical + 83 D01, with the M06 diagnostic plugin); adapter environment
202 passed/six expected skips (119 historical + 83 D01). The complete M06
checker, both 13-fixture replays, pinned-source audit, Torch-free imports and
raw/python-chess idle UCI probes passed. Success checks exit 0; the incomplete
manifest annotation gate exits 1 as required. Preservation evidence and focused
file inventory: `runs/d01/preservation-check.json`, `runs/d01/change-manifest.json`.


### D01/S01 AI review acceptance and integration — 2026-10-08

The user reports independent planning-agent **AI review acceptance** of D01
commit `df58f37de792cee248a89ff6644d77b144360a1f` for manifest preparation
and S01 commit `834eab9317d96547333b4958c863b503f02ecba5` for CPU evaluator
plumbing. S01 was accepted on its separate `codex/s01-dummy-evaluator` branch,
which remains available. The updated repository context confirms that **both
accepted tracks are now integrated** in `main` merge commit
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`, the common D02 starting point.
The reported combined checks passed: 403 model tests, 235 adapter tests with
six expected skips and both 13-fixture replays; pinned teacher executable,
source-package and embedded-NNUE hashes plus a fresh idle UCI check passed.
There were no blocking findings. This is user-supplied independent AI review
attribution, not human teammate review or a new reviewer run by the D02
implementer. Historical evidence and pending-status entries remain intact;
this dated entry supersedes their D01/S01 status. Structured attribution:
`runs/d02/review-integration.json`. D02 has no independent review acceptance.


### D02 PGN source replay completion — 2026-10-08

D02 only passes automated acceptance and **stops for independent review**.
It has no independent review acceptance. It is on `codex/d02-pgn-replay` in
`/private/tmp/rl-chess-d02`, from merged base
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`, with both accepted D01/S01 tracks.
Shared README/status ownership records both AI review acceptances as integrated
work; S01's separate accepted branch remains available. S02 is not implemented.

Changed core files: `src/chess_rl/pgn_replay.py`, `scripts/prepare_d02.py`,
`tests/test_pgn_replay.py`, `tests/fixtures/d02_sources.json`, selected manifest
and its test expectations, `docs/d02-pgn-replay.md`, README/this status, and
focused `runs/d02/` evidence. Standard PGN mainlines are parsed with
python-chess and validated/rebuilt by the shared adapter. Genuine initial
boards, full actual UCI moves and complete-history/source provenance survive.
Malformed/variant/setup/null/illegal/parser-error sources are rejected whole;
partial parser prefixes never become accepted games. Legal replay can cross
episode draw cutoffs. Explicit '*' sources remain unfinished source metadata;
missing moves/results are never inferred or made into teacher labels.

Exact selected ZIP retrieval: 82,724,777 bytes, SHA-256
`f2fa14565bcdaba7ad6de6a4f8f2348d0f9f8f262935c46e61925c5acce6f7b7`,
UTC `2026-10-08T07:45:09.776038+00:00`, HTTP 200, no redirects, final URL equal
to the requested January 2025 URL. Manifest retrieval fields were resolved
together. Full selected archive: **1 archive / 1 member / 1 PGN member**,
**289,776 source games accepted / 0 rejected**, **25,429,063 actual plies**.
No unfinished sources, result discrepancies or declared setups occurred in
this member. ZIP/member hashes, identities, CRC and counts are saved. These
are source/replay measurements before D03 deduplication, splits or D04 sampling;
no sampling coverage or label success is claimed.

The final strict full run exited 0. Three full passes produced byte-identical
record files and summaries; the first two began before additional syntax
hardening. The all-source output audit matched ordinals, byte/line intervals,
source hashes, member/output identities and counts against the raw archive.
Archive and bulk JSONL remain outside Git; authoritative output directory is
`/private/tmp/rl-chess-d02-accepted-prepared`. Final records: 541,174,538 bytes,
SHA-256 `69c10231f4d569a6b3cb4b26bc22a9535346d68f0a4753bd83c0c92981ea950b`.

Actual final checks: **155 focused tests** (72 D02 + 83 D01), **475 model
tests**, **307 adapter tests / six expected skips**, both 13-fixture replays,
complete M06 checker, pinned-source/import-boundary checks, exact teacher/source/
embedded-NNUE hashes and fresh idle UCI; all exit 0. Annotation gate exits 1
as required. Preservation matched 305 protected baseline files. Exact argv,
exit codes, UTC timestamps, development failures, checksum evidence and raw
results are retained under `runs/d02/`; final inventory is
`runs/d02/change-manifest.json`. The teacher executable remains the exact D01
bytes at `/private/tmp/Stockfish-sf_17.1/src/stockfish`; no rebuild, substitution
or search command was issued.

D03 canonical IDs/duplicate consolidation/split assignment and D04 sampling/
phase/overlap/heldout algorithms and audits remain unresolved. The six manifest
blockers still prevent annotation. No splitting, sampling, annotation, benchmark,
training or search is implemented. Plan1/interface semantics, dependency pins,
accepted implementation and all historical evidence are preserved. CPU macOS
arm64/Python 3.12.12 intake only; unsupported syntax is rejected rather than
salvaged. Review the focused D02 commit before starting D03.


### D02/S02 independent AI review acceptance and integration — 2026-10-08

At the user's request, the coordinating Codex agent independently reviewed
D02 `cb37b0df79f85ee7615d89778efbad75c1b19576` and
S02 `88e62a418bfd8cce54bc8c7e90bae60bb5719c6a` from common base
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`. Both scoped milestones
are accepted with no blocking findings and no production fixes. This is fresh
AI review, not an attribution to the implementers or human teammate acceptance.

Fresh checks: 155 D02 tests and 95 S02 tests on their original commits;
570 combined model tests; 402 combined adapter tests with six expected skips;
26 reviewer-authored cases; both identical 13-fixture replays; M06, pinned
source/notice, teacher-byte/idle-UCI and preservation checks. Full raw/output
audits matched all 289,776 source records and 25,429,063 stored plies; 291 games
were freshly parsed/replayed. Three saved bulk outputs were byte-identical.
A fresh full-archive legal replay was not rerun. See `docs/d02-s02-review.md`
and `runs/d02-s02-review/acceptance.json` for exact scope, commands and limits.

The independent commits merge without conflicts. Original branches/evidence
remain available. This entry supersedes historical pending-review statements.
D03 canonicalization/splitting and S03 backup are the next scoped tasks.
D04 sampling/phase/overlap decisions still follow D03; annotation remains
blocked. S04/S05 remain pending and **Gate S is not accepted**.
