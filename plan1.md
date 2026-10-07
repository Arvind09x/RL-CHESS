# Plan 1: Stockfish pretraining followed by MCTS self-play RL

Last revised: 7 October 2026

Execution companion: [coding-plan.md](coding-plan.md) breaks this strategy into small coding/run tasks, dependencies, owners and acceptance checks. [interface.md](interface.md) remains the shared behavior contract.

## Project scope and current status

This is a three-week reinforcement learning course project for two people. We will build our own small chess transformer with MCTS, initialize it using Stockfish supervision, and improve it through MCTS-guided self-play where the available budget permits.

At the time of this plan, the local project repository has no implementation, datasets, or experiment results. The decisions below consolidate the project discussions. Proposed dimensions and compute allocations are planning targets, not measured results.

## Project goal and required RL assessment

Scope revised on 7 October 2026: the goal is a working, capable chess model with MCTS and an honest assessment of what self-play training changes. Matches between the frozen pretrained checkpoint P and RL checkpoints R are required. Comparisons with and without MCTS and attention ablations remain outside the required scope. Demonstrating a statistically significant RL improvement is not a completion condition.

The intended pipeline is supervised pretraining, integration with MCTS, self-play training, and a playable evaluated agent. Save the initial supervised checkpoint as immutable P and subsequent self-play checkpoints as R for reproducibility, recovery and mandatory checkpoint evaluation. One implementation loads either checkpoint; separate baseline and RL code branches are unnecessary.

Assess the final model with MCTS using legal play, tactical positions, matches against fixed opponents, P-versus-R matches and measured search cost. Compare P and R with identical search settings and paired openings with colors reversed. Direct policy play without search remains an optional diagnostic.

Self-play from random initialization is outside the initial implementation scope. Use the locked 128-channel representation throughout the project.

## Candidate network

| Component | Latest candidate |
|---|---|
| Tokens | 64, one per board square |
| Input features | Locked: 128 channels per square, projected to width 256 |
| Encoder layers | 8 |
| Embedding width | 256 |
| Attention heads | 8 |
| Feedforward width | 256 initially |
| Positional information | Locked: absolute square information + compact GAB + the authors' implemented Q/K/V Shaw-style relative terms |
| Outputs | Locked: 4,352 move-policy logits and three-class win/draw/loss probabilities |
| Policy head | Locked: Chessformer-style source-destination scoring with explicit promotion handling |
| Initial backbone defaults | Maia-3-style Post-LayerNorm, GELU, learned additive square embeddings; training dropout 0.1 |
| Initial WDL head | Mean pooling, LayerNorm, Linear(128), ReLU, Linear(3) |
| Board orientation | Locked: side-to-move perspective; reflect ranks for Black and preserve files |
| Parameter budget | Approximately 6–8M target; count the complete implemented model before freezing the configuration |

Compact GAB compresses each square to eight features, forms a 32-feature board summary, and generates 32 coefficients per head over a shared bank of attention-bias templates. Use the published small-model configuration as an implementation reference.

Implementation defaults clarified on 5 October 2026: use newer Chessformer/Maia-3 as the primary backbone/head reference and the earlier authors' code only for the accepted Q/K/V relative operations. The exact initial model and shared adapter API/provenance rules are in `interface.md`. Use Leela as the practical reference for chess search/self-play, with the final 2018 AlphaZero paper/supplement supplying the foundational learning loop. These choices fit our scope but do not establish that the hybrid is optimal. Keep PUCT for initial integration; later Gumbel search is optional only after the working pilot exposes a need. Preserve Stockfish initialization and the existing input/game-ending contracts.

Decision revised on 7 October 2026: fix the architecture as GAB plus the Q/K/V relative-position mechanisms in the earlier Chessformer authors' released `lczero-training/tf/tfprocess.py`, throughout supervised and RL training. Port `make_rpe_map`, `RPELogits`, `RPEValue`, and the attention operations to PyTorch; retain Maia-3's compact GAB and source-destination policy head. Keep the operations separately inspectable and permit disabling relative terms for correctness checks, without a second training track. Replace Maia-3's human-specific input construction with our locked 128-channel input and omit Elo/time conditioning. Do not substitute Maia-3's static `use_relative_bias` for these mechanisms. Pin the exact upstream commits when importing or porting code and record them in model metadata.

The implemented Shaw-style variant includes query-content/position and key-content/position logit terms plus relative values. It omits the position-only `dot(aQ, aK)` term implied by Equation 6 of the earlier paper. Follow the released implementation, rather than adding that term. No confirmed author rationale for the discrepancy has been found; the term's interpretation as a static relative bias is a mathematical observation, not evidence of an author decision. See `interface.md` for the exact operations and naming.

Validate the combined model's operations and gradients against pinned reference operations before training. Profile its training throughput, inference throughput at realistic search batch sizes, memory use, full parameter count and held-out prediction quality during the pilot. Use those measurements to set practical batch sizes, dataset scale and search budgets. There is no GAB-only training comparison or attention-selection gate; superiority of the combination over GAB-only remains untested.

Measure the actual combined model's cost. Earlier relative-arithmetic estimates are not a runtime prediction and do not set the allocation.

The combined architecture is fixed before implementation and used for subsequent self-play training. Its relative-attention port remains on the training critical path; budget time for numerical verification and fixes. Version any later architecture changes explicitly. Transformer loops are outside the initial implementation.

## Inputs, outputs, and conventions

### Locked input representation: 128 channels

Decision accepted on 2 October 2026: use the Leela-style 112-channel history encoding specified in the newer Chessformer paper's Appendix A.2, add the 15 engineered channels from Inputs V2.0 in Representation Matters, and add one explicit en-passant square mask.

Decision reaffirmed on 5 October 2026: keep 128 channels. Each of the eight snapshots contains only 12 piece-placement channels and one repetition flag. The eight current-position metadata/constant channels, 15 engineered channels, and explicit en-passant mask are supplied once for the current position. Historical engineered-feature extensions are excluded from the implementation plan.

A snapshot is the complete board position at a particular half-move: the current position or one of the seven preceding positions. The piece-placement channels encode the locations of all pieces, not just their counts. The channel allocation is `8 * (12 + 1) + 8 + 15 + 1 = 128`.

This is our hybrid representation. Neither paper establishes the performance of this exact combination. The base is a specific published Leela-style schema, not a claim that every Leela encoding version is identical or that 112 channels outperform AlphaZero's 119.

| Feature group | Channels | Meaning |
|---|---:|---|
| Piece placement across eight snapshots | 96 | Current position and seven preceding half-move positions; 12 piece channels per snapshot |
| Repetition indicators across those snapshots | 8 | One repetition indicator per snapshot, following the specified base |
| Castling rights | 4 | Kingside and queenside rights for the current player and opponent |
| Actual player color | 1 | Whether Black is to move |
| Halfmove clock | 1 | Plies since the last pawn move or capture, divided by 100 in the published base |
| Constant channels | 2 | One zero channel and one one channel, retained from the published base |
| Occupancy masks | 2 | All current-player pieces and all opponent pieces |
| Checkerboard pattern | 1 | Alternating square-color pattern |
| Material differences | 5 | Current-player minus opponent counts for pawns, knights, bishops, rooks, and queens |
| Opposite-colored-bishops indicator | 1 | Use the engineered-feature paper's exact condition |
| Checking-piece mask | 1 | Locations of pieces giving check to the current player's king |
| Current-player material counts | 5 | Counts of pawns, knights, bishops, rooks, and queens |
| Explicit en-passant square mask | 1 | Current en-passant target square; zero when unavailable |
| **Total** | **128** | **112 base + 15 engineered + 1 explicit en-passant** |

The tensor contract is `[batch, 64, 128]` before projection and `[batch, 64, 256]` afterward. It includes eight board snapshots plus current-position metadata and engineered summaries. Historical piece ownership and square coordinates must use one consistent current-player perspective across the entire history window.

Keep complete rule-relevant history in the environment for legal moves and repetition adjudication; the eight-snapshot network input is not a replacement for that history. An isolated FEN does not provide the historical snapshots or full repetition record.

Decision accepted and padding revised on 5 October 2026: the detailed conventions are locked in [interface.md](interface.md), encoding version `chess128-v2`. Use newest-first 13-channel history blocks; our and opponent `P,N,B,R,Q,K` followed by a second-occurrence-or-later repetition flag. Fill absent older snapshots by repeating the earliest available position, following Maia-3's history-padding convention. Copy its encoded block and real-history repetition flag without counting padding as additional occurrences or inventing moves. With only an isolated FEN, repeat the current position and keep repetition flags at 0. This supersedes the earlier zero-padding decision and applies throughout pretraining, self-play and evaluation. Current castling channels are our queenside, our kingside, opponent queenside, opponent kingside. Keep actual Black-to-move, halfmove clock divided by 100, zero and one constants, then the 15 engineered channels and legal en-passant mask. Divide every engineered material count/difference by 8 without clipping; checkerboard is network-coordinate parity with a1 equal to 0. The bishop indicator requires exactly one bishop per side on opposite colors, regardless of other material. Repetition uses each real snapshot's own historical prefix. Preserve the halfmove clock's rule meaning: castling does not reset it.

Use this identical 128-channel representation in supervised pretraining, self-play, and evaluation. GAB + Q/K/V relative attention is fixed; the complete model parameter count and practical resource costs are measured in implementation and the pilot.

Use the side-to-move perspective consistently:

- The WDL head predicts probabilities for the whole position, not a separate value for each move.
- MCTS converts WDL predictions to `v = P(win) - P(loss)`.
- Value perspective changes across a move, so backup must handle the corresponding sign change.
- The policy head scores the fixed move vocabulary; illegal moves are masked before normalization.
- Promotions, castling, en passant, and any board-orientation transformations must use the same move conventions throughout.

Keep inputs, outputs, and conventions consistent across the checkpoints of a given model configuration.

### Locked coordinate convention: reflect ranks for Black

Decision accepted on 5 October 2026: orient inputs so the current player is at the bottom and pawns advance toward increasing network-view ranks. White-to-move positions retain their square coordinates. For Black-to-move positions, reflect ranks while keeping files unchanged:

`network_file = actual_file`

`network_rank = 9 - actual_rank` for ranks numbered 1 through 8, or `network_rank_index = 7 - actual_rank_index` for zero-based ranks.

This is a vertical reflection, not a 180-degree rotation. Examples for Black to move:

| Actual coordinates | Network-view coordinates |
|---|---|
| Black king on e8 | Our king on e1 |
| Black pawn on a7 | Our pawn on a2 |
| e7 to e5 | e2 to e4 |
| Kingside castling e8 to g8 | e1 to g1 |

Piece channels describe the current player and opponent. Apply the current position's ownership convention and coordinate transform consistently to all eight history snapshots, even when a historical snapshot had the other player to move. Keep the actual-color indicator so Black-to-move positions are still explicitly identified.

Transform all square-based inputs, including occupancy, checkers, and en-passant masks. Castling rights remain kingside/queenside rights assigned to the current player and opponent; preserving files does not swap the two flanks. Define the checkerboard pattern in the same network coordinate system.

Teacher moves, legal masks, MCTS policy targets, and move IDs use network-view coordinates. Map the chosen move back to actual coordinates before applying it to the chess environment. The rank reflection is its own inverse. Complete environment history and legal move generation remain in actual game coordinates.

### Locked policy head: source-destination scoring

Decision accepted on 5 October 2026: use a Chessformer-style source-destination policy head rather than a single global linear layer over all moves.

From the 64 contextual square vectors returned by the transformer, learned projections produce origin query vectors and destination key vectors. Score each origin-destination pair with a scaled dot product, `score(i, j) = dot(q_i, k_j) / sqrt(d_policy)`, yielding a 64-by-64 logit matrix. Each square vector already includes context gathered from the whole board.

Add explicit promotion scoring so queen, rook, bishop, and knight promotions have distinct logits even when their origin and destination coincide. Use the published promotion mechanism as the implementation reference. Castling and en-passant moves use their canonical origin and destination coordinates; legality comes from the shared chess adapter.

Map the pair and promotion logits into a fixed move-ID vector, mask illegal moves, and apply one softmax over all legal move logits. This scores moves jointly in one network evaluation; it does not choose an origin first and a destination second. Handle terminal positions before policy normalization, since they may have no legal moves.

The selected head shares parameters across square pairs and fits the square-token representation. Its selection is not a claim of proven Elo superiority over a pooled linear head. Use the locked move vocabulary and output ordering below across pretraining, self-play, and evaluation.

### Locked move vocabulary: 4,352 outputs

Decision accepted on 5 October 2026: follow the released Chessformer/Maia-3 output layout, with 4,096 ordinary origin-destination slots followed by 256 explicit promotion slots. Retain this layout for the initial experiment rather than compacting promotions to 88 slots. Matching the reference reduces custom mapping work; the extra slots share scoring parameters and do not each introduce independent learned weights.

A move ID is the zero-based index of a move's logit in the policy output vector. It is a fixed address, not a move score or rank. Define one shared, reversible table of `(origin_square, destination_square, promotion_piece)` tuples and use it for model outputs, Stockfish labels, legal masks, MCTS visit targets, and move selection.

All squares and moves in this table use network-view coordinates after the locked side-to-move transform:

- Square order is `a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63`, with `square = 8 * rank_index + file_index` and zero-based rank/file indices.
- Ordinary slots have `promotion_piece = none` and `move_id = 64 * origin_square + destination_square`, giving IDs 0 through 4,095. Flatten the 64-by-64 pair-score matrix in this same order, with destination varying fastest.
- Promotion origins are rank 7 and destinations are rank 8. Enumerate origin files `a` through `h`, then destination files `a` through `h`, then pieces `[queen, rook, bishop, knight]` with indices `[0, 1, 2, 3]`.
- Promotion IDs are `4096 + 4 * (8 * origin_file_index + destination_file_index) + promotion_piece_index`, giving IDs 4,096 through 4,351. Each corresponding logit uses the origin-destination pair score plus the destination-derived adjustment for that promotion piece, following the released model implementation.

Examples in network-view coordinates:

| Move | ID |
|---|---:|
| e2 to e4, no promotion | 796 |
| e7 to e8, promote to queen (`e7e8q`) | 4,240 |
| e7 to e8, promote to rook (`e7e8r`) | 4,241 |
| e7 to e8, promote to bishop (`e7e8b`) | 4,242 |
| e7 to e8, promote to knight (`e7e8n`) | 4,243 |

The 256 promotion slots are `8 origin files * 8 destination files * 4 pieces`; only 88 have pawn-compatible geometry, and legality still depends on the current board. Permanently impossible slots remain in the vocabulary and are always masked. The ordinary e7-to-e8 slot does not stand for a queen promotion: when a pawn promotes, only the explicit promotion IDs can be legal.

Policy logits have shape `[batch, 4352]`. Legal masks and MCTS policy targets use the same 4,352-slot ordering. Construct each legal mask by enumerating actual legal moves in the chess environment, transforming them into network-view coordinates, and looking up their IDs. Mask illegal logits before softmax and policy-loss normalization; they receive zero probability and no policy-loss gradient. Convert the selected legal ID back to actual coordinates before applying the move. Handle terminal states before normalization.

Castling uses the king's origin and destination (for example, e1 to g1), and en passant uses the pawn's origin and landing square; both use ordinary slots. Keep this vocabulary and its version in dataset and checkpoint metadata. Compacting it is an optional later optimization requiring evidence of a useful benefit, not a prerequisite for the pilot.

## Stage 1: supervised pretraining

Train a randomly initialized transformer, including GAB and all accepted Q/K/V relative parameters, using Stockfish move targets and WDL estimates. "Pretrained" means our own resulting checkpoint, not downloaded weights.

Decision revised on 7 October 2026: retain 10,000 labeled positions for plumbing and the initial learning pilot. This is not a promised sufficient dataset for a capable 6–8M-parameter model. Consider staged expansion to 100,000, several hundred thousand and potentially 1 million labels using measured annotation/replay/encoding/storage costs, training throughput and held-out learning curves. Remove the former 50,000 expansion target as a ceiling; a million-position run is not mandatory. Use one fixed monthly Lichess Elite PGN archive, recorded with its URL and checksum in the dataset manifest before annotation; audit whether it supplies enough distinct games under the sampling policy before choosing the main scale.

Replay standard games to preserve full history. Sample at most eight nonterminal positions per game, spread across game progress, with opening, middlegame, and endgame coverage. Use a fixed sampling seed and record the actual sampling settings. Use fixed Stockfish settings: 100,000 nodes per position initially, Threads=1, Hash=128 MB, MultiPV=1, UCI_ShowWDL=true. Benchmark 1,000 positions before scaling. Pin the binary/version and settings in the manifest; pilot evidence may justify an explicitly versioned change to the annotation budget.

Store one best-move ID and soft [win, draw, loss] teacher targets from the root side-to-move perspective. Keep the PGN result separate from Stockfish WDL estimates. Save game identity, starting FEN, replayable moves, sampled ply, root FEN, targets, teacher metadata and encoding version. FEN-only positions repeat the current snapshot to fill history and carry incomplete-history metadata. See `interface.md` for record contracts.

Split 90/5/5 into training/validation/test by game before extracting positions. Check for overlapping positions across splits and remove overlapping samples before annotation. Report opening/middlegame/endgame counts per split before and after removal, because common opening overlap can shift held-out composition. Keep validation and final test data separate. Expansion preserves game assignments and protected held-out sets, repeats overlap audits, and records actual split counts.

Save an immutable initial checkpoint P, including its configuration and preprocessing version. Continue self-play training from that checkpoint, saving subsequent versions as R. Evaluate frozen P against R through the same match runner; record which P initializes each RL run.

## Stage 2: MCTS-guided self-play RL

1. Load the current network and generate games with network-guided MCTS.
2. At each position, save its encoded state, legal mask, and root visit distribution.
3. After the game finishes, attach the win/draw/loss outcome from each saved position's side-to-move perspective.
4. Train the policy toward the visit distribution and WDL toward the completed-game outcome, while mixing in training-split Stockfish samples with their supervised targets.
5. Save a checkpoint and repeat.

Use policy cross-entropy plus a weighted WDL cross-entropy as the initial training objective. Before R03, record a supervised sample fraction strictly between 0 and 1, source dataset ID, optimizer/learning rate, loss weights and replay/update schedule. Keep supervised best-move classes/soft teacher WDL distinct from self-play visit distributions/exact final WDL, and log sample counts/losses by source. Held-out samples never enter replay. Use conservative updates relative to pretraining; record initial settings before the local cycle and refine the learning rate and update-to-data ratio through the pilot before the full run. Mixing reduces forgetting risk without guaranteeing improvement. Stockfish WDL and our agents' game outcomes reflect different playing conditions, so prediction changes alone do not establish degradation.

Decision revised on 7 October 2026: use `python-chess` with a small AlphaZero-style PUCT implementation and ending policy `actual-occurrence-v1`. In search, self-play and matches, first check `board.outcome(claim_draw=False)`; otherwise adjudicate a draw on an actual third repetition (`board.is_repetition(3)`) or reached 100-halfmove condition (`board.is_fifty_moves()`), checking repetition first if both apply. An available intended move does not terminate the game. This simplified rule still automatically accepts reached claim conditions. No twofold-draw search heuristic is enabled initially. Use a configurable 512-ply pilot limit; check terminal outcomes before the limit. Length-limited games are truncated, interrupted games are resumable, and neither receives a fabricated draw target. Initially train self-play outcome targets only on completed games. Resume interrupted games with their full rule history and the recorded model/search configuration. Version the ending policy separately; tensor encoding and move vocabulary are unchanged.

Define `evaluate(states) -> list[Evaluation | TerminalResult]` before building search, returning results in input order and batching only nonterminal states. Implement batch-of-one first. Profile concurrent games contributing one pending leaf each to a shared batch and choose practical concurrency before the main run. Keep CPU encoding and game execution in the throughput measurements.

Set and record root-noise and move-temperature schedules before self-play. Measure duplicate completed-game rates; replayed randomized book openings are an available adjustment if games repeat, with opening source/seed/prefix recorded and full history retained. Log completion, truncation and interruption counts/rates. High truncation can bias completed-only training toward short games and must be diagnosed before scaling. The 16–32-simulation budget is an integration starting point; choose the main budget from measured throughput and checkpoint results. Gumbel search, KL anchoring, Q-blended value targets and playout-cap randomization remain later options if the pilot demonstrates a need.

Stockfish provides initialization labels. The RL phase learns from our own search and game outcomes. MCTS produces training targets; gradient updates train the network. All trainable components can update during RL, while the saved initial checkpoint P remains an immutable artifact.

## Pilot and compute

The pilot should include:

- The fixed GAB + Q/K/V relative-attention transformer with an initial 10,000 labeled positions and expansion guided by measurements.
- A tiny subset that the model can deliberately overfit.
- 16–32 MCTS simulations per move for initial integration.
- Approximately 100–300 self-play games and subsequent network updates.
- Required P-versus-R validation matches under identical search, weaker fixed-opponent diagnostics, and an interrupted-job/resume check.

Measure Stockfish annotation throughput, replay/encoding and storage costs, supervised training throughput and held-out learning curves, GPU memory, inference throughput at batch one and realistic concurrent-game batches, complete games per hour, average game length, and evaluation runtime. Measure the complete self-play pipeline, including CPU work, and log duplicate-game and truncation rates. Reserve explicit compute for mandatory P-versus-R matches.

Before scaling, verify special moves, legal masks, terminal outcomes, value backup signs, draw handling, and checkpoint resumption. The pilot does not have to reach a particular Elo or establish RL improvement.

Provisional allocations are 12–24 rented GPU-hours for the pilot and approximately 200–300 university GPU-hours for the main project, subject to pilot measurements. The proposed university resources are one GPU with at least 24 GB memory, 16–32 CPU cores, 64 GB RAM, and approximately 100 GB working storage.

Choose the full self-play and evaluation search budgets after measuring throughput. Use the batched evaluator contract throughout and set concurrent-game batching from measured end-to-end performance.

## Three-week schedule

These are relative project days, not a confirmed calendar deadline.

| Period | Required outcome |
|---|---|
| Days 1–3 | Shared interfaces and encoding, environment, model, basic MCTS, tiny overfit check |
| Days 4–7 | Complete GPU pilot, combined-model profiling, initial pretraining, saved checkpoint P and pilot P-versus-R assessment |
| Days 8–14 | Self-play training with supervised replay mixture, periodic P-versus-R validation matches, saved RL checkpoints |
| Days 15–17 | Final fixed-opponent and P-versus-R evaluation using checkpoints selected through validation |
| Days 18–21 | Analysis, reproducibility, report, and playable demonstration |

Reserve the final days for evaluation and writing rather than continuing training until the deadline. These windows are targets; the combined relative-attention port and correctness gates remain required. If measurements make the schedule unrealistic, reduce optional scale and later search experiments while reserving time for the required RL assessment and deliverables.

## Ownership

| Owner | Responsibilities |
|---|---|
| Person 1 | Dataset, network, supervised pretraining, training updates, checkpoints |
| Person 2 | Chess environment, move encoding, MCTS, self-play generation, match runner |
| Together | Interface specification, correctness checks, experiment settings, integration, analysis |

Integrate daily. Develop search against a dummy evaluator returning uniform legal-move probabilities and zero value before the real model is ready.

## Evaluation and deliverables

Evaluate the final model with MCTS against fixed, documented opponents and perform mandatory P-versus-R matches in pilot/main validation and final evaluation. Include weaker diagnostic rungs: random legal moves, one-ply material-greedy, and a pinned Stockfish binary with small node budgets, as well as any stronger configured opponent. Record seeds, greedy scoring/tie handling, engine settings and search budgets; do not infer an Elo from a node limit. Use paired openings with colors reversed, separate validation and test openings, and no exploration noise. Compare P and R under identical search and ending policies. If throughput permits, aim for 200–400 final games; allocate games explicitly across comparisons and report the actual counts and uncertainty for each.

Report wins, draws, losses, score `(wins + 0.5 * draws) / completed_games`, and uncertainty that respects opening pairs, alongside failures/truncations and their denominators. Record self-play games and positions, duplicate/truncation rates, dataset phase composition, search budgets, supervised replay fraction, training compute, inference speed, and evaluation latency. Freeze a chosen R comparison checkpoint and the playable checkpoint using validation, then evaluate on reserved test openings. The playable checkpoint may be P if validation favours it; report the chosen R-versus-P test result regardless. A single run measures checkpoint change under the documented protocol and does not establish a general causal advantage.

The deliverables are reproducible code/configurations, saved training checkpoints and a selected final model, training curves, match results with uncertainty, compute measurements, a report, and a playable demonstration.

A specific Elo or positive RL gain is not the success criterion. Completion means a functioning model-and-MCTS chess agent, a verified training/search pipeline, mandatory P-versus-R results, and an honest assessment of its capabilities and practical limits. Stronger research claims and comparisons without MCTS remain optional.

## What to work on next

### 1. Use the completed interface specification

The accepted contract is recorded in [interface.md](interface.md). Both people should implement against it before building substantial components. It specifies:

- Implement the locked 128-channel feature groups, eight-snapshot history, Black rank reflection, and square ordering; specify channel ordering, repetition thresholds, missing-history padding, and normalization.
- Document the locked 4,352-slot move vocabulary and reversible move-to-ID formulas above, including promotion logits and special moves.
- Document the 4,352-slot legal mask and specify behavior when no legal moves exist.
- Policy/WDL output shapes, the batched evaluator API and the side-to-move convention.
- Training-record format for teacher labels and self-play targets.
- Actual-occurrence draw adjudication, terminal precedence, and how unfinished or truncated games are handled.
- Configuration and checkpoint metadata needed to load the correct encoding and model.

Specification revised on 7 October 2026. Implementation completion means both people can independently construct and interpret the same encoded position, move ID, legal mask, and target without guessing conventions.

The 5 October clarification adds callable adapter signatures, incomplete-prefix provenance, terminal return objects and explicit input validation, plus initial backbone/WDL defaults. Feature encoding remains `chess128-v2` and move vocabulary remains `pair4352-v1`; the clarifications do not reorder or change tensor channels or move IDs.

The 7 October revision fixes combined attention, specifies batched evaluation, ending policy `actual-occurrence-v1`, supervised replay mixing and mandatory checkpoint matches. Earlier draw-policy records cannot be silently reused or mixed with new outcomes.

### 2. Implement and check the shared chess adapter

Use `python-chess`. Implement position encoding, legal-move mapping, terminal detection, and outcome conversion in one shared module.

Check move-ID round trips and unique IDs among legal moves. Include White/Black positions, both castling sides, en passant, queen and underpromotions, checkmate, stalemate, repetition, and the halfmove rule. Keep rule state in the environment rather than trying to reconstruct all history from a FEN alone.

Completion means every consumer uses the same adapter and special-move/perspective checks pass.

### 3. Build both halves against that interface

Person 1 implements the encoder-only policy/WDL network, dataset loader, and trainer. Check output dimensions, count parameters, and overfit a tiny labeled subset.

Person 2 implements basic PUCT search and a match runner using the dummy evaluator. Check terminal handling, root visits, legal actions, and value signs using simple positions with known outcomes. Then replace the dummy evaluator with the real network.

Completion means the model can produce valid predictions and the search can complete legal games using them.

### 4. Complete one tiny learning cycle locally

Generate a few self-play games, store their search distributions and final outcomes, run a training update, save/reload a checkpoint, and run a short match. This is a plumbing check, not a strength result.

Completion means the entire path from position to search to game record to training update to resumed evaluation works.

### 5. Run the rented-GPU pilot and finalize the main training plan

Use the pilot to profile the fixed combined architecture and choose practical batch/search budgets, supervised replay settings and staged dataset expansion. Write down the measured cost of pretraining, self-play, mandatory P-versus-R matches and final evaluation before requesting or spending the larger allocation.

The immediate implementation priority is step 2, using the completed interface in step 1. It establishes the shared adapter on which model training, MCTS, self-play, and credible evaluation all depend.

## Locked decisions and remaining pilot choices

Current decisions as of 7 October 2026: retain detailed input conventions, PGN source family/sampling policy, initial teacher settings/target semantics and the `python-chess` adapter; fix GAB + implemented Q/K/V relative attention; use actual-occurrence draw adjudication and the batched evaluator; mix supervised samples into RL replay; require P-versus-R matches. The shared contract is recorded in `interface.md`; the next implementation deliverable is the chess adapter.

Before generating data, fill the dataset manifest with the selected archive URL/checksum, Stockfish binary/version, and sampling seed. Before porting code, record upstream commit identifiers. These are reproducibility records, not changes to the accepted algorithms. Optimizer, loss weighting, nonzero supervised replay fraction and replay/update schedule need explicit configuration before R03 and measured refinement before the full run.

The combined attention architecture is fixed. Do not freeze GPU batch sizes/concurrency, final dataset size or full-run simulation budgets before the pilot.

## Reference material

- [Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/html/2605.19091), including compact GAB configuration in Appendix A.2.
- [Released Maia-3 move vocabulary](https://github.com/CSSLab/maia3/blob/main/maia3/utils.py) and [policy-head implementation](https://github.com/CSSLab/maia3/blob/main/maia3/models.py), for the locked 4,352-output layout and explicit promotion ordering.
- [Maia-3 history padding](https://github.com/CSSLab/maia3/blob/main/maia3/dataset.py#L40-L49), for repeating the earliest available position when history is short.
- [Earlier Chessformer preprint](https://arxiv.org/pdf/2409.12272), discussed for Shaw-style attention and the small-model backbone.
- [Research attention implementation](https://github.com/daniel-monroe/lczero-training/blob/master/tf/tfprocess.py), the accepted source for porting the Q/K/V Shaw-style variant; omit the paper's position-only cross term as in this implementation.
- [Final AlphaZero paper and supplement (2018)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), for the canonical search and self-play training reference; [2017 preprint](https://arxiv.org/html/1712.01815v1) for earlier context.
- [Leela's search primer](https://lczero.org/dev/lc0/search/alphazero/) and [DeepMind Mctx](https://github.com/google-deepmind/mctx), for practical search guidance and optional later Gumbel search investigation.
- [Representation Matters for Mastering Chess](https://arxiv.org/html/2304.14918v2#sec3), Table 3 and Inputs V2.0 for the 15 engineered additions.
- [Leela input encoder](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc), for inspecting version-specific encoding conventions; our base feature list is pinned to the Chessformer Appendix A.2 description.
- [Representation Matters accompanying feature encoder](https://github.com/QueensGambit/CrazyAra/blob/902bc5efd1a2aae03c1b0dc9301b2a98c7dff4d3/DeepCrazyhouse/src/domain/variants/classical_chess/v3/input_representation.py) and [helpers](https://github.com/QueensGambit/CrazyAra/blob/902bc5efd1a2aae03c1b0dc9301b2a98c7dff4d3/DeepCrazyhouse/src/domain/util.py), for engineered-feature values, material normalization, bishop condition and legal en-passant masking.
- [Lichess Elite PGN archives](https://database.nikonoel.fr/), accepted pretraining game source.
- [python-chess rules](https://python-chess.readthedocs.io/en/latest/core.html) and [UCI interface](https://python-chess.readthedocs.io/en/latest/engine.html), accepted adapter dependency.
