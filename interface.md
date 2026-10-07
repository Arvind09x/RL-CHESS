# Shared chess interface

Accepted: 5 October 2026; revised: 7 October 2026. Encoding version: `chess128-v2`. Move vocabulary version: `pair4352-v1`. Ending policy: `actual-occurrence-v1`. WDL order: `[win, draw, loss]`. Version v2 replaces v1's zero-filled missing history with earliest-position padding. The latest revision fixes GAB + Q/K/V relative attention, adds the batched evaluator and supervised replay mixture, and replaces intended-move draw adjudication. Tensor channels and move IDs are unchanged.

This contract applies to supervised pretraining, self-play, MCTS and evaluation. It preserves the decisions in [plan1.md](plan1.md). The representation is our hybrid; matching individual references does not imply checkpoint compatibility with their complete models.

The companion [coding-plan.md](coding-plan.md) supplies the implementation order and task acceptance checks; it does not replace this contract.

## Rule state and coordinates

Use `python-chess` for standard chess, legal moves, PGN replay, checks, draw adjudication and Stockfish UCI communication. The environment retains the initial board and complete actual-coordinate move stack. Do not reconstruct repetition or history from a current FEN alone. Chess960 is outside this contract.

Squares use `a1=0, b1=1, ..., h8=63`, with `square=8*rank_index+file_index`. For White to move, actual and network coordinates coincide. For Black to move, use `network_square=actual_square XOR 56`: reflect ranks and preserve files. This transform is its own inverse.

“Our” means the current position's side to move, including when encoding earlier snapshots. Use the current position's ownership and transform for every snapshot. Square masks and teacher/search moves use that transform. Environment operations remain in actual coordinates.

## Input tensor

Return finite `float32` features of shape `[64,128]`; batches have shape `[B,64,128]`. Constant scalar features are broadcast to all 64 squares. The network projects these to its embedding width, initially 256.

| Zero-based channels | Meaning |
|---|---|
| `13*t + 0..5`, `t=0..7` | Our pawn, knight, bishop, rook, queen, king placement at snapshot t |
| `13*t + 6..11` | Opponent placement in the same piece order |
| `13*t + 12` | Snapshot repetition flag |
| `104..107` | Our queenside, our kingside, opponent queenside, opponent kingside castling rights |
| `108` | Actual Black-to-move indicator |
| `109` | Current halfmove clock divided by 100 |
| `110` | Constant 0 |
| `111` | Constant 1 |
| `112..113` | Current our/opponent occupancy masks, including kings |
| `114` | Network-coordinate checkerboard |
| `115..119` | Current our-minus-opponent counts for P,N,B,R,Q, each divided by 8 |
| `120` | Opposite-colored-bishops indicator |
| `121` | Locations of pieces checking the current player's king |
| `122..126` | Current our counts for P,N,B,R,Q, each divided by 8 |
| `127` | Current legal en-passant target mask |

Snapshot t=0 is the current board; t=1 is one ply earlier when available. Retain up to seven preceding positions, including positions before captures or pawn moves when available. Fill each absent older snapshot by copying the earliest available position's encoded 13-channel block, using the current position's ownership and coordinate transform. Copy its repetition flag as computed from its real historical prefix; padding never increases that flag or contributes occurrences to rule history. Do not synthesize prior moves. Record `history_complete` and `available_snapshots` outside the tensor; the latter counts real snapshots only, not padding. For an isolated FEN, repeat its current snapshot into all seven missing slots; earlier repetitions are unknown and all eight repetition flags are 0.

With newest-first ordering, game start encodes `[S0,S0,S0,S0,S0,S0,S0,S0]`. After two plies it encodes `[S2,S1,S0,S0,S0,S0,S0,S0]`, where S0 is the initial position, S1 the position after one ply, and S2 the current position. Use this identical padding convention in pretraining, self-play and evaluation. The reference Maia-3 helper repeats the earliest available piece-placement snapshot; copying its accompanying repetition flag is the explicit extension for our 13-channel history blocks.

A real snapshot's repetition flag is 1 if the same rule position has appeared at least once earlier in that snapshot's game prefix (second occurrence or later). Equivalence uses pieces, side to move, castling rights and en-passant availability affecting legal moves; it excludes halfmove/fullmove counters. `board.is_repetition(2)` on the board at that prefix is the reference behavior. Never use moves later than the snapshot to calculate its flag. The eight snapshots do not bound the environment's repetition record.

Binary masks/flags use 0 and 1. Castling channels describe rights, not whether castling is currently a legal move. The halfmove clock increases on castling and resets only on pawn moves or captures. Keep `/100` without clipping, following the locked Chessformer base description rather than other Leela encoding versions.

The checkerboard is `(network_rank_index + file_index) % 2`: a1=0, b1=1. It is generated in network coordinates, not an actual-color mask reflected from the board. All material types use the divisor 8; preserve signed differences and values above 1 from promotions. The bishop flag is 1 exactly when each side has one bishop and their square colors differ. It does not require the absence of queens, rooks, knights or pawns. En-passant channel 127 is one-hot at the transformed `board.ep_square` only when `board.has_legal_en_passant()` is true; otherwise it is all zero.

## Move vocabulary and legality

Return policy logits `[B,4352]` in this fixed order:

- Ordinary move IDs: `64*from_square + to_square`, range `0..4095`, with no promotion piece.
- Promotion IDs: `4096 + 4*(8*from_file + to_file) + piece_index`, where origins are network rank 7, destinations rank 8, and piece indices are queen=0, rook=1, bishop=2, knight=3.

Examples: e2e4=796; e7e8q=4240; e7e8r=4241; e7e8b=4242; e7e8n=4243. All 256 promotion slots remain present; impossible geometry is always illegal. An ordinary origin-destination slot cannot represent a pawn promotion.

Castling uses king origin/destination; en passant uses pawn origin/landing square. Enumerate actual legal moves, transform squares, and map to IDs to form a boolean `[4352]` legal mask. ID decoding must preserve the promotion piece and inverse-transform squares before environment application. Check legality before pushing a decoded move.

The policy head uses `dot(origin_query,destination_key)/sqrt(d_policy)`. For promotion logits, add the destination-derived piece adjustment from the released Maia-3 implementation, including its `sqrt(d_policy)` multiplier on `promo_bias_proj(destination_key)`. Flatten ordinary scores with destination varying fastest, then append the promotion scores in vocabulary order.

Mask illegal logits before one softmax over legal moves and before policy cross-entropy normalization. Illegal probabilities are zero. Calculate loss over legal indices to avoid `0*(-infinity)` numerical errors. Handle all terminal outcomes before normalization, including adjudicated draws that still have legal moves. A nonterminal state must have at least one legal move.

## WDL and evaluator contract

The model returns policy logits `[B,4352]` and WDL logits `[B,3]`. Apply WDL softmax in `[win,draw,loss]` order, from each input position's side-to-move perspective. The search evaluator returns legal normalized priors and `v=P(win)-P(loss)` in `[-1,1]`.

Expose `evaluate(states: Sequence[ChessState]) -> list[Evaluation | TerminalResult]`. Return one result per input in the same order; an empty input returns an empty list. Adjudicate each state first, batch only nonterminal states for model inference, and restore the input order. Terminal states skip encoding/model inference and policy normalization. A batch of one is the initial execution path. Search uses this API from the start; concurrent games can later contribute one pending leaf each to a shared inference batch without introducing concurrent traversal of a single tree. Evaluation must leave all input states and their histories unchanged.

Adjudicate terminal positions before calling the model. Checkmate is loss for the checkmated side to move; draws have v=0. Exact terminal WDL labels are respectively `[0,0,1]` and `[0,1,0]`. For completed-game records, a position gets `[1,0,0]` if its player eventually won, `[0,0,1]` if that player lost, and `[0,1,0]` for a draw. Flip value sign at each ply during backup; parent edge values are from the parent's player perspective.

Search must preserve rule history along every path. Avoid merging nodes or caching evaluations solely by piece placement/current FEN: history inputs, repetition and counters may differ.

## Adapter API and history provenance

Clarified on 5 October 2026. The following Python API and error/provenance conventions are project engineering decisions. They implement the accepted representation and rule policy; they are not additional claims from the source papers.

`ChessState` owns an actual-coordinate `chess.Board` with its retained move stack, its initial FEN, and explicit `history_complete` provenance. Feature extraction must not replace that board with a mirrored or FEN-reconstructed board. All inspection functions leave the state unchanged, including its move stack.

- `make_state(start_fen=None, moves_uci=(), *, history_complete=None) -> ChessState`: construct the standard initial board when `start_fen` is omitted, or parse the supplied FEN, then replay the supplied actual-coordinate legal moves. The default completeness is true for an omitted starting FEN and false for an explicit FEN. An explicit true value requires provenance that this is the actual game initial position and every subsequent move is retained, including games genuinely beginning from a declared setup. A FEN's counters or resemblance to the standard starting position are not evidence of provenance. Replay validates chess legality; source PGNs may continue through positions where our execution policy would adjudicate a draw, so replay does not itself enforce that policy at every historical prefix. Consumers adjudicate each sampled/search root using the accepted policy.
- `encode_position(state) -> EncodedPosition`: return `features` as a CPU NumPy `float32` array `[64,128]`, `history_complete` as a boolean, and `available_snapshots` as an integer in `1..8`. Tensor conversion and batching belong to consumers.
- `move_to_id(state, move: chess.Move) -> int`: require a legal actual-coordinate move and encode it using the current state's perspective.
- `id_to_move(state, move_id: int) -> chess.Move`: require an integer ID in `0..4351`, decode exactly, and require the resulting actual-coordinate move to be legal. Never infer queen promotion for an ordinary slot. Encoding/decoding inspect chess legality; environment termination is checked separately.
- `legal_mask(state) -> ndarray[bool,4352]`: enumerate chess-legal moves, even if our ending policy has ended the game. The mask describes chess legality, not permission to continue the episode.
- `terminal_result(state) -> TerminalResult | None`: first use `board.outcome(claim_draw=False)`; if it returns none, adjudicate a draw when `board.is_repetition(3)` or `board.is_fifty_moves()` is true, checking repetition first if both apply. Return none for a nonterminal position; otherwise return `winner` (White, Black or none for a draw), `ending_reason` (the corresponding `chess.Termination.name`, including `THREEFOLD_REPETITION` or `FIFTY_MOVES` for our added adjudications), exact side-to-move `wdl` `[3]`, scalar `value`, and `priors=None`. An available intended move alone never ends a game. See the ending policy below for precedence and versioning.
- `apply_move(state, move_id) -> None`: reject a terminal state first, decode and validate the ID, then push the move in actual coordinates. This is the explicit mutating operation. MCTS works on copies retaining the full stack, or uses controlled push/pop with exact restoration.

Each nonterminal evaluator result is a distinct `Evaluation` with finite `float32` `priors` `[4352]`, finite WDL probabilities `[3]`, and scalar value. Terminal handling uses `TerminalResult` and never invokes the model or normalizes a policy. An adjudicated draw may therefore have a nonempty legal mask and still prohibit `apply_move`. The episode runner handles interruption and the ply cap separately; those statuses do not become rule outcomes.

`available_snapshots = min(8, len(retained_move_stack) + 1)` counts positions actually retained since the recorded initial board. It is independent of `history_complete`: two plies into a fully replayed game give three snapshots and complete history; a FEN followed by twenty known moves gives eight snapshots and incomplete history. Completeness stays false when adding moves to an incomplete prefix.

For a FEN followed by known moves, compute each real snapshot's repetition flag using only its available historical prefix, with the recorded initial FEN counted once. A flag of 1 establishes an observed repeat; a flag of 0 with incomplete history means no repeat was established from available history, not proof that none occurred earlier. Pad the earliest real block as already specified. Do not infer moves, earlier repetitions or draw claims from missing history. Rule adjudication on incomplete states uses the retained prefix and must retain the incomplete-history metadata. Initial supervised game records, ordinary self-play and resumed self-play require complete history; isolated-FEN tactical probes may use incomplete states and record that limitation.

Reject invalid boards using `board.is_valid()` and legal replay checks. Basic board validity does not prove reachability from the standard initial position. Reject Chess960/other variants, null moves, malformed FEN/UCI input, noninteger or out-of-range IDs, illegal actions, and attempted actions after termination with `ValueError` and an explanatory message. Failed operations leave the state unchanged. Pin the installed `python-chess` dependency version in the implementation environment.

## Initial backbone and heads

Selected on 5 October 2026 for our small PyTorch model and limited pretraining/self-play budget. These are implementation defaults, not a demonstrated optimum. Retain the current candidate of 8 layers, width 256, 8 heads and feedforward width 256; measure the complete parameter count before freezing the run configuration.

Use the newer Chessformer/Maia-3 backbone and heads as the primary reference. The earlier transformer supplies the already accepted Q/K/V relative operations, rather than a second competing backbone. Start with:

- Linear projection from 128 input channels to width 256, followed by a learned additive square embedding `[64,256]`, initialized to zero as in Maia-3's `AbsolutePE`. The earlier paper's multiplicative square factors are not part of this initial backbone.
- Post-LayerNorm encoder blocks: attention, residual addition, LayerNorm; then `Linear(256,256)`, GELU, `Linear(256,256)`, residual addition, LayerNorm. Use ordinary LayerNorm and retain projection biases initially. Follow Maia-3's residual/attention/feedforward dropout placement with initial training dropout 0.1, followed by its final encoder LayerNorm. Dropout is disabled for search/evaluation and may be set to zero for the tiny overfit check; record any training-default change.
- The locked compact GAB with d1=8, d2=32, d3=32 and one shared template bank, combined with all three accepted Q/K/V relative terms. Use this combined architecture throughout supervised and RL training. The pilot measures its quality, inference cost and memory; it does not select between attention variants.
- Bias-free policy origin/destination projections with initial `d_policy=128` and the existing promotion formula/vocabulary. Policy width is a model configuration field; logits remain `[B,4352]`.
- WDL head: mean over the 64 final square embeddings, LayerNorm(256), Linear(256,128), ReLU, Linear(128,3). This follows the newer Chessformer output-head description and Maia-3 value path. WDL pooling does not replace GAB's separately specified board compression. Do not introduce the earlier model's flattened value head or auxiliary value heads into this initial implementation.

Remove Maia-3's Elo/time conditioning and ponder head, since our target is playing strength using the agreed features and teacher/self-play targets. Use the released operations for GAB and promotion scoring; pin upstream commits and record the resolved module configuration/initialization before porting. Numerical validation is still required for the new relative-attention contractions.

Use Leela as the practical chess-search/self-play reference, with the final 2018 AlphaZero paper and supplement as the foundational search/training-loop reference. Retain our planned PUCT search and completed-game policy/WDL training. Search constants, exploration/temperature schedules and replay/optimizer settings must be recorded before their corresponding learning runs; large-run settings from either system are not assumed optimal at our 16–32-simulation pilot budget. Gumbel AlphaZero/MuZero search is a possible later experiment if low-budget search limits learning, not a change to the current PUCT milestone or permission to import a learned dynamics model.

Our Stockfish initialization, 128-channel hybrid, actual-occurrence draw adjudication and exclusion of unfinished games from outcome training remain deliberate project choices. AlphaZero's training from scratch, original encoding and length-limit labeling do not override them. Evaluate practical strength and throughput before describing this combination as better than a reference system.

## Accepted attention implementation

Port the earlier Chessformer authors' `make_rpe_map`, `RPELogits`, `RPEValue` and attention contractions from TensorFlow to PyTorch. Retain Maia-3's compact GAB and policy head, replacing its human-specific inputs with this contract. Pin upstream commits before porting and retain attribution. Maia-3's static relative bias is not a substitute for the Q/K/V operations.

Define displacement `delta=(rank_i-rank_j,file_i-file_j)` and index `15*(rank_i-rank_j+7)+(file_i-file_j+7)`, range `0..224`. Each layer learns three independent relative-vector tables, logically `[heads,225,head_dim]`, initialized to zero as in the inspected reference. Preserve index and parameter-layout conventions when checking the port.

For one head, let U_delta be the reference's table contracted with q, T_delta the table contracted with k, and V_delta the relative-value table. These names avoid confusing the code's `rpe_q`/`rpe_k` labels with the paper's aQ/aK labels. The accepted operations are:

`logit(i,j) = (dot(q_i,k_j) + dot(q_i,U_delta) + dot(k_j,T_delta))/sqrt(head_dim) + GAB(i,j)`

`alpha(i,:) = softmax(logit(i,:))`

`output_i = sum_j alpha(i,j)*(v_j + V_delta)`

All three relative-vector terms are enabled in the chosen architecture. Keep their operations separately inspectable for numerical checks; a diagnostic disable path may recover GAB-only behavior without creating a second training track. GAB remains after logit scaling, as in the reference generated-bias path. Use compact GAB dimensions d1=8, d2=32, d3=32, with the template bank shared across encoder layers.

Do not add `dot(aQ_delta,aK_delta)`: it appears when expanding the earlier paper's Equation 6 but is absent from the inspected implementation. Describe this as the authors' implemented Q/K/V Shaw-style variant, not an exact reproduction of that equation. No author rationale for the omission is established. Validate numerical outputs/gradients against reference operations before training the combined model. Profile the combined model during the pilot; superiority over GAB-only is untested.

## Supervised data and teacher targets

Use one fixed monthly Lichess Elite standard-chess PGN archive. Before annotation, record its exact archive URL/checksum and the sampling seed in a dataset manifest. Reject invalid or unreplayable games. Split games 90/5/5 into train/validation/test before sampling. Remove duplicate-game and overlapping-position samples across splits; retain validation/test separation. Audit overlap using rule position identity, independently of network-history differences.

Sample at most eight distinct nonterminal positions per game, spread across game progress, with opening, middlegame and endgame coverage across the dataset. Record the exact sampling algorithm/phase boundaries in the manifest and report phase counts per split before and after overlap removal. Start with 10,000 total labeled positions for plumbing and the initial learning pilot; benchmark 1,000 positions first. Consider staged expansion to 100,000, several hundred thousand and potentially 1 million labels using measured annotation/replay/encoding/storage costs, training throughput and held-out learning curves. There is no 50,000-label ceiling or mandatory million-position allocation. Expansion retains game assignments and protected held-out sets, and repeats overlap audits before annotation.

Use a fixed Stockfish binary with Threads=1, Hash=128 MB, MultiPV=1 and UCI_ShowWDL=true. Initial limit: 100,000 nodes per position. Pass the replayed board with full move history to the engine. Record binary/version/checksum, all settings, requested nodes and actual returned search information. Reset the hash before each independent annotation so worker ordering does not silently change its warm state. An interrupted or failed annotation is retried, not treated as a finished target.

For a successful analysis, store the root PV's first legal move as an actual UCI move and its transformed best-move ID. Obtain root WDL in the root player's perspective; store `[wins,draws,losses]/1000` as soft probabilities, checking nonnegativity and unit sum. Store raw returned counts and score/mate information for provenance. Missing WDL is an annotation failure requiring retry or explicit exclusion, not an implicit centipawn conversion. Keep PGN game result separate; it does not replace teacher WDL.

Use these named record fields, independently of the eventual storage container:

- `game_id`, `split`, `start_fen`, actual-coordinate `moves_uci`, `sample_ply` (number of moves applied before the sample) and `root_fen`.
- `history_complete`, `available_snapshots`, `encoding_version`, `move_vocab_version` and `wdl_order`.
- `best_move_uci`, `best_move_id`, `teacher_wdl`, `teacher_wdl_counts`, returned search information and `teacher_config_id` linking to the manifest.
- Optional `pgn_result` as separate source metadata.

Materialize tensors/legal masks using the shared adapter or cache them under the same encoding version. Game replay records remain authoritative so the cache can be checked/rebuilt. Policy training uses the best-move class within the legal mask; WDL training uses the soft teacher distribution. Do not store a dense one-hot policy when a class ID suffices.

## Self-play records and game endings

Use ending policy `actual-occurrence-v1` consistently in MCTS, self-play and matches. First return `board.outcome(claim_draw=False)` when present, preserving library terminal precedence, including checkmate and automatic fivefold/75-move endings. Otherwise adjudicate a draw on `board.is_repetition(3)` or `board.is_fifty_moves()` (a reached halfmove clock of at least 100 with legal moves remaining). Check repetition first if both apply. Do not use intended-move claim checks for termination. This is a simplified environment rule that automatically accepts an already reached claim condition; players do not choose whether to claim. No twofold-draw search heuristic is enabled initially. Any later heuristic is recorded separately and cannot change episode adjudication or exact outcome labels.

At every move boundary, check true terminal outcomes before applying the configurable 512-ply pilot cap. Length-limited games have `status=truncated`, `ending_reason=ply_limit`, and no outcome/WDL target. Interrupted games have `status=interrupted` and no outcome/WDL target; save full history and resume with the recorded model/search settings. Completed games have `status=completed`, an outcome and an exact ending reason.

Only completed games enter the initial self-play training buffer. Save per-position state/replay location, legal mask, raw root visits and normalized root visit policy `[4352]`. Require at least one root visit for a training policy; attach exact final WDL only after completion. Do not substitute a network evaluation or fabricate a draw for unfinished games.

Game records include initial board, actual moves, position records, status/reason, winner or draw when completed, model checkpoint ID, search settings and random-generator state needed for resumption. Search can be rebuilt on resume; environment history and saved targets cannot be discarded.

Record the self-play opening source, selection seed and replayed opening prefix, along with root-noise and temperature schedules. Start with a deliberate temperature schedule and measure diversity; replayed randomized book openings are an available adjustment when games repeat. Opening prefixes retain full rule history. Count duplicate completed games using initial board plus the full actual move sequence, and report completed/truncated/interrupted counts and rates separately. High truncation or duplication rates require diagnosis before scaling; completed-only outcome data may otherwise overrepresent short games.

## RL updates and evaluation

Before R03, record a supervised sample fraction strictly between 0 and 1, source dataset ID, optimizer/learning rate, loss weights and replay/update schedule. Mix training-split Stockfish samples with completed self-play samples, drawing their policy and WDL losses using their respective target types. Supervised policies remain class labels and soft teacher WDL; self-play policies remain visit distributions and exact final WDL. Keep target provenance, actual sample counts and per-source losses explicit. Use conservative updates relative to pretraining. Record initial learning-rate/update-to-data settings before the local cycle and refine them using the pilot before the full run. Held-out data never enters replay. The replay mixture reduces forgetting risk but does not guarantee improvement.

Retain immutable P and saved R checkpoints. P-versus-R matches are required during pilot/main validation and final evaluation, using paired openings, identical search settings and no exploration noise. Freeze the chosen R comparison checkpoint and playable checkpoint using validation before reserved test matches; the playable checkpoint may be P if validation favours it, but report R-versus-P results regardless. Fixed opponents also include weaker diagnostic rungs (random legal moves, one-ply material-greedy, and pinned Stockfish with small node budgets). Report settings and uncertainty; an improvement claim is not a completion requirement. Gumbel search, KL anchoring, Q-blended value targets and playout-cap randomization remain later options if pilot evidence justifies them. Teacher WDL and self-play outcomes reflect different playing conditions, so changes in WDL predictions alone do not establish degradation.

## Versioning and verification

Dataset/checkpoint metadata carries encoding and vocabulary versions, WDL order, model configuration, enabled attention mechanisms, upstream commits, teacher configuration, source/sampling manifest and search/end-policy settings. Game records also carry the ending-policy ID; checkpoints retain supervised replay source/fraction and opening/exploration settings needed for resumption. Checkpoints also carry optimizer/training state and random state for training resumption. Reject incompatible versions rather than silently reinterpreting data. The new ending policy does not change `chess128-v2` or `pair4352-v1`; records and runs using the former draw policy require explicit identification and cannot be silently mixed with new outcomes.

Before integration, verify channel counts/scales, missing-history handling, repetition prefixes, Black reflection, castling, en passant, queen/underpromotions, unique legal IDs and round trips, policy normalization, checkmate/stalemate, actual-occurrence draws, nontermination for intended-move-only claims, value signs, terminal-before-cap precedence and interrupted-game resumption. Check evaluator order, empty/batch-of-one/mixed-terminal batches and input preservation. Validate attention contractions with nonzero relative weights and verify disabling them recovers the GAB-only operations. Check replay source separation and required P-versus-R evaluation. Full-run optimizer, loss weights, replay fraction/schedule, batch sizes and search budgets remain explicit configuration/pilot choices.

## Reference sources

- [Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/html/2605.19091): locked base feature groups and compact GAB.
- [Maia-3 model](https://github.com/CSSLab/maia3/blob/main/maia3/models.py), [vocabulary](https://github.com/CSSLab/maia3/blob/main/maia3/utils.py) and [history padding](https://github.com/CSSLab/maia3/blob/main/maia3/dataset.py#L40-L49): policy/promotion operations, ordering and earliest-position padding.
- [Mastering Chess with a Transformer Model](https://arxiv.org/html/2409.12272) and [author implementation](https://github.com/daniel-monroe/lczero-training/blob/master/tf/tfprocess.py): Q/K/V relative attention and documented equation/code distinction.
- [Leela encoder](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc): history blocks, repetition threshold and classical castling order. The earliest-position padding and `/100` conventions are explicitly specified above.
- [Representation Matters](https://arxiv.org/html/2304.14918v2), its [feature encoder](https://github.com/QueensGambit/CrazyAra/blob/902bc5efd1a2aae03c1b0dc9301b2a98c7dff4d3/DeepCrazyhouse/src/domain/variants/classical_chess/v3/input_representation.py) and [helpers](https://github.com/QueensGambit/CrazyAra/blob/902bc5efd1a2aae03c1b0dc9301b2a98c7dff4d3/DeepCrazyhouse/src/domain/util.py): engineered features.
- [Lichess Elite archives](https://database.nikonoel.fr/), [python-chess rules](https://python-chess.readthedocs.io/en/latest/core.html), [engine API](https://python-chess.readthedocs.io/en/latest/engine.html) and [Stockfish WDL](https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html): data and teacher/rules interfaces.
- [Final AlphaZero paper and supplement (2018)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), linked by [DeepMind's publication announcement](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/): canonical AlphaZero reference for search and the self-play update loop, superseding reliance on the 2017 preprint for implementation details.
- [Leela's AlphaZero search primer](https://lczero.org/dev/lc0/search/alphazero/) and [DeepMind Mctx](https://github.com/google-deepmind/mctx): practical search concepts and later Gumbel search alternatives. Mctx is a JAX library and is not the implementation dependency of our PyTorch/python-chess PUCT milestone.
