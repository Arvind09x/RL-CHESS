# RL-Chess finishing plan (revised 13 Oct 2026, 14:00 IST)

Deadline: Wednesday 21 October. Compute: Mac (M3, 8 cores, 16 GiB) for training,
self-play and all evaluation; a rented 32-vCPU RunPod CPU pod for Stockfish
labelling; college GPU if it arrives (larger self-play rounds, 1M-label
training; on no critical path).

**Guaranteed deliverable:** P trained on the 100k prefix plus one completed
self-play round R1 with its validation and test matches, rungs, report and demo.
300k/1M training, R2 and the GPU are extensions.

## Status at 13 Oct

- main at cf5bf87: AdamW (grouped decay 0.01), global-norm clip 10, warmup 500 +
  cosine to 10%, recorded in checkpoint provenance and enforced on resume.
  Full suite 1,645 passed on the Mac.
- P02 strength check (step-1410 vs untrained, 64 pairs, 32 sims): 64 wins,
  62 draws, 0 losses, 2 truncated. Score 0.754, paired bootstrap 95%
  [0.714, 0.802]. 53 draws by threefold repetition, 8 by fifty-move rule: the
  candidate cannot convert won endgames. Hypothesis (unmeasured): value head and
  FPU=0; RL is the planned fix.
- Data inputs: the saved bundle `.../transfer-bundle-v1/data/d08` (manifest,
  289,427 canonical games, 2.31M audited references, D08 labels 9k/500/500).
  Nothing references `/private/tmp` again.
- Capacity: 260,379 training games carry samples; at most 4 per game gives
  1,040,950 train references. Validation 115.7k, test 115.4k available.

## Frozen decisions

### Data (D09 expansion)
- Target 1,000,000 train / 10,000 validation / 10,000 test. Teacher unchanged:
  Stockfish 17.1, 100k nodes, 1 thread, 128 MB hash cleared before each search,
  MultiPV 1.
- Membership order: the 10,000 D08 references (9,000/500/500), then the
  remaining validation to 10,000, then test to 10,000, then train in rank
  order. Ranking: sha256 of compact JSON {version "d09-expansion-sha256-v1",
  seed, split, game_id, sample_ply}; at most 4 references per game, D08's
  counted. Freeze records post-cap capacity per split and refuses a quota
  above it. Any train prefix is the deterministic selection for that size;
  `finalize --train-prefix N` cuts a bundle with N train labels plus the
  complete 10k/10k held-out sets, and may run while workers continue.
- Label equivalence criterion (fixed before running): for every reference
  labelled by two runs (the 500-position Mac compare; the 10k D08 overlap in
  production), label records must be equal on all fields except
  `returned_search.time`, `returned_search.nps` and attempt timestamps. Any
  mismatch fails finalize; the summary records field-wise match counts.
- Build acceptance is separate from equivalence: a pod build is accepted when
  its attestation matches the binary and the manifest's release tag, source
  commit and embedded NNUE hashes; that establishes provenance only.
- Engine policy: `persistent-clear-hash-v1` (ucinewgame + Clear Hash before
  each search) if the 500-position compare passes; fresh otherwise.
- Deferred: MultiPV soft targets, phase quotas, any search change.

### Supervised training (P)
- AdamW lr 1e-4, decay 0.01 on matrix weights (experimental; Maia-3 uses 1e-6),
  clip 10, batch 32 (64 max on GPU, lr unchanged), loss weights 1/1.
- 100k: 5 epochs = 15,625 steps, warmup 500, horizon 15,625, validation every
  500. 300k: 3 epochs = 28,125 steps, same warmup, horizon 28,125. 1M (GPU):
  2 epochs.
- P = lowest validation total loss on the 10k validation set; policy-only loss
  also reported.

### Self-play and mixed training (R)
- 200 games per round, 32 sims, 4 workers, temperature 1 to ply 30 then 0,
  Dirichlet 0.3/0.25, 512-ply cap, truncated games excluded, value target =
  exact final WDL. Round 1 from P; round k from R(k-1).
- Mixed updates for round k use the accumulated replay of all admitted rounds:
  50/50 batches of 32 (16+16), AdamW lr 1e-5 decay 0.01 clip 10 via the shared
  builder, no warmup; ceil(admitted positions / 16) updates (one pass; an equal
  number of supervised draws); checkpoint every 250 for resumption/curves only.
  New driver; run_r03/run_r04 untouched.
- Selection: candidates are round-final checkpoints. R = highest P-vs-R
  validation score. Demo checkpoint = R if that score > 0.5 else P (a selection
  rule, not an improvement claim). Improvement is claimed only if the pair
  bootstrap interval excludes 0.5. Test openings used once.

### Evaluation
- Validation P-vs-R 64 pairs (~5.6 h on the Mac); rungs vs random, greedy,
  Stockfish 1k and 10k nodes, 16 pairs each; final test P-vs-R 64 pairs. Same
  search settings for every neural player, all on the Mac. No adjudication;
  report W/D/L, incomplete games, truncations and the pair bootstrap interval;
  no Elo.

Runtime: float32, strict determinism, highest matmul precision.

## Code to write (plan-first, one branch each)

1. `scripts/attest_stockfish.py`; alternate-build acceptance in
   `stockfish_worker.verify_teacher_bytes` and `supervised._provenance`; tests.
2. `scripts/expand_labels.py`: freeze / run (K workers, per-worker durable
   files, resumable) / finalize (merge, summary, open_dataset acceptance,
   overlap diff, --train-prefix) / compare.
3. `configs/data-d08.json`, `configs/data-d09-<N>.json`, `docs/runpod-labelling.md`.
4. `scripts/run_selfplay_campaign.py`, `scripts/train_mixed.py`,
   `scripts/run_match.py`, `scripts/play.py`.

## Schedule

- Tue 13: items 1-3; Mac compare; pod labelling started (held-out first, 100k
  prefix usable within the hour). Fallback: Mac labels 100k overnight.
- Wed 14: finalize prefixes; train P on 100k (~25 min/epoch plus validation);
  select P; rungs. Item 4 during training. Overnight: self-play round 1.
- Thu 15: admit; train_mixed -> R1. Overnight: P-vs-R1 validation match.
- Fri 16: round 2 if time; R decision. Start report.
- Sat 17: test-opening evaluation; demo check.
- Sun 18 - Mon 19: report, README, backup. Tue 20: slack. Wed 21: due.

## Known risks

- Repetition draws inflate draw rates and cut match sensitivity (64 pairs:
  ~89% power for +0.20, ~36% for +0.10).
- Pod vCPUs are hyperthreads: one worker per physical core; measure.
- Mac thermal/background load: size overnight runs at 20 games/h.
- Memory: do not run labelling and self-play together at 8 workers each.
- Data regime: CF-6M (same architecture) trained on ~3B positions; report
  100k-1M as small-data. Expected top-1: mid-30s at 100k, low 40s near 1M.
