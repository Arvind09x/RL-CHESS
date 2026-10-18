"""Where does a self-play simulation's time go? A read-only benchmark; plays and stores nothing.

Measures, for one checkpoint on one device:
  - CPU work per position: encode_position and legal_mask;
  - the network alone on pre-encoded inputs at batch 1, 8, 32 and 64 (ms per call and per position);
  - RealEvaluator calls (encode + network + priors) at the same batch sizes;
  - one 128-simulation search per test position, split into evaluator time and everything else
    (tree selection, backup, state copies);
  - the same searches in N spawned processes at once, to show contention between workers.
The answer decides whether batching leaf evaluations across games would pay off.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import platform
import random
import statistics
import time

import chess


def positions(count, seed):
    """Positions from seeded random play, 8 to 60 plies deep, nonterminal."""
    from chess_rl.adapter import make_state, terminal_result
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board, depth = chess.Board(), rng.randint(8, 60)
        for _ in range(depth):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        state = make_state(moves_uci=[m.uci() for m in board.move_stack], history_complete=True)
        if terminal_result(state) is None:
            out.append(state)
    return out


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return dict(median_ms=1000 * statistics.median(times), mean_ms=1000 * statistics.fmean(times), n=repeats)


def searches(checkpoint, device, simulations, count, seed, configure_runtime):
    """Search timing in this process: total ms per simulation and the evaluator's share."""
    from chess_rl.evaluation.players import SearchPlayer, SearchSettings
    from chess_rl.search.real_evaluator import RealEvaluator
    if configure_runtime:
        from chess_rl.runtime import configure
        configure(device, threads=1)
    evaluator = RealEvaluator(checkpoint) if device == "cpu" else RealEvaluator(checkpoint, device=device)
    spent = [0.0, 0]

    def timed_evaluator(states):
        started = time.perf_counter()
        result = evaluator(states)
        spent[0] += time.perf_counter() - started
        spent[1] += 1
        return result

    player = SearchPlayer("bench", "R", "bench", timed_evaluator, SearchSettings(simulations))
    states = positions(count, seed)
    player.choose(states[0].copy(), seed=0)  # warm-up
    spent[0], spent[1] = 0.0, 0
    started = time.perf_counter()
    for state in states:
        player.choose(state.copy(), seed=0)
    total = time.perf_counter() - started
    sims = simulations * len(states)
    return dict(searches=len(states), simulations=sims, evaluator_calls=spent[1],
                ms_per_simulation=1000 * total / sims, evaluator_ms_per_simulation=1000 * spent[0] / sims,
                other_ms_per_simulation=1000 * (total - spent[0]) / sims, evaluator_share=spent[0] / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--searches", type=int, default=6, help="searches per process")
    parser.add_argument("--processes", type=int, default=5, help="concurrent processes for the contention test")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    import numpy as np
    import torch
    from chess_rl.adapter import encode_position, legal_mask
    from chess_rl.runtime import configure
    from chess_rl.search.real_evaluator import RealEvaluator
    configure(args.device, threads=1)
    report = dict(device=args.device, host=platform.node(), torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0) if args.device != "cpu" else None,
                  checkpoint=args.checkpoint, simulations=args.simulations)
    states = positions(64, 1)
    report["cpu_per_position"] = dict(
        encode_position=timed(lambda: [encode_position(s) for s in states], 5),
        legal_mask=timed(lambda: [legal_mask(s) for s in states], 5), positions=len(states))
    for key in ("encode_position", "legal_mask"):
        report["cpu_per_position"][key]["per_position_ms"] = report["cpu_per_position"][key]["median_ms"] / len(states)
    evaluator = RealEvaluator(args.checkpoint) if args.device == "cpu" else RealEvaluator(args.checkpoint, device=args.device)
    model = evaluator.checkpoint.model
    dev = next(model.parameters()).device
    features = torch.from_numpy(np.stack([encode_position(s).features for s in states])).to(dev)
    network, calls = {}, {}
    for batch in (1, 8, 32, 64):
        x = features[:batch]

        def forward():
            with torch.inference_mode():
                policy, wdl = model(x)
                policy.float().cpu(), wdl.float().cpu()  # include the device-to-host copy, as the evaluator does

        forward()
        evaluator(states[:batch])
        network[batch] = timed(forward, args.repeats)
        calls[batch] = timed(lambda: evaluator(states[:batch]), max(5, args.repeats // 5))
        for record in (network[batch], calls[batch]):
            record["per_position_ms"] = record["median_ms"] / batch
    report["network_only"] = network
    report["evaluator_call"] = calls
    report["search_single_process"] = searches(args.checkpoint, args.device, args.simulations, args.searches, 2, False)
    with ProcessPoolExecutor(args.processes, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(searches, args.checkpoint, args.device, args.simulations, args.searches, 10 + i, True)
                   for i in range(args.processes)]
        concurrent = [f.result() for f in futures]
    report["search_concurrent"] = dict(
        processes=args.processes, per_process=concurrent,
        mean_ms_per_simulation=statistics.fmean(c["ms_per_simulation"] for c in concurrent),
        mean_evaluator_ms_per_simulation=statistics.fmean(c["evaluator_ms_per_simulation"] for c in concurrent),
        mean_other_ms_per_simulation=statistics.fmean(c["other_ms_per_simulation"] for c in concurrent))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
