"""Development-only acceptance fixtures; never read the reserved opening registry."""
from concurrent.futures import ThreadPoolExecutor
import random
import time

import numpy as np

from chess_rl.adapter import make_state, terminal_result
from chess_rl.search.batched_evaluator import EvaluationCollector, FixedBatchEvaluator
from chess_rl.search.real_evaluator import RealEvaluator


def development_states(random_count=24):
    import chess
    cycle = ['g1f3', 'g8f6', 'f3h4', 'f6h5', 'h4f3', 'h5f6', 'f3g1', 'f6g8']
    states = [make_state(), make_state(moves_uci=['e2e4']),
              *[make_state(f'r3k2r/8/8/8/8/8/8/R3K2R {c} KQkq - 17 1') for c in ('w', 'b')],
              make_state(moves_uci=['e2e4', 'a7a6', 'e4e5', 'd7d5']),
              make_state(moves_uci=['a2a3', 'e7e5', 'd2d4', 'e5e4', 'f2f4']),
              make_state('3r1r1k/4P3/8/8/8/8/8/K7 w - - 0 1'),
              make_state('k7/8/8/8/8/8/4p3/3R1R1K b - - 0 1'),
              make_state(moves_uci=cycle + ['e2e4', 'e7e5', 'g1f3']),
              make_state(chess.STARTING_FEN, cycle + ['e2e4'], history_complete=False),
              make_state(moves_uci=['g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1']),
              make_state('4k3/8/8/8/8/8/8/R3K3 w - - 99 1')]
    rng = random.Random(20261018)
    while len(states) < random_count + 12:
        board = chess.Board()
        for _ in range(rng.randint(6, 100)):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(sorted(board.legal_moves, key=lambda m: m.uci())))
        state = make_state(moves_uci=[m.uci() for m in board.move_stack])
        if terminal_result(state) is None:
            states.append(state)
    return states


def identical(a, b):
    return (a.priors.tobytes() == b.priors.tobytes() and a.wdl.tobytes() == b.wdl.tobytes()
            and np.float64(a.value).tobytes() == np.float64(b.value).tobytes())


def probe(reference, batch_size, *, states=None, repeats=2):
    states = states or development_states()
    fixed = FixedBatchEvaluator(reference, batch_size)
    counts = sorted({1, min(2, batch_size), max(1, batch_size // 2), batch_size})
    failures, comparisons, max_abs = [], 0, 0.0
    baseline = [fixed([state])[0] for state in states]
    for repeat in range(repeats):
        for position, state in enumerate(states):
            for count in counts:
                for slot in range(count):
                    neighbours = [states[(position + repeat + i + 1) % len(states)] for i in range(count)]
                    if repeat % 2:
                        neighbours.reverse()
                    neighbours[slot] = state
                    result = fixed(neighbours)[slot]
                    comparisons += 1
                    error = max(float(np.max(np.abs(result.priors - baseline[position].priors))),
                                float(np.max(np.abs(result.wdl - baseline[position].wdl))),
                                abs(result.value - baseline[position].value))
                    max_abs = max(max_abs, error)
                    if not identical(result, baseline[position]) and len(failures) < 20:
                        failures.append(dict(position=position, slot=slot, real_requests=count,
                                             repeat=repeat, max_abs=error))
    return dict(bit_identical=not failures, comparisons=comparisons, max_abs=max_abs, failures=failures)


def arrival_probe(path, checkpoint_hash, batch_size, *, device='cpu', states=None):
    states = states or development_states(0)
    baselines = None
    differences = []
    for flush in (.2, 5.0):
        results = {}
        with EvaluationCollector({checkpoint_hash: lambda: RealEvaluator(path, device=device)},
                                 batch_size=batch_size, games=len(states), max_wait_ms=flush) as collector:
            def game(index):
                time.sleep((index % 4) * flush / 2000)
                with collector.game(str(index)) as evaluate:
                    results[index] = evaluate(checkpoint_hash, [states[index]])[0]
            order = list(range(len(states)))
            if flush > 1:
                order.reverse()
            with ThreadPoolExecutor(len(states)) as pool:
                list(pool.map(game, order))
        if baselines is None:
            baselines = results
        else:
            differences = [i for i in results if not identical(results[i], baselines[i])]
    return dict(bit_identical=not differences, changed_positions=differences)
