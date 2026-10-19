"""Read-only synchronized evaluator profiling; no opening registry or games."""
import argparse
from contextlib import ExitStack
import cProfile
import json
from pathlib import Path
import pstats
import time
from unittest.mock import patch


def profile(checkpoint, device, batch_size, output):
    import torch
    from chess_rl.evaluation.batching_acceptance import development_states
    from chess_rl.search import real_evaluator as R
    from chess_rl.search.batched_evaluator import FixedBatchEvaluator
    states = development_states(0)
    reference = R.RealEvaluator(checkpoint, device=device)
    fixed = FixedBatchEvaluator(reference, batch_size)
    sync = (lambda: torch.cuda.synchronize()) if device == 'cuda:0' else lambda: None
    for _ in range(3):
        fixed(states[:min(len(states), batch_size)])
    sync()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda:0':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    durations = {}
    def span(name, fn):
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(name):
                sync()
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                sync()
                durations[name] = durations.get(name, 0.0) + time.perf_counter() - start
                return result
        return wrapped
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        for name in ('terminal_result', 'legal_mask', 'encode_position', 'validate_legal_support',
                     'finite_matrix', 'legal_policy_probs'):
            stack.enter_context(patch.object(R, name, span(name, getattr(R, name))))
        model = reference.checkpoint.model
        stack.enter_context(patch.object(model, 'forward', span('network', model.forward)))
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            cpu = cProfile.Profile()
            cpu.enable()
            sync()
            start = time.perf_counter()
            for _ in range(5):
                fixed(states[:min(len(states), batch_size)])
            sync()
            total = time.perf_counter() - start
            cpu.disable()
    cpu.dump_stats(str(output / 'cpu.pstats'))
    with (output / 'cpu.txt').open('w') as f:
        pstats.Stats(cpu, stream=f).sort_stats('cumulative').print_stats(60)
    profiler.export_chrome_trace(str(output / 'trace.json'))
    (output / 'operators.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=80))
    (output / 'timings.json').write_text(json.dumps(dict(device=device, batch_size=batch_size,
        real_requests=min(len(states), batch_size), calls=5, synchronized_seconds=total,
        instrumented_phase_seconds=durations,
        note='Instrumented diagnostic only; sync and profiler overhead prevent throughput interpretation. '
             'Transfers, softmax, finite checks, sync and allocation details are in operators/trace.'), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=('cpu','cuda:0'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from chess_rl.runtime import configure
    configure(args.device, threads=1)
    profile(args.checkpoint, args.device, args.batch_size, args.output)


if __name__ == '__main__':
    main()
