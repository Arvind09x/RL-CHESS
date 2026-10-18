"""Isolated opt-in acceptance/search/game runs. Never submits jobs or changes production."""
import argparse
import json
from pathlib import Path
import time

from chess_rl import selfplay_campaign as C
from chess_rl.selfplay import read_game_record
from chess_rl.selfplay_execution import Execution, source_identity
from chess_rl.selfplay_benchmark import (corpus, hardware, numerical_acceptance,
                                        benchmark_search, resume_probe)

ROOT = Path(__file__).resolve().parents[1]


def publish(path, value):
    """Small benchmark outputs use the same atomic/fsynced writer; plain JSON for CLI consumers."""
    C._publish_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def game_run(path, checkpoint, device, execution, acceptance, games, seed, simulations, deadline):
    from chess_rl.replay import ReplayBuffer
    plan = C.create(path, checkpoint, games=games, seed=seed, purpose='isolated self-play batching game benchmark',
                    simulations=simulations, device=device, execution=execution, acceptance=acceptance)
    started = time.monotonic()
    status = C.run(path, workers=execution.workers, deadline=deadline)
    seconds = time.monotonic() - started
    finished = sum(status['status_counts'][s] for s in C.FINAL)
    events = C._Events(path / 'events.jsonl').read()
    positions = sum(len(read_game_record(path / g['directory'])['positions'])
                    for g in plan['games'] if (path / g['directory'] / 'boundary.json').exists())
    report = dict(execution=execution.to_dict(), settings=plan['settings'], games=games, seed=seed,
                  wall_seconds=seconds, status=status, generated_positions=positions,
                  positions_per_second=positions / seconds,
                  committed_simulations_per_second=positions * simulations / seconds,
                  failures=[e for e in events if e['event'] in ('pool-broken', 'run-stopped')
                            or e['event'] == 'game-end' and e.get('status') not in C.FINAL],
                  worker_metrics=[e for e in events if e['event'] == 'batch-worker'],
                  rate_note='Only committed simulations counted; aborted search work is excluded.')
    if finished == games:
        report['archive'] = C.archive(path)
        buffer = ReplayBuffer.create(path / 'benchmark-replay.sqlite')
        usable = 0
        for line in (path / 'games.jsonl').read_text().splitlines():
            usable += buffer.ingest(json.loads(line))['added_samples']
        buffer.close()
        # These are measured capped game runs, explicitly distinguish actual terminal games.
        report.update(finished_games_per_hour=finished * 3600 / seconds,
                      completed_games_per_hour=status['status_counts']['completed'] * 3600 / seconds,
                      usable_training_positions=usable, usable_training_positions_per_hour=usable * 3600 / seconds,
                      rate_scope='Actual full campaign from standard start with production 512-ply cap; '
                                 'replay excludes truncations/duplicates. Timing includes startup/play, excludes ingest.')
    else:
        report['rate_note'] += ' Campaign incomplete: game/training-production rates withheld.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--source-record', type=Path)
    parser.add_argument('--mode', choices=('accept', 'search', 'games'), default='search')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--active-games-per-worker', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--positions', type=int, default=160)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--games', type=int, default=160)
    parser.add_argument('--deadline-seconds', type=int, default=6600)
    args = parser.parse_args()
    if len(args.batch_sizes) != len(args.active_games_per_worker):
        parser.error('provide one active-game count per batch size')
    if (len(set(args.batch_sizes)) != len(args.batch_sizes) or any(b < 2 for b in args.batch_sizes)
            or min(args.repeats, args.simulations, args.games, args.positions, args.deadline_seconds) < 1
            or args.seed < 0):
        parser.error('positive counts and distinct B >= 2 required; seed must be nonnegative')
    if args.mode == 'search' and args.positions < 5:
        parser.error('at least five positions are required for the five-worker comparison')
    if args.device == 'cuda:0' and args.source_record is None:
        parser.error('CUDA benchmark requires --source-record prepared after committing')
    source_record = None
    if args.source_record:
        from prepare_batching_source import verify
        source_record = verify(ROOT, args.source_record)
    def check_pin():
        if C._sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('checkpoint hash differs from explicit pin')
    check_pin()
    from chess_rl.runtime import configure
    configure(args.device)
    layouts = [Execution(workers=5)] + [Execution(mode='batched', workers=5, batch_size=b,
               games_per_worker=k) for b, k in zip(args.batch_sizes, args.active_games_per_worker)]
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + args.deadline_seconds
    metadata = dict(schema='selfplay-benchmark-v1', arguments={k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, hardware=hardware(), source=source_identity(),
                    committed_source=source_record, layouts=[e.to_dict() for e in layouts],
                    note='Both saved benchmark-576 and benchmark-3g-577 identified B200 MIG 1g.23gb. '
                         'This report uses actual hardware above; no full-B200 or whole-game extrapolation.')
    publish(args.output / 'metadata.json', metadata)
    try:
        acceptances = {}
        for execution in layouts[1:]:
            check_pin()
            if time.time() >= deadline:
                raise TimeoutError('benchmark deadline before acceptance completed')
            report = numerical_acceptance(args.checkpoint, args.device, execution.batch_size)
            check_pin()
            publish(args.output / f'acceptance-b{execution.batch_size}.json', report)
            if not report['accepted']:
                raise ValueError(f'B={execution.batch_size} numerical acceptance failed; results retained')
            # Validation is explicitly scoped to the same frozen execution configuration.
            resume = resume_probe(args.output / f'resume-b{execution.batch_size}', args.checkpoint,
                                  args.device, execution, report)
            publish(args.output / f'resume-b{execution.batch_size}.json', resume)
            if not resume['accepted']:
                raise ValueError('same-configuration interrupted/resumed comparison failed')
            acceptances[execution.batch_size] = report
        reports = []
        if args.mode == 'search':
            rows = corpus(args.positions, args.seed)
            publish(args.output / 'corpus.json', rows)
            # Rotate the order on repeats to expose contention/thermal/order effects.
            for repeat in range(args.repeats):
                order = layouts[repeat % len(layouts):] + layouts[:repeat % len(layouts)]
                for execution in order:
                    check_pin()
                    if time.time() >= deadline:
                        raise TimeoutError('benchmark deadline; completed layout reports retained')
                    result = benchmark_search(args.checkpoint, args.device, execution, rows,
                                              args.simulations, acceptances.get(execution.batch_size))
                    result['repeat'] = repeat
                    publish(args.output / f'search-r{repeat}-b{execution.batch_size}.json', result)
                    reports.append(result)
                    if result['failures']:
                        raise RuntimeError('benchmark search failures; results retained')
        elif args.mode == 'games':
            for execution in layouts:
                check_pin()
                if time.time() >= deadline:
                    raise TimeoutError('benchmark deadline; earlier campaign results retained')
                result = game_run(args.output / f'games-b{execution.batch_size}', args.checkpoint,
                                  args.device, execution, acceptances.get(execution.batch_size),
                                  args.games, args.seed, args.simulations, deadline)
                publish(args.output / f'games-b{execution.batch_size}.json', result)
                reports.append(result)
                if result['failures'] or 'usable_training_positions_per_hour' not in result:
                    raise RuntimeError('game comparison incomplete or failed; resume campaign separately')
        publish(args.output / 'complete.json', dict(accepted=True, mode=args.mode,
                    reports=len(reports), finished_utc=C._now(), hardware=hardware()))
    except Exception as error:
        publish(args.output / 'failure.json', dict(error=f'{type(error).__name__}: {error}', time=C._now()))
        raise


if __name__ == '__main__':
    main()
