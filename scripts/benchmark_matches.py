"""Development-only batching acceptance, full-game resume checks and match timing.

Uses generated openings and explicit checkpoint paths. No production records or
reserved registry files are read. New plans are immutable; `run` resumes them.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    from chess_rl.evaluation.pilot import atomic_json
    atomic_json(path, value)


def accept(args):
    import numpy as np
    from chess_rl.runtime import configure
    from chess_rl.evaluation import development_matches as M
    from chess_rl.evaluation.batching_acceptance import development_states, probe, arrival_probe
    from chess_rl.search.batched_evaluator import FixedBatchEvaluator
    from chess_rl.search.real_evaluator import RealEvaluator
    from chess_rl.search.puct import Node
    from chess_rl.search.runner import search
    configure(args.device, threads=1)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(runtime=M.runtime_identity(args.device), sources=M.M.source_hashes(ROOT),
                  batch_size=args.batch_size, checkpoints=[])
    states = development_states()
    for path in (args.left, args.right):
        sha = M.sha_file(path)
        reference = RealEvaluator(path, device=args.device)
        fixed = FixedBatchEvaluator(reference, args.batch_size)
        numeric = probe(reference, args.batch_size, states=states)
        agreement, moves, max_abs = True, True, 0.0
        margins = []
        for state in states:
            a, = reference([state])
            b, = fixed([state])
            error = max(float(np.max(np.abs(a.priors-b.priors))), float(np.max(np.abs(a.wdl-b.wdl))),
                        abs(a.value-b.value))
            max_abs = max(max_abs, error)
            agreement &= error <= 1e-5  # Frozen absolute bound; no rounding.
            positive = np.sort(a.priors[a.priors > 0])
            margins.append(float(positive[-1]-positive[-2]) if len(positive)>1 else 1.0)
            moves &= search(Node(state), 4, evaluator=reference).move_id == search(Node(state), 4, evaluator=fixed).move_id
        arrivals = arrival_probe(path, sha, args.batch_size, device=args.device)
        report['checkpoints'].append(dict(path=str(path.resolve()), sha256=sha, fixed_batch=numeric,
            arrivals=arrivals, serial_tolerance=1e-5, serial_max_abs=max_abs, serial_agreement=bool(agreement),
            identical_selected_moves=bool(moves), minimum_raw_prior_margin=min(margins)))
        del fixed, reference
    report['accepted'] = all(c['fixed_batch']['bit_identical'] and c['arrivals']['bit_identical']
                            and c['serial_agreement'] and c['identical_selected_moves'] for c in report['checkpoints'])
    write(directory / 'acceptance.json', report)
    print(json.dumps(dict(accepted=report['accepted'], report=str(directory / 'acceptance.json'))))
    if not report['accepted']:
        raise SystemExit('Batching rejected: numerical or move acceptance failed; production remains serial.')


def overlap(values):
    from datetime import datetime
    events = []
    for value in values.values():
        timing = value['timing']
        events += [(datetime.fromisoformat(timing['started_utc']).timestamp(), 1),
                   (datetime.fromisoformat(timing['finished_utc']).timestamp(), -1)]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def benchmark(args):
    from chess_rl.evaluation import development_matches as M
    from chess_rl.evaluation.development_matches import freeze
    from chess_rl.evaluation.benchmark_layouts import parse_layout
    layouts = [(layout, parse_layout(layout, batch_size=args.batch_size, max_wait_ms=args.flush_ms))
               for layout in args.layouts]
    if len({layout for layout, _ in layouts}) != len(layouts):
        raise ValueError('duplicate benchmark layouts')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runtime = M.worker_runtime(args.device)
    results = []
    batched_records = {}
    for layout, execution in layouts:
        mode = execution.mode
        folder = directory / layout
        p = freeze(folder, ROOT, (args.left,args.right), execution, runtime, pairs=args.pairs,
                   simulations=args.simulations, ply_cap=args.ply_cap)
        started = time.monotonic()
        M.run(folder, ROOT)
        wall = time.monotonic() - started
        values = M.read_results(folder / 'results.sqlite3', p)
        timing = M.timing_report(p, values)
        metrics = [json.loads(f.read_text()) for f in folder.glob('collector-*.json')]
        worker_memory = [json.loads(f.read_text()) for f in folder.glob('worker-memory-*.json')]
        current_records = {k:v['record'] for k,v in values.items()}
        scheduling_identical = None
        if mode == 'batched':
            baseline = batched_records.get(execution.batch_size)
            scheduling_identical = baseline is None or baseline == current_records
            if baseline is None:
                batched_records[execution.batch_size] = current_records
        plies = sum(v['timing']['plies_played'] for v in values.values())
        statuses = {s:sum(v['record']['status']==s for v in values.values())
                    for s in ('completed','truncated','failed','interrupted')}
        total_batches = sum(m['batches'] for m in metrics)
        requests = sum(m['requests'] for m in metrics)
        peak = overlap(values)
        result = dict(layout=layout, execution=execution.to_dict(), plan_sha256=p['sha256'],
            total_wall_seconds=wall, warmed_game_span_seconds=timing['wall_seconds'], plies=plies,
            total_wall_plies_per_second=plies/wall,
            warmed_plies_per_second=plies/timing['wall_seconds'] if timing['wall_seconds'] else None,
            statuses=statuses, peak_overlapping_games=peak, collector_metrics=metrics,
            worker_memory=worker_memory, scheduling_identical_to_first_batched=scheduling_identical,
            scheduling_comparison_batch_size=execution.batch_size if mode == 'batched' else None,
            concurrent_capacity=execution.processes * execution.games_per_process,
            schedule_fills_capacity=p['planned_games'] >= execution.processes * execution.games_per_process,
            batch_occupancy=None if not total_batches else requests/(execution.batch_size*total_batches),
            mean_queue_seconds=None if not requests else sum(m['queue_seconds'] for m in metrics)/requests)
        results.append(result)
        write(directory / 'benchmark.json', dict(runtime=runtime, results=results,
            note='Total wall includes spawn/load/warmup; warmed span begins after per-process model warmup/barrier. '
                 'Same synthetic openings/checkpoints/seeds/budget in all layouts. '
                 'Exact game comparisons apply only within the same B. Underfilled layouts are labelled. No strength claims.'))
        if scheduling_identical is False:
            raise SystemExit('Full-game scheduling independence failed; production remains serial.')
        if statuses['failed'] or statuses['interrupted'] or (execution.processes * execution.games_per_process > 1 and p['planned_games'] > 1 and peak < 2):
            raise SystemExit('Benchmark rejected: failed games or workload did not overlap.')
    print(json.dumps(results, indent=2))


def check_games(args):
    from chess_rl.evaluation import development_matches as M
    from chess_rl.evaluation.development_matches import freeze
    from chess_rl.evaluation.execution import Execution
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runtime = M.worker_runtime(args.device)
    config = Execution(mode='batched', processes=1, games_per_process=4, batch_size=args.batch_size)
    reference = None
    for name, execution in (('baseline',config), ('reordered',replace(config,processes=2,games_per_process=2,max_wait_ms=7)),
                            ('killed',replace(config,games_per_process=2))):
        folder = directory / name
        p = freeze(folder, ROOT, (args.left,args.right), execution, runtime,
                   pairs=4, simulations=4, ply_cap=24)
        before = {}
        if name == 'killed':
            with (folder / 'killed.log').open('w') as log:
                process = subprocess.Popen([sys.executable, __file__, 'run', '--directory', str(folder)],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic()+180
                    while not before and time.monotonic()<deadline:
                        try:
                            before = M.read_results(folder / 'results.sqlite3', p)
                        except sqlite3.OperationalError:
                            pass
                        if process.poll() is not None:
                            break
                        time.sleep(.01)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            if not 0 < len(before) < p['planned_games']:
                raise SystemExit('Kill/resume acceptance did not interrupt a partially committed workload')
        M.run(folder, ROOT)
        values = M.read_results(folder / 'results.sqlite3', p)
        if any(values[k] != v for k,v in before.items()):
            raise SystemExit('Committed row changed after resume')
        records = {k:v['record'] for k,v in values.items()}
        if any(r['status'] in ('failed','interrupted') for r in records.values()):
            raise SystemExit('Full-game acceptance had failures')
        if reference is None:
            reference = records
        elif records != reference:
            write(directory / 'game-acceptance.json', dict(accepted=False, difference=name))
            raise SystemExit('Full-game records differ with scheduling/resume; production remains serial')
    write(directory / 'game-acceptance.json', dict(accepted=True, cases=['baseline','reordered','kill/resume']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    for command in ('accept','benchmark','check-games'):
        sub = commands.add_parser(command)
        sub.add_argument('--left',type=Path,required=True)
        sub.add_argument('--right',type=Path,required=True)
        sub.add_argument('--output',type=Path,required=True)
        sub.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
        sub.add_argument('--batch-size',type=int,default=32)
        if command == 'benchmark':
            from chess_rl.evaluation.benchmark_layouts import DEFAULT_LAYOUTS
            sub.add_argument('--layouts',nargs='+',default=DEFAULT_LAYOUTS, help='PxKxB; PxK uses --batch-size')
            sub.add_argument('--flush-ms',type=float,default=20.0)
            sub.add_argument('--pairs',type=int,default=8)
            sub.add_argument('--simulations',type=int,default=128)
            sub.add_argument('--ply-cap',type=int,default=512)
    run = commands.add_parser('run')
    run.add_argument('--directory',type=Path,required=True)
    run.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    if args.command == 'accept':
        accept(args)
    elif args.command == 'benchmark':
        benchmark(args)
    elif args.command == 'check-games':
        check_games(args)
    else:
        from chess_rl.evaluation.development_matches import run
        run(args.directory, args.root)


if __name__ == '__main__':
    main()
