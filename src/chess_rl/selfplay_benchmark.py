"""Opt-in self-play search benchmark and numerical admission; no match protocol inputs."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import chess
import numpy as np
import torch

from chess_rl.adapter import make_state
from chess_rl.evaluation.batching_acceptance import development_states, probe, arrival_probe
from chess_rl.evaluation.matches import digest
from chess_rl.search.batched_evaluator import FixedBatchEvaluator
from chess_rl.search.puct import Node, SearchConfig
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.runner import search
from chess_rl.selfplay import Settings
from chess_rl.selfplay_execution import Execution, binding, session_runtime, source_identity
from chess_rl.selfplay_batching import (collector_for, bounded_games, BoundEvaluator, memory,
                                       validate_acceptance)
from chess_rl import selfplay_campaign as C


def corpus(count=160, seed=20261018):
    """Fixed full histories, stratified over opening/middle/late; no reserved openings."""
    if type(count) is not int or count < 1:
        raise ValueError('positive corpus size required')
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        target = (0, 20, 60, 120, 180)[index % 5]
        for _ in range(500):
            board = chess.Board()
            while len(board.move_stack) < target and not board.is_game_over(claim_draw=True):
                board.push(rng.choice(sorted(board.legal_moves, key=lambda m: m.uci())))
            if len(board.move_stack) == target and not board.is_game_over(claim_draw=True):
                break
        else:
            raise RuntimeError('could not generate nonterminal long-history fixture')
        rows.append(dict(index=index, seed=C.game_seed(seed, index),
                         moves_uci=[m.uci() for m in board.move_stack],
                         phase='opening' if target < 30 else 'middle' if target < 100 else 'late'))
    return rows


def hardware():
    def command(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError) as error:
            return f'unavailable: {error}'
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_initialized() else None
    return dict(host=platform.node(), platform=platform.platform(),
                cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                slurm={k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK',
                       'SLURM_MEM_PER_NODE', 'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS')},
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=dict(name=props.name, total_memory=props.total_memory,
                         uuid=str(getattr(props, 'uuid', 'unavailable'))) if props else None,
                nvidia_smi_L=command(['nvidia-smi', '-L']) if props else None,
                nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                    '--format=csv,noheader']) if props else None)


def numerical_acceptance(checkpoint, device, batch_size, *, states=None, repeats=2):
    """Exact fixed-B slot/neighbour/padding/arrival checks; tolerance only against B=1."""
    sha, source = C._sha256(checkpoint), source_identity()
    reference = RealEvaluator(checkpoint, device=device)
    fixed = FixedBatchEvaluator(reference, batch_size)
    if states is None:
        states = development_states(0) + [make_state(moves_uci=r['moves_uci']) for r in corpus(10)]
    fixed_report = probe(reference, batch_size, states=states, repeats=repeats)
    serial_error, selected_move_differences = 0., []
    for index, state in enumerate(states):
        a, b = reference([state])[0], fixed([state])[0]
        serial_error = max(serial_error, float(np.max(np.abs(a.priors - b.priors))),
                           float(np.max(np.abs(a.wdl - b.wdl))), abs(a.value - b.value))
        if search(Node(state), 2, evaluator=reference).move_id != search(Node(state), 2, evaluator=fixed).move_id:
            selected_move_differences.append(index)
    arrival = arrival_probe(checkpoint, sha, batch_size, device=device, states=states)
    if C._sha256(checkpoint) != sha or source_identity() != source:
        raise ValueError('checkpoint/source changed during numerical acceptance')
    current = session_runtime(device)
    current['batch_size'] = batch_size
    record = dict(schema='selfplay-numerical-acceptance-v1', checkpoint_sha256=sha,
                  batch_size=batch_size, source_sha256=source['sha256'], runtime=current,
                  fixed_shape=fixed_report, changed_arrival=arrival,
                  serial_max_abs=serial_error, serial_atol=1e-5,
                  serial_selected_move_differences=selected_move_differences,
                  positions=len(states), history_plies=[len(s.board.move_stack) for s in states],
                  accepted=fixed_report['bit_identical'] and arrival['bit_identical'] and serial_error <= 1e-5,
                  hardware=hardware(), scope='Exact checks apply only within this frozen B/runtime/checkpoint; '
                  'serial numerical agreement does not imply identical cross-B game trajectories.')
    return dict(record, sha256=digest(record))


_BARRIER = None


def init_worker(device, barrier):
    global _BARRIER
    from chess_rl.runtime import configure
    configure(device)
    _BARRIER = barrier


class TimedSerial:
    def __init__(self, reference):
        self.reference = reference
        self.metrics = dict(batches=0, requests=0, padded_slots=0, queue_seconds=0.,
                            inference_seconds=0., peak_pending=1, flush_counts={'serial': 0})

    def __call__(self, states):
        start = time.monotonic()
        result = self.reference(states)
        self.metrics['inference_seconds'] += time.monotonic() - start
        self.metrics['batches'] += 1
        self.metrics['requests'] += len(states)
        self.metrics['flush_counts']['serial'] += 1
        return result


def search_worker(rows, plan):
    execution = Execution.from_dict(plan['execution'])
    current = session_runtime(plan['device'], binding(plan))
    if current != plan['expected_runtime']:
        raise ValueError('benchmark worker runtime differs from frozen runtime')
    if execution.mode == 'batched':
        validate_acceptance(plan['numerical_acceptance'], plan, current)
    settings = Settings.from_dict(plan['settings'])
    failures, results = [], []

    def one(row, evaluator):
        started = time.monotonic()
        try:
            state = make_state(moves_uci=row['moves_uci'])
            result = search(Node(state), settings.simulation_budget,
                            config=SearchConfig(settings.c_puct, settings.unvisited_value),
                            evaluator=evaluator, exploration=settings.exploration(len(row['moves_uci'])),
                            rng=np.random.Generator(np.random.PCG64(row['seed'])))
            return dict(index=row['index'], phase=row['phase'], history_plies=len(row['moves_uci']),
                        simulations=result.completed_simulations, seconds=time.monotonic() - started,
                        result_sha256=digest([result.move.uci(), result.root_visits.tolist()]))
        except Exception as error:
            return dict(index=row['index'], failure=f'{type(error).__name__}: {error}')

    def start():
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if _BARRIER is not None:
            _BARRIER.wait(timeout=180)  # sleeping barrier; setup failures cannot strand peers forever
        return time.monotonic()

    if execution.mode == 'batched':
        with collector_for(plan) as (collector, checkpoint):
            with collector.game('warmup') as channel:
                for _ in range(2):
                    channel(plan['checkpoint']['sha256'], [make_state()])
            # Exclude warmup counters, retain actual high-water pending count.
            warmup = deepcopy(collector.metrics)
            started = start()
            def play(row):
                with collector.game(row['index']) as channel:
                    channel.expect(plan['checkpoint']['sha256'])
                    return one(row, BoundEvaluator(channel, checkpoint, plan['checkpoint']['sha256']))
            results = list(bounded_games(rows, execution.games_per_worker, play))
            ended = time.monotonic()
            metrics = deepcopy(collector.metrics)
            for key in ('batches', 'requests', 'padded_slots', 'queue_seconds', 'inference_seconds'):
                metrics[key] -= warmup[key]
            metrics['flush_counts'] = {k: v - warmup['flush_counts'][k] for k, v in metrics['flush_counts'].items()}
            metrics.pop('per_checkpoint')
    else:
        if C._sha256(plan['checkpoint']['path']) != plan['checkpoint']['sha256']:
            raise ValueError('benchmark checkpoint changed')
        reference = RealEvaluator(plan['checkpoint']['path'], device=plan['device'])
        if C._sha256(plan['checkpoint']['path']) != plan['checkpoint']['sha256']:
            raise ValueError('benchmark checkpoint changed during serial model loading')
        for _ in range(2):
            reference([make_state()])
        evaluator = TimedSerial(reference)
        started = start()
        results = [one(row, evaluator) for row in rows]
        ended = time.monotonic()
        metrics = evaluator.metrics
    failures = [r for r in results if 'failure' in r]
    return dict(started=started, ended=ended, results=results, failures=failures,
                collector=metrics, memory=memory(), pid=os.getpid(),
                runtime=session_runtime(plan['device'], binding(plan)))


def summarize_workers(workers, batch_size, wall_seconds):
    seconds = max(w['ended'] for w in workers) - min(w['started'] for w in workers)
    good = [r for w in workers for r in w['results'] if 'failure' not in r]
    batches = sum(w['collector']['batches'] for w in workers)
    padding = sum(w['collector']['padded_slots'] for w in workers)
    flushes = {}
    for worker in workers:
        for key, value in worker['collector']['flush_counts'].items():
            flushes[key] = flushes.get(key, 0) + value
    return dict(timed_wall_seconds=seconds, wall_seconds_including_startup=wall_seconds,
                completed_positions=len(good), completed_simulations=sum(r['simulations'] for r in good),
                positions_per_second=len(good) / seconds,
                simulations_per_second=sum(r['simulations'] for r in good) / seconds,
                useful_batch_occupancy=(batches * batch_size - padding) / (batches * batch_size) if batches else 0.,
                batches=batches, padded_slots=padding, flush_counts=flushes,
                requests=sum(w['collector']['requests'] for w in workers),
                queue_seconds=sum(w['collector']['queue_seconds'] for w in workers),
                inference_seconds=sum(w['collector']['inference_seconds'] for w in workers),
                failures=[r for w in workers for r in w['failures']],
                per_worker=workers,
                note='Independent full-history searches with self-play exploration; no measured games/hour '
                     'or usable training positions/hour. Per-process memory peaks are not simultaneous job peaks.')


def benchmark_search(checkpoint, device, execution, rows, simulations, acceptance=None):
    plan = dict(checkpoint=dict(path=str(Path(checkpoint).resolve()), sha256=C._sha256(checkpoint)),
                device=device, execution=execution.to_dict(), execution_source=source_identity(),
                settings=asdict(C.frozen_settings(simulations)))
    plan['expected_runtime'] = session_runtime(device, binding(plan))
    if execution.mode == 'batched':
        if acceptance is None:
            raise ValueError('search benchmark requires numerical acceptance')
        validate_acceptance(acceptance, plan, plan['expected_runtime'])
        plan['numerical_acceptance'] = acceptance
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(execution.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(execution.workers, mp_context=context, initializer=init_worker,
                             initargs=(device, barrier)) as pool:
        futures = [pool.submit(search_worker, rows[i::execution.workers], plan)
                   for i in range(execution.workers)]
        workers = [future.result() for future in futures]
    result = summarize_workers(workers, execution.batch_size, time.monotonic() - started)
    active_capacity = [min(execution.games_per_worker, len(rows[i::execution.workers]))
                       for i in range(execution.workers)]
    return dict(execution=execution.to_dict(), settings=plan['settings'], corpus_sha256=digest(rows),
                checkpoint=plan['checkpoint'], effective_active_games_per_worker=active_capacity,
                underfilled=any(k < execution.batch_size for k in active_capacity),
                **result)


def resume_probe(directory, checkpoint, device, execution, acceptance):
    """Small same-configuration whole-Session comparison, not a throughput measurement."""
    import shutil
    from chess_rl.selfplay import Session, read_game_record
    from chess_rl.selfplay_batching import play_group, freeze_runtime
    directory = Path(directory)
    a, b = directory / 'control', directory / 'resumed'
    plan = C.create(a, checkpoint, games=min(3, execution.games_per_worker), seed=1983,
                    purpose='numerical acceptance: tiny interrupted/resumed game probe',
                    settings=Settings(2, 4), device=device, execution=execution, acceptance=acceptance)
    shutil.copytree(a, b)
    def tasks(path):
        return [dict(index=g['index'], directory=str(path / g['directory']), seed=g['seed'],
                     checkpoint=plan['checkpoint']['path'], opening=plan['opening'], settings=plan['settings'],
                     campaign_seed=plan['seed'], device=device) for g in plan['games']]
    play_group(tasks(a), plan, a)
    freeze_runtime(b, plan)
    with collector_for(plan) as (collector, checkpoint_info):
        def stop(task):
            with collector.game(task['index']) as channel:
                channel.expect(plan['checkpoint']['sha256'])
                evaluator = BoundEvaluator(channel, checkpoint_info, plan['checkpoint']['sha256'])
                session = Session.create(task['directory'], checkpoint, C.STANDARD_OPENING,
                    settings=Settings.from_dict(plan['settings']), seed=task['seed'], opening_selection_seed=plan['seed'],
                    device=device, journal=True, evaluator=evaluator, execution=binding(plan))
                session.run(stop_after_moves=2)
        list(bounded_games(tasks(b), execution.games_per_worker, stop))
    play_group(list(reversed(tasks(b))), plan, b)
    differences = []
    for game in plan['games']:
        left, right = [read_game_record(p / game['directory']) for p in (a, b)]
        left.pop('interruptions'); right.pop('interruptions')
        if left != right or left['status'] not in C.FINAL:
            differences.append(game['index'])
    C.archive(a); C.archive(b)
    return dict(accepted=not differences, changed_games=differences, games=len(plan['games']),
                plies=4, simulations=2, scope='same frozen configuration; cooperative move-boundary interruption')
