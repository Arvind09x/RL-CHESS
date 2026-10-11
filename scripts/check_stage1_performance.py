"""Targeted CPU Stage 1 checks, synthetic valid sources; no campaign training.

Run from the repository root: .venv/bin/python scripts/check_stage1_performance.py
The predeclared criterion is median 10k/1k <=1.25 and bootstrap upper 95% <=1.35.
Setup and steady-state timings are separate. No neural self-play throughput claim.
"""
from dataclasses import replace
import argparse
import copy
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import torch

from chess_rl.adapter import make_state, terminal_result
from chess_rl.canonical_games import assign_split, canonical_payload, game_id
from chess_rl.replay import ReplayBuffer
from chess_rl.selfplay import Phase, Session, Settings
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import DataOrder, ModelConfig, Progress, save_checkpoint
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedSources, MixedState
from chess_rl.training.trainer import LossWeights
from tests import test_supervised as fixtures
from tests.test_checkpoints import assert_tree_equal
from tests.test_mixed_update import optimizer
from tests.test_selfplay import controlled, opening


CRITERION = dict(median_ratio_max=1.25,bootstrap_upper_95_max=1.35,
    samples=[1000,10000], batch_size=4, supervised_batch_size=2, cache_bytes=0,
    warmup_updates=5, measured_updates=30, paired_blocks=5,
    model="M06 float32; heads 7/5; dropout=0; standard Adam", threads=1)


def stats(values):
    return dict(raw_ms=values, median_ms=statistics.median(values),
        mean_ms=statistics.mean(values), stdev_ms=statistics.stdev(values),
        cv=statistics.stdev(values)/statistics.mean(values),
        min_ms=min(values), max_ms=max(values),
        q25_ms=float(np.quantile(values,.25)), q75_ms=float(np.quantile(values,.75)))


def valid_bundle(path, count):
    path.mkdir()
    original = fixtures.make_game
    counter = 0
    def unique_game(manifest, split, fen, moves):
        nonlocal counter
        while True:
            counter += 1
            start = fen.rsplit(' ',1)[0] + f' {counter}'
            identity = game_id(canonical_payload(start,list(moves)))
            if assign_split(identity,manifest.split_seed) == split:
                break
        game = original(manifest,split,fen,moves)
        game.update(game_id=identity,start_fen=start)
        return game
    fixtures.make_game = unique_game
    try:
        # Each unique game has an identical valid one-ply sample/target. This
        # gives equivalent numerical updates regardless of sampled reference.
        return fixtures.fixture_bundle(path, specifications=[
            ('train',chess.STARTING_FEN,['e2e4'],[1],None)] * count)
    finally:
        fixtures.make_game = original


def dataset_benchmark(out):
    setups, datasets, configs = {}, {}, {}
    for count in CRITERION['samples']:
        start=time.perf_counter()
        configs[count]=valid_bundle(out / f'data-{count}',count)
        generated=time.perf_counter()
        datasets[count]=open_dataset(**configs[count],split='train')
        loaded=time.perf_counter()
        setups[count]=dict(fixture_generation_s=generated-start,validation_open_s=loaded-generated)
    config=ModelConfig(d_policy=7,d_value=5,dropout=0)
    checkpoint=out/'checkpoint.pt'
    model=config.build()
    save_checkpoint(checkpoint,model=model,optimizer=optimizer(model,MixedSettings()),
        checkpoint_id='stage1-synthetic',provenance={'purpose':'Stage 1 bounded measurement fixture'},
        weights=LossWeights(),progress=Progress(0,0,0),data_order=DataOrder(
            'synthetic',DataOrder.reference_hash(datasets[1000]),tuple(range(1000)),0,4,False,0))
    result=subprocess.run([sys.executable,'-m','tests.selfplay_crash_worker','control',
        str(out/'replay-game'),str(checkpoint),str(out/'replay-game.json')],cwd=ROOT,
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    buffer=ReplayBuffer.create(out/'replay.sqlite')
    buffer.ingest(json.loads((out/'replay-game.json').read_text())['snapshot'])
    instances={}
    settings=MixedSettings()
    for count in CRITERION['samples']:
        args=configs[count]
        start=time.perf_counter()
        sources=MixedSources(datasets[count],buffer,
            heldout_games_path=args['prepared']/'heldout-games.jsonl',
            protected_positions_path=args['prepared']/'protected-heldout.jsonl',cache_bytes=0)
        setups[count]['freeze_and_membership_s']=time.perf_counter()-start
        torch.manual_seed(481)
        live=config.build()
        state=MixedState.initial(sources,settings,seed=32)
        # Supervised shuffles consume different RNG counts. The timed window
        # never wraps either supervised cursor; align the replay stream explicitly.
        state=replace(state,selfplay=type(state.selfplay)(tuple(range(len(sources.replay_membership)))),
                      sampling_rng=random.Random(91).getstate())
        instances[count]=[live,optimizer(live,settings),sources,state]
    buffer.close()
    def update(count):
        live,opt,sources,state=instances[count]
        start=time.perf_counter()
        _,instances[count][3]=mixed_step(live,opt,sources,state)
        return (time.perf_counter()-start)*1000
    for count in CRITERION['samples']:
        for _ in range(CRITERION['warmup_updates']): update(count)
    measured={n:[] for n in CRITERION['samples']}
    for block in range(CRITERION['paired_blocks']):
        for count in ([1000,10000] if block%2==0 else [10000,1000]):
            gc.collect()
            measured[count].extend(update(count) for _ in range(6))
        assert_tree_equal(instances[1000][0].state_dict(),instances[10000][0].state_dict())
    ratio=statistics.median(measured[10000])/statistics.median(measured[1000])
    rng=np.random.default_rng(184)
    boot=[np.median(rng.choice(measured[10000],30))/np.median(rng.choice(measured[1000],30))
          for _ in range(10000)]
    upper=float(np.quantile(boot,.95))
    return dict(setup=setups,steady_state={n:stats(v) for n,v in measured.items()},
        median_ratio=ratio,bootstrap_upper_95=upper,
        acceptance=ratio<=1.25 and upper<=1.35,equivalent_model_updates_exact=True),checkpoint


def long_game():
    """512 legal plies, no post-termination moves; quiet play and pawn resets."""
    rng=random.Random(761)
    state=make_state()
    moves=[]
    for ply in range(512):
        choices=list(state.board.legal_moves)
        rng.shuffle(choices)
        # Keep material. Pawn resets after ~70 quiet plies prevent fifty moves.
        prefer_pawn=state.board.halfmove_clock>=70
        choices.sort(key=lambda m:(state.board.is_capture(m),
            (state.board.piece_type_at(m.from_square)==chess.PAWN)!=prefer_pawn))
        for move in choices:
            board=state.board.copy(stack=True)
            board.push(move)
            if board.outcome(claim_draw=False) is None and not board.is_repetition(3) and not board.is_fifty_moves():
                state.board.push(move)
                moves.append(move.uci())
                break
        else: raise AssertionError(f'long fixture terminated at {ply}')
    # Independent checked admission of every retained legal move.
    make_state(moves_uci=moves)
    return moves


def deep_size(value, seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value))
    size=sys.getsizeof(value)
    if isinstance(value,dict):
        size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
    elif isinstance(value,(list,tuple)):
        size+=sum(deep_size(v,seen) for v in value)
    return size


def selfplay_profile(out,checkpoint):
    moves=long_game()
    session=Session.create(out/'long-game',checkpoint,opening(),settings=Settings(1,512,
        phases=(Phase(0,0.,0.),)),seed=971,opening_selection_seed=73,journal=True)
    session.evaluator=lambda states: (controlled(moves[len(states[0].board.move_stack)])(states)
        if len(states[0].board.move_stack)<512 else __import__('chess_rl.search.evaluator',fromlist=['evaluate']).evaluate(states))
    publication=session._publish
    publish_times=[]
    checkpoints=[]
    positions_id=id(session.record['positions'])
    def publish(record,**kwargs):
        ply=len(record['moves_uci'])
        tracked=ply in (1,32,64,128,256,384,480,512)
        if tracked: tracemalloc.start()
        start=time.perf_counter()
        publication(record,**kwargs)
        elapsed=(time.perf_counter()-start)*1000
        publish_times.append(elapsed)
        assert id(session.record['positions'])==positions_id
        if tracked:
            _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            checkpoints.append(dict(ply=ply,publish_ms=elapsed,publish_peak_bytes=peak,
                retained_record_bytes=deep_size(session.record),boundary_bytes=(session.directory/'boundary.json').stat().st_size))
    session._publish=publish
    start=time.perf_counter()
    snapshot=session.run()
    duration=time.perf_counter()-start
    assert snapshot['game']['status']=='truncated' and len(snapshot['game']['positions'])==512
    # Explicit cost of historical assembly vs current append-only publication.
    old_copy=[]
    for length in (32,128,256,512):
        prefix={**snapshot['game'],'positions':snapshot['game']['positions'][:length]}
        raw=[]
        for _ in range(3):
            tracemalloc.start(); t=time.perf_counter(); copied=copy.deepcopy(prefix)
            raw.append((time.perf_counter()-t)*1000)
            _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); del copied
        old_copy.append(dict(plies=length,deepcopy=stats(raw),peak_bytes=peak))
    start=time.perf_counter(); recovered=Session.load(session.directory)
    recovery_s=time.perf_counter()-start
    assert recovered.snapshot()==snapshot
    return dict(plies=512,ending_reason='ply_limit',dummy_evaluator=True,total_run_s=duration,
        windows={str(a)+'-'+str(b):stats(publish_times[a-1:b]) for a,b in ((1,32),(65,128),(257,384),(481,512))},
        checkpoints=checkpoints,positions_list_identity_stable=True,
        historical_deepcopy_cost=old_copy,recovery_audit_s=recovery_s,
        final_snapshot_copies_full_record_once=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/stage1/performance')
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'criterion.json').write_text(json.dumps(CRITERION,indent=2)+'\n')
    print('Criterion fixed:',json.dumps(CRITERION),flush=True)
    data,checkpoint=dataset_benchmark(out)
    print('Dataset:',json.dumps(data),flush=True)
    profile=selfplay_profile(out,checkpoint)
    result=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        environment=dict(python=platform.python_version(),platform=platform.platform(),
            machine=platform.machine(),numpy=np.__version__,torch=str(torch.__version__),chess=chess.__version__),
        criterion=CRITERION,dataset=data,selfplay=profile)
    (out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Self-play:',json.dumps(profile),flush=True)
    assert data['acceptance'],'predeclared dataset scaling criterion failed'


if __name__=='__main__': main()
