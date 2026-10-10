"""Reopen accepted sources; independently audit exclusions and sampler."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import torch
from checks import ROOT,OUT,PRIMARY,EXTERNAL,identity,save
from scripts.run_r03 import default_source_config,tree_hash
from chess_rl.adapter import make_state,legal_mask
from chess_rl.canonical_games import canonical_payload,game_id
from chess_rl.position_sampling import rule_position
from chess_rl.replay import ReplayBuffer,finalize
from chess_rl.supervised import open_dataset
from chess_rl.training.mixed_state import MixedSources,MixedState,MixedSettings
from chess_rl.training.checkpoints import ModelConfig,load_inference,_rng_state

torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
plan=json.loads((ROOT/'runs/r03/frozen-plan.json').read_text())
for n,expected in plan['inputs'].items(): assert identity(n)==expected,n
config=default_source_config()
dataset=open_dataset(**{k:Path(v) for k,v in config['dataset'].items()},split='train')
buffer=ReplayBuffer(config['replay']); export=buffer.export()
sources=MixedSources(dataset,buffer,heldout_games_path=config['heldout_games'],
    protected_positions_path=config['protected_positions'])
historical=json.loads((ROOT/'runs/r03/summary.json').read_text())['sources']
for field in ('supervised_source_id','supervised_membership_sha256','replay_id'):
    assert getattr(sources,field)==historical[field],field
assert list(sources.replay_membership)==historical['replay_membership']==plan['replay']['sample_ids']
held_games={json.loads(line)['game_id'] for line in Path(config['heldout_games']).read_text().splitlines()}
positions={json.loads(line)['rule_position'] for line in Path(config['protected_positions']).read_text().splitlines()}
blocked=set(); unfinished=0
for receipt in export['sources']:
    result=finalize(receipt['snapshot'])
    if result['status']!='completed':
        assert result['samples']==[]; unfinished+=1
    elif game_id(canonical_payload(result['snapshot']['game']['initial_fen'],result['snapshot']['game']['moves_uci'])) in held_games:
        blocked.add(result['game_id'])
excluded=[]; eligible=[]
for sample in export['samples']:
    state=make_state(sample['initial_fen'],sample['moves_uci'],history_complete=True)
    assert sample['position']['legal_mask']==legal_mask(state).tolist()
    assert sample['wdl'] in ([1,0,0],[0,1,0],[0,0,1])
    assert sample['game_id'] not in {s['game_id'] for s in export['sources'] if s['snapshot']['game']['status']!='completed'} or any(s['game_id']==sample['game_id'] and s['disposition']=='admitted' for s in export['sources'])
    (excluded if sample['game_id'] in blocked or rule_position(state.board) in positions else eligible).append(sample['sample_id'])
assert tuple(excluded)==sources.excluded_heldout and tuple(eligible)==sources.replay_membership
assert len(dataset)==12 and len(eligible)==4
for sample in dataset:
    assert sample.split=='train'
settings=MixedSettings(); state=MixedState.initial(sources,settings,seed=plan['seed'])
rng=random.Random(plan['seed']); orders=[list(range(len(dataset))),list(range(len(eligible)))]
for order in orders: rng.shuffle(order)
cursors=[0,0]; epochs=[0,0]; seen=[0,0]
assert tuple(orders[0])==state.supervised.order and tuple(orders[1])==state.selfplay.order
for update in range(1,102):
    expected=[]
    for k in range(2):
        draw=[]
        for _ in range(2):
            if cursors[k]==len(orders[k]):
                orders[k]=list(range(len(orders[k]))); rng.shuffle(orders[k]); cursors[k]=0; epochs[k]+=1
            draw.append(orders[k][cursors[k]]); cursors[k]+=1; seen[k]+=1
        expected.append(tuple(draw))
    si,ri,state=state.next_selection()
    assert (si,ri)==tuple(expected) and state.sampling_rng==rng.getstate()
    for k,cursor in enumerate((state.supervised,state.selfplay)):
        assert asdict(cursor)==dict(order=tuple(orders[k]),cursor=cursors[k],epoch=epochs[k],samples_seen=seen[k])
    if update==3:
        actual=json.loads((OUT/'full.json').read_text())['final']['mixed_state']
        assert json.loads(json.dumps(asdict(state)))==actual
legacy=[]
before=json.loads((OUT/'preservation-before.json').read_text())
for n in before['external']:
    if Path(n).suffix!='.pt': continue
    payload=torch.load(n,map_location='cpu',weights_only=False)
    if payload.get('version')!='m06-checkpoint-v1': continue
    rng_before=tree_hash(_rng_state()); loaded=load_inference(n,expected_config=ModelConfig(**payload['model_config']))
    assert rng_before==tree_hash(_rng_state())
    legacy.append(dict(path=n,identity=identity(n),checkpoint_id=loaded.checkpoint_id,version=payload['version'],inference_rng_exact=True))
assert len(legacy)>=3
save(OUT/'source-audit.json',dict(frozen_inputs_verified=len(plan['inputs']),
    source_config=config,supervised_source_identity=dataset.source_identity,
    supervised_source_id=sources.supervised_source_id,supervised_membership_sha256=sources.supervised_membership_sha256,
    supervised_count=len(dataset),replay_id=sources.replay_id,replay_membership=list(sources.replay_membership),
    receipt_counts=export['summary'],unfinished_receipts_unlabeled=unfinished,
    heldout_games=len(held_games),protected_rule_positions=len(positions),blocked_replay_games=sorted(blocked),
    excluded_heldout=excluded,actual_heldout_overlaps_zero=True,
    sampler_updates_independently_reproduced=101,permutations_cursors_epochs_counts_and_rng_exact=True,
    actual_three_update_sampling_state_exact=True,legacy_historical_checkpoints=legacy))
print(json.dumps(dict(train_labels=len(dataset),eligible_replay_roots=len(eligible),unfinished_receipts=unfinished,
    heldout_overlaps=0,sampling_updates=101,legacy_checkpoints=len(legacy))))
