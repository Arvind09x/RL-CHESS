"""R04 bounded CPU integration; append-only evidence, no pilot or P selection.

freeze -> generate -> admit -> boundary -> full/prefix/resume -> evaluate.
Commit the plan before generate and eligible membership before boundary.
All chess, search, targets, updates and persistence use accepted shared APIs.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess

import chess
import numpy as np
import torch

from chess_rl.adapter import apply_move, encode_position, make_state, move_to_id, terminal_result
from chess_rl.evaluation import Checkpoint, Opening, OpeningSets, SearchPlayer, SearchSettings, run_match
from chess_rl.evaluation.matches import digest
from chess_rl.evaluation.replay import audit_match
from chess_rl.pgn_replay import file_identity
from chess_rl.replay import ReplayBuffer, finalize, read_source
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.selfplay import Session, Settings, summarize
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (ModelConfig, Progress, _rng_state, load_inference,
    restore_training, runtime_config, save_checkpoint)
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedSources
from run_r03 import default_source_config, oracle, tree_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/r04'
EXTERNAL = ROOT.parent / 'external-artifacts/r04-local-cycle-20261010'
BASE = '2a41e27'  # Full accepted identity is frozen from git below.


def save(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def read(name):
    return json.loads((OUT / name).read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def opening(value):
    return Opening(**value)


def op(name, fen=chess.STARTING_FEN, moves=(), source='R04 declared diagnostic ordinary opening'):
    return asdict(Opening(name, fen, moves, True, source))


def freeze():
    assert git('branch', '--show-current') == 'codex/r04-local-cycle'
    assert git('rev-parse', 'HEAD').startswith(BASE)
    assert not EXTERNAL.exists(), 'new external namespace required'
    historical = json.loads((ROOT / 'runs/r01/real/frozen-plan.json').read_text())['checkpoint']
    checkpoint = Path(historical['path'])
    assert file_identity(checkpoint) == {k: historical[k] for k in ('bytes', 'sha256')}
    loaded = load_inference(checkpoint)
    assert loaded.checkpoint_id == 'T03-plumbing-step-26'
    assert asdict(loaded.config) == historical['model_config']
    source = default_source_config()
    source['replay'] = str(OUT / 'replay.sqlite')
    inputs = {str(checkpoint): file_identity(checkpoint)}
    prior = json.loads((ROOT / 'runs/r03/frozen-plan.json').read_text())
    # Rebind unchanged accepted inputs to this checkout, excluding old R02 replay.
    for name, expected in prior['inputs'].items():
        if name.endswith('fresh-replay.sqlite'):
            continue
        path = Path(name)
        if '/r03-mixed-update/' in name:
            path = ROOT / name.split('/r03-mixed-update/', 1)[1]
        assert file_identity(path) == expected, f'accepted source drift: {path}'
        inputs[str(path)] = expected
    near = [op('r04-fifty-white', '4k3/8/8/8/8/8/8/R3K3 w - - 99 1',
               source='R04 separately declared setup game; actual initial FEN, no prior moves'),
            op('r04-fifty-black', 'r3k3/8/8/8/8/8/8/4K3 b - - 99 1',
               source='R04 separately declared color counterpart setup game; actual initial FEN, no prior moves')]
    # Prove validity/nonterminal and every legal continuation completes, without inference/search.
    validations = []
    for item in near:
        state = opening(item).state()
        assert state.board.is_valid() and terminal_result(state) is None
        reasons = []
        for move in list(state.board.legal_moves):
            child = state.copy()
            apply_move(child, move_to_id(child, move))
            result = terminal_result(child)
            assert result is not None and result.winner is None
            reasons.append(result.ending_reason)
        validations.append(dict(opening_id=item['opening_id'], legal_moves=len(reasons),
            all_legal_children_terminal_draws=True, reasons=sorted(set(reasons)),
            complete_history_provenance='game genuinely begins at declared setup; no invented earlier history'))
    settings = asdict(Settings(simulation_budget=8, ply_cap=12))
    plan = dict(schema='r04-frozen-cpu-plan-v1', task='R04 only; Gate I independent review pending',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'), checkpoint_path=str(checkpoint),
        checkpoint_identity=file_identity(checkpoint), checkpoint_id=loaded.checkpoint_id,
        model_config=asdict(loaded.config), checkpoint_model_sha256=tree_hash(loaded.model.state_dict()),
        inputs=inputs, source_config=source, runtime=runtime_config(),
        generation=dict(settings=settings, opening_selection_seed=2026101004,
            ordinary=[dict(opening=op('r04-start'),seed=40401),
                      dict(opening=op('r04-open-game',moves=('e2e4','e7e5','g1f3','b8c6')),seed=40402)],
            fallback=dict(condition='all ordinary games truncated; otherwise never execute fallback',
                settings=asdict(Settings(simulation_budget=8,ply_cap=1)),
                games=[dict(opening=near[0],seed=40403),dict(opening=near[1],seed=40404)],
                validation=validations),
            maximum_games=4, maximum_generated_roots=22, maximum_simulations=176),
        mixed_settings=asdict(MixedSettings()), sampling_seed=2026101004, global_seed=2026101004,
        optimizer=dict(kind='new standard Adam',lr=1e-5,betas=[.9,.999],eps=1e-8,
            weight_decay=0.0,amsgrad=False,foreach=False,fused=False),
        new_boundary='load T03 inference weights only; new empty Adam; MixedState.initial at zero; save v2 mixed step zero',
        update_plan=dict(logical_updates=3, verification_optimizer_executions=6,
            paths='full 3; prefix 1; fresh-process resume 2; all start from the saved new mixed step-zero boundary',
            schedule='one update per 2 supervised + 2 eligible selfplay samples; constant lr',
            replay_membership='freeze after R02 audit + R03 held-out filtering, before any optimizer execution',
            ratio_per_consumed_mixed_sample=.25,ratio_per_consumed_selfplay_sample=.5,
            replay_unique_ratio='report after eligible membership is known; no pilot extrapolation'),
        evaluation=dict(match_id='r04-starting-versus-updated',seed=40405,simulation_budget=8,ply_cap=8,
            validation=[op('r04-eval-start'),op('r04-eval-fifty', '4k3/8/8/8/8/8/8/1R2K3 w - - 99 1',source='R04 separate evaluation setup fixture; actual initial FEN, no prior moves')],
            test=[op('r04-unused-protocol-test',moves=('c2c4',))],
            planned_pairs=2,planned_games=4,maximum_generated_plies=18,maximum_simulations=144,
            split='validation',protocol_only=True,
            meaning='actual starting T03 versus updated R04 fixture checkpoints; E01 P/R slots are protocol names only; no P selection',
            search='identical budget 8, c_puct 1.5, unvisited_value 0; fresh roots, no noise, max visits/smallest ID',
            scoring='accepted E01 completed-games and complete-pairs only; report all attrition; no strength claim'),
        external_directory=str(EXTERNAL),scope='no accepted-source edits, integration, CUDA, pilot spending, pretrained P selection or strength claim')
    save('frozen-plan.json',plan)
    print(json.dumps(dict(plan_sha256=digest(plan),checkpoint=plan['checkpoint_identity'],fixture_validation=validations)))


def plan():
    value = read('frozen-plan.json')
    for name, expected in value['inputs'].items():
        assert file_identity(Path(name)) == expected, f'frozen input drift: {name}'
    assert runtime_config() == value['runtime']
    # Require the immutable plan to be committed before any dependent execution.
    assert subprocess.check_output(['git','show','HEAD:runs/r04/frozen-plan.json'],cwd=ROOT) == (OUT/'frozen-plan.json').read_bytes()
    return value


def generate(p):
    results = []
    def play(item, settings):
        session = Session.create(OUT / item['opening']['opening_id'], p['checkpoint_path'],
            opening(item['opening']),settings=Settings.from_dict(settings),seed=item['seed'],
            opening_selection_seed=p['generation']['opening_selection_seed'])
        result = session.run()
        assert read_source(session.directory) == result
        results.append(result)
    for item in p['generation']['ordinary']:
        play(item,p['generation']['settings'])
    ordinary_summary = summarize(results)
    fallback_used = all(v['game']['status'] == 'truncated' for v in results)
    if fallback_used:
        for item in p['generation']['fallback']['games']:
            play(item,p['generation']['fallback']['settings'])
    assert len(results) <= p['generation']['maximum_games']
    roots = sum(len(v['game']['positions']) for v in results)
    assert roots <= p['generation']['maximum_generated_roots']
    save('games.json',results)
    save('generation.json',dict(ordinary=ordinary_summary,all_games=summarize(results),
        fallback_used=fallback_used,generated_roots=roots,completed_simulations=roots*8,
        games_sha256=digest(results),diagnosis='short ordinary cap tests truncation safely; unfinished roots excluded, never relabeled'))
    print(json.dumps(read('generation.json')))


def sources(p):
    config = p['source_config']
    dataset = open_dataset(**{k:Path(v) for k,v in config['dataset'].items()},split='train')
    return MixedSources(dataset,ReplayBuffer(config['replay']),heldout_games_path=config['heldout_games'],
        protected_positions_path=config['protected_positions'])


def membership(s):
    return dict(supervised_source_id=s.supervised_source_id,
        supervised_membership_sha256=s.supervised_membership_sha256,supervised_count=len(s.dataset),
        replay_id=s.replay_id,replay_membership=list(s.replay_membership),excluded_heldout=list(s.excluded_heldout))


def admit(p):
    buffer = ReplayBuffer.create(p['source_config']['replay'])
    games = read('games.json')
    receipts = [buffer.ingest(v) for v in games]
    export = buffer.export()
    # Independent root->target identity and exact final-outcome audit.
    finalized = [finalize(v) for v in games]
    save('replay-export.json',export)
    s = sources(p)
    save('eligible-membership.json',dict(**membership(s), plan_sha256=digest(p),
        replay_summary=export['summary'],receipts=receipts,
        finalized_summary=[dict(status=v['game']['status'],generated_roots=len(v['game']['positions']),
            outcome=v['game']['outcome'],finalized_targets=len(f['samples'])) for v,f in zip(games,finalized)],
        updates_per_unique_eligible_root=3/len(s.replay_membership),
        freeze='owned R03 membership; no replay admission, eviction or replacement during this cycle'))
    with (OUT/'targets.jsonl').open('x') as stream:
        for sample in export['samples']:
            stream.write(json.dumps(sample,sort_keys=True)+'\n')
    print(json.dumps(dict(replay=export['summary'],eligible=len(s.replay_membership),excluded=len(s.excluded_heldout))))


def checked_sources(p):
    frozen = read('eligible-membership.json')
    assert subprocess.check_output(['git','show','HEAD:runs/r04/eligible-membership.json'],cwd=ROOT) == (OUT/'eligible-membership.json').read_bytes()
    s = sources(p)
    assert membership(s) == {k:frozen[k] for k in membership(s)}
    return s


def provenance(p,s):
    return dict(task='R04 local plumbing only; never pretrained P',plan_sha256=digest(p),
        starting_checkpoint=p['checkpoint_identity'],starting_checkpoint_id=p['checkpoint_id'],
        boundary='new Adam and mixed state at zero; no supervised optimizer/cursor restore',
        membership_sha256=digest(read('eligible-membership.json')),sources=membership(s))


def checkpoint(p,model,opt,state,name):
    path = Path(p['external_directory']) / (name+'.pt')
    before = tree_hash(_rng_state())
    save_checkpoint(path,model=model,optimizer=opt,checkpoint_id='R04-plumbing-'+name,
        provenance=provenance(p,checked_sources(p)),weights=state.settings.weights,
        progress=Progress(state.updates,0,state.updates*state.settings.batch_size),mixed_state=state)
    loaded = load_inference(path,expected_config=ModelConfig(**p['model_config']))
    assert tree_hash(loaded.model.state_dict()) == tree_hash(model.state_dict())
    assert before == tree_hash(_rng_state())
    return dict(path=str(path),identity=file_identity(path),checkpoint_id=loaded.checkpoint_id,
        inference_model_exact=True,inference_and_save_rng_unchanged=True)


def boundary(p):
    from chess_rl.training.mixed_state import MixedState
    s = checked_sources(p)
    loaded = load_inference(p['checkpoint_path'],expected_config=ModelConfig(**p['model_config']))
    random.seed(p['global_seed']); np.random.seed(p['global_seed']); torch.manual_seed(p['global_seed'])
    model = loaded.model
    settings = MixedSettings(**p['mixed_settings'])
    opt = torch.optim.Adam(model.parameters(),lr=settings.learning_rate,betas=(.9,.999),eps=1e-8,
        weight_decay=0.0,amsgrad=False,foreach=False,fused=False)
    state = MixedState.initial(s,settings,seed=p['sampling_seed'])
    assert len(opt.state) == 0 and state.updates == 0
    assert tree_hash(model.state_dict()) == p['checkpoint_model_sha256']
    Path(p['external_directory']).mkdir(parents=True,exist_ok=False)
    saved = checkpoint(p,model,opt,state,'step-0')
    save('new-boundary.json',dict(**saved,model_sha256=tree_hash(model.state_dict()),
        optimizer_sha256=tree_hash(opt.state_dict()),mixed_state_sha256=tree_hash(asdict(state)),
        optimizer_entries=0,progress=asdict(Progress(0,0,0)),old_optimizer_and_cursor_restored=False))
    print(json.dumps(saved))


def train(p,mode):
    s = checked_sources(p)
    input_name = 'step-1' if mode == 'resume' else 'step-0'
    input_path = Path(p['external_directory'])/(input_name+'.pt')
    random.seed(9); np.random.seed(8); torch.manual_seed(7)
    restored = restore_training(input_path,expected_config=ModelConfig(**p['model_config']),
        expected_provenance=provenance(p,s),mixed_sources=s,
        expected_mixed_settings=MixedSettings(**p['mixed_settings']))
    model,opt,state = restored.model,restored.optimizer,restored.mixed_state
    assert restored.data_order is None and state.updates == (1 if mode == 'resume' else 0)
    records,captured = [],[]
    def capture(module,inputs,output):
        for tensor in output: tensor.retain_grad()
        captured[:] = output
    hook = model.register_forward_hook(capture)
    stop = 1 if mode == 'prefix' else p['update_plan']['logical_updates']
    while state.updates < stop:
        si,ri,candidate = state.next_selection()
        sb,rb = s.batches(si,ri)
        metrics,state = mixed_step(model,opt,s,state)
        assert state == candidate
        policy,wdl = [v.detach().numpy() for v in captured]
        parts = [oracle(policy[:2],sb.legal_mask,classes=sb.policy_classes),
            oracle(wdl[:2],np.ones((2,3),bool),targets=sb.wdl),
            oracle(policy[2:],rb.legal_mask,targets=rb.policy_distributions),
            oracle(wdl[2:],np.ones((2,3),bool),targets=rb.wdl)]
        actual = [metrics.supervised.policy_loss,metrics.supervised.wdl_loss,
            metrics.selfplay.policy_loss,metrics.selfplay.wdl_loss]
        assert all(abs(a-b)<2e-6 for a,b in zip(parts,actual))
        assert abs(.5*sum(parts)-metrics.total_loss)<2e-6
        gradient = {name:v.grad for name,v in model.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in gradient.values())
        mask = torch.tensor(np.concatenate((sb.legal_mask,rb.legal_mask)))
        assert not torch.count_nonzero(captured[0].grad[~mask])
        records.append(dict(update=state.updates,metrics=asdict(metrics),independent_losses=parts,
            supervised_references=sb.references,selfplay_sample_ids=rb.sample_ids,
            teacher_classes=sb.policy_classes.tolist(),soft_teacher_wdl=sb.wdl.tolist(),
            final_selfplay_wdl=rb.wdl.tolist(),visit_nonzero_counts=np.count_nonzero(rb.policy_distributions,axis=1).tolist(),
            batch_sha256=tree_hash([torch.tensor(sb.features),torch.tensor(rb.features)]),
            logits_sha256=tree_hash(captured),logit_gradients_sha256=tree_hash([v.grad for v in captured]),
            gradients_sha256=tree_hash(gradient),finite_parameter_gradients=len(gradient),
            model_sha256=tree_hash(model.state_dict()),optimizer_sha256=tree_hash(opt.state_dict()),
            state_sha256=tree_hash(asdict(state)),rng_sha256=tree_hash(_rng_state())))
    hook.remove()
    final = dict(model_sha256=tree_hash(model.state_dict()),optimizer_sha256=tree_hash(opt.state_dict()),
        gradients_sha256=tree_hash({name:v.grad for name,v in model.named_parameters()}),
        mixed_state=asdict(state),rng_sha256=tree_hash(_rng_state()),progress=asdict(Progress(state.updates,0,state.updates*4)))
    saved = checkpoint(p,model,opt,state,dict(full='full-step-3',prefix='step-1',resume='updated-step-3')[mode])
    save(mode+'.json',dict(mode=mode,restored_from=dict(path=str(input_path),identity=file_identity(input_path)),
        sources=membership(s),steps=records,final=final,checkpoint=saved))
    print(json.dumps(dict(mode=mode,losses=[v['metrics']['total_loss'] for v in records],checkpoint=saved)))


def evaluate(p):
    full,prefix,resume = [read(v+'.json') for v in ('full','prefix','resume')]
    assert full['steps'] == prefix['steps']+resume['steps']
    assert full['final'] == resume['final']
    assert full['final']['model_sha256'] != p['checkpoint_model_sha256']
    start,updated = p['checkpoint_path'],resume['checkpoint']['path']
    config = ModelConfig(**p['model_config'])
    evaluators = [RealEvaluator(path,expected_config=config) for path in (start,updated)]
    settings = SearchSettings(simulation_budget=p['evaluation']['simulation_budget'])
    players = [SearchPlayer(label,role,evaluator.checkpoint.checkpoint_id,evaluator,settings,
        Checkpoint.from_file(evaluator.checkpoint.checkpoint_id,path,kind='fixture'))
        for label,role,evaluator,path in zip(('r04-starting-T03','r04-updated'),('P','R'),evaluators,(start,updated))]
    book = OpeningSets('r04-diagnostic-book',tuple(opening(v) for v in p['evaluation']['validation']),
        tuple(opening(v) for v in p['evaluation']['test']))
    record = run_match(*players,book,split='validation',match_id=p['evaluation']['match_id'],
        comparison='P-versus-R',seed=p['evaluation']['seed'],ply_cap=p['evaluation']['ply_cap'],protocol_only=True).to_dict()
    audit = audit_match(record)
    assert record['configuration']['left_player']['settings'] == record['configuration']['right_player']['settings']
    assert audit['generated_plies'] <= p['evaluation']['maximum_generated_plies']
    assert audit['completed_search_simulations'] <= p['evaluation']['maximum_simulations']
    with (OUT/'matches.jsonl').open('x') as stream: stream.write(json.dumps(record,sort_keys=True)+'\n')
    save('evaluation.json',dict(summary=record['summary'],audit=audit,continuation_exact=True,
        complete_steps_exact=[2,3],complete_final_state_exact=True,
        meaning=p['evaluation']['meaning'],strength_claim=False,pretrained_P_selected=False))
    print(json.dumps(read('evaluation.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','generate','admit','boundary','full','prefix','resume','evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if args.mode == 'freeze': freeze(); return
    p = plan()
    if args.mode in ('full','prefix','resume'): train(p,args.mode)
    else: globals()[args.mode](p)


if __name__ == '__main__': main()
