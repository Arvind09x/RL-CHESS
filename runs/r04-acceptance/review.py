"""Independent bounded R04 review; all new output belongs to this directory."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import chess
import numpy as np
import torch

ROOT = Path('/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/r04-local-cycle')
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT/'runs/r04'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
os.environ['GIT_OPTIONAL_LOCKS'] = '0'
sys.path.insert(0, str(ROOT/'src'))
from chess_rl.adapter import make_state, encode_position, legal_mask, move_to_id, apply_move
from chess_rl.canonical_games import canonical_payload, game_id
from chess_rl.position_sampling import rule_position
from chess_rl.evaluation import Opening, OpeningSets, Checkpoint, SearchPlayer, SearchSettings, run_match
from chess_rl.replay import ReplayBuffer, finalize
from chess_rl.selfplay import Session, Settings
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import ModelConfig, Progress, load_inference, restore_training, save_checkpoint, _rng_state, runtime_config
from chess_rl.training.mixed_state import MixedSources, MixedSettings, MixedState
from chess_rl.training.mixed import mixed_step

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def read(name):
    return json.loads((EVIDENCE/name).read_text())

def save(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')

def identity(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=h.hexdigest())

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()

def fingerprint(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().contiguous()
        return dict(dtype=str(v.dtype), shape=list(v.shape), sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
    if isinstance(v, dict):
        return {str(k):fingerprint(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)):
        return [fingerprint(x) for x in v]
    return v

def tree(v):
    return digest(fingerprint(v))

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git',*args], cwd=cwd)

P = read('frozen-plan.json')
MEM = read('eligible-membership.json')
CONFIG = ModelConfig(**P['model_config'])
SETTINGS = MixedSettings(**P['mixed_settings'])

def sources():
    c = P['source_config']
    s = MixedSources(open_dataset(**{k:Path(v) for k,v in c['dataset'].items()}, split='train'),
        ReplayBuffer(OUT/'replay.sqlite'), heldout_games_path=c['heldout_games'], protected_positions_path=c['protected_positions'])
    assert s.supervised_source_id == MEM['supervised_source_id']
    assert s.supervised_membership_sha256 == MEM['supervised_membership_sha256']
    assert s.replay_id == MEM['replay_id'] and list(s.replay_membership) == MEM['replay_membership']
    assert list(s.excluded_heldout) == MEM['excluded_heldout']
    return s

def provenance():
    return torch.load(read('new-boundary.json')['path'], weights_only=True, map_location='cpu')['provenance']

def rule_outcome(board):
    result = board.outcome(claim_draw=False)
    if result is not None:
        return result.winner, result.termination.name
    if board.is_repetition(3):
        return None, 'THREEFOLD_REPETITION'
    if board.is_fifty_moves():
        return None, 'FIFTY_MOVES'
    return None

def audit():
    assert git('rev-parse','HEAD').decode().strip() == '01f3c3f' or git('rev-parse','HEAD').decode().startswith('01f3c3f')
    assert git('rev-list','--reverse','2a41e27..HEAD').decode().splitlines() == [git('rev-parse',r).decode().strip() for r in ('cec350b','bbd2e40','01f3c3f')]
    sequence=[]
    for ref, name, modes in [('cec350b','frozen-plan.json',['generate','admit']), ('bbd2e40','eligible-membership.json',['boundary','full','prefix','resume'])]:
        assert git('show',f'{ref}:runs/r04/{name}') == (EVIDENCE/name).read_bytes()
        commit_time = datetime.fromisoformat(git('show','-s','--format=%cI',ref).decode().strip())
        starts=[]
        for mode in modes:
            receipt=read(mode+'.command.json')
            started=datetime.fromisoformat(receipt['started_at_utc'])
            assert started > commit_time and receipt['exit_code'] == 0
            for path, expected in receipt['tested_sources'].items():
                # Retained audit correction is the only intentional source revision.
                if path != 'runs/r04/verify.py':
                    assert identity(ROOT/path) == expected
            starts.append(dict(mode=mode, started=started.isoformat()))
        sequence.append(dict(commit=ref,time=commit_time.isoformat(),execution=starts))
    for path, expected in P['inputs'].items():
        assert identity(path) == expected
    assert runtime_config() == P['runtime']
    summary=read('summary.json')
    for path, expected in summary['evidence_file_identities'].items():
        assert identity(ROOT/path) == expected, path
    for path, expected in summary['external_artifacts'].items():
        assert identity(path) == expected, path
    # Exact accepted source preservation, beyond the supplied preservation claim.
    names=git('ls-tree','-r','--name-only','2a41e27').decode().splitlines()
    for name in names:
        old=git('show','2a41e27:'+name)
        current=(ROOT/name).read_bytes()
        assert current.startswith(old) if name in ('README.md','coding-plan.md') else current==old, name
    before=read('preservation-before.json')
    for path, expected in before['accepted_external'].items():
        assert identity(path)==expected, path
    for name, expected in before['primary']['tracked'].items():
        assert identity(PRIMARY/name)==expected, name
    assert identity(PRIMARY/'.git/index') == before['primary']['raw_index']
    import base64
    assert base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode() == before['primary']['patch_base64']
    assert base64.b64encode(git('diff','--cached','--binary',cwd=PRIMARY)).decode() == before['primary']['staged_patch_base64']
    save('preservation-baseline.json',dict(branch_files={str(p.relative_to(ROOT)):identity(p) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts}))
    shutil.copyfile(EVIDENCE/'replay.sqlite', OUT/'replay.sqlite')
    exported=ReplayBuffer(OUT/'replay.sqlite').export()
    assert digest(exported)==MEM['replay_id'] and digest(exported)==digest(read('replay-export.json'))
    targets={s['sample_id']:s for s in exported['samples']}
    held={json.loads(line)['game_id'] for line in Path(P['source_config']['heldout_games']).read_text().splitlines()}
    protected={json.loads(line)['rule_position'] for line in Path(P['source_config']['protected_positions']).read_text().splitlines()}
    facts=[]
    for v in read('games.json'):
        c,g=v['configuration'],v['game']
        state=make_state(g['initial_fen'],c['opening']['moves_uci'],history_complete=g['history_complete'])
        actual=finalize(v)
        assert state.history_complete and g['moves_uci'][:len(c['opening']['moves_uci'])]==c['opening']['moves_uci']
        setup = g['initial_fen'] != chess.STARTING_FEN
        if setup:
            assert 'declared' in c['opening']['source'] and c['opening']['moves_uci']==[]
            assert state.board.halfmove_clock==99 and len(state.board.move_stack)==0
            assert encode_position(state).available_snapshots==1 and rule_outcome(state.board) is None
            for move in list(state.board.legal_moves):
                child=state.board.copy(stack=True);child.push(move)
                assert rule_outcome(child)==(None,'FIFTY_MOVES')
        for pos in g['positions']:
            assert rule_outcome(state.board) is None
            assert pos['root_fen']==state.board.fen(en_passant='fen') and pos['ply']==len(state.board.move_stack)
            assert pos['legal_mask']==legal_mask(state).tolist()
            visits=np.array(pos['root_visits']);mask=np.array(pos['legal_mask'])
            assert (visits>=0).all() and not visits[~mask].any() and int(visits.sum())==pos['completed_simulations']==8
            assert np.array_equal(visits/8,np.array(pos['visit_policy']))
            assert pos['move_id']==move_to_id(state,chess.Move.from_uci(pos['move_uci'])) and visits[pos['move_id']]>0
            apply_move(state,pos['move_id'])
        outcome=rule_outcome(state.board)
        assert state.board.fen(en_passant='fen')==g['final_fen']
        if g['status']=='truncated':
            assert outcome is None and g['winner'] is None and g['outcome'] is None and actual['samples']==[]
            assert not any(t['source_id']==actual['source_id'] for t in targets.values())
        else:
            assert outcome==(None,'FIFTY_MOVES') and g['ending_reason']==outcome[1]
            for sample in actual['samples']:
                assert targets[sample['sample_id']]==sample and sample['position']==g['positions'][0] and sample['wdl']==[0,1,0]
                root=make_state(sample['initial_fen'],sample['moves_uci'],history_complete=True)
                excluded=game_id(canonical_payload(g['initial_fen'],g['moves_uci'])) in held or rule_position(root.board) in protected
                assert excluded==(sample['sample_id'] in MEM['excluded_heldout'])
                assert (not excluded)==(sample['sample_id'] in MEM['replay_membership'])
        facts.append(dict(opening=c['opening']['opening_id'],setup=setup,status=g['status'],roots=len(g['positions']),targets=len(actual['samples'])))
    s=sources()
    assert len(s.dataset)==12 and len(s.replay_membership)==2
    save('audit.json',dict(freezes=sequence,inputs_rehashed=len(P['inputs']),historical_files_exact=len(names),historical_external_files_exact=len(before['accepted_external']),primary_user_edits_exact=True,evidence_and_checkpoints_rehashed=True,games=facts,membership_matches=True))

def regenerate():
    originals=read('games.json'); actual=[]
    for item in P['generation']['ordinary'] + P['generation']['fallback']['games']:
        setup = item in P['generation']['fallback']['games']
        session=Session.create(OUT/('generated-'+item['opening']['opening_id']),P['checkpoint_path'],Opening(**item['opening']),
            settings=Settings.from_dict(P['generation']['fallback']['settings'] if setup else P['generation']['settings']),
            seed=item['seed'],opening_selection_seed=P['generation']['opening_selection_seed'])
        snapshot=session.run()
        assert snapshot==originals[len(actual)], item['opening']['opening_id']
        actual.append(snapshot)
    buffer=ReplayBuffer.create(OUT/'regenerated-replay.sqlite')
    receipts=[buffer.ingest(v) for v in actual]
    assert receipts==MEM['receipts'] and digest(buffer.export())==MEM['replay_id']
    save('regeneration.json',dict(exact_all_configurations_games_rng_visits_actions=True,exact_replay_and_receipts=True,roots=22,simulations=176))

def boundary():
    s=sources();loaded=load_inference(P['checkpoint_path'],expected_config=CONFIG)
    random.seed(P['global_seed']);np.random.seed(P['global_seed']);torch.manual_seed(P['global_seed'])
    opt=torch.optim.Adam(loaded.model.parameters(),lr=SETTINGS.learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=0.0,amsgrad=False,foreach=False,fused=False)
    state=MixedState.initial(s,SETTINGS,seed=P['sampling_seed'])
    assert len(opt.state)==0 and state.updates==state.supervised.samples_seen==state.selfplay.samples_seen==0
    save_checkpoint(OUT/'fresh-step-0.pt',model=loaded.model,optimizer=opt,checkpoint_id='R04-plumbing-step-0',provenance=provenance(),weights=state.settings.weights,progress=Progress(0,0,0),mixed_state=state)
    new=torch.load(OUT/'fresh-step-0.pt',weights_only=True)
    old=torch.load(read('new-boundary.json')['path'],weights_only=True)
    assert tree(new)==tree(old) and 'data_order' not in new['training']
    save('boundary.json',dict(exact_complete_step_zero_payload=True,new_empty_adam=True,no_legacy_cursor=True,model_matches_T03=tree(loaded.model.state_dict())==P['checkpoint_model_sha256']))

def cross_entropy(logits,mask,targets):
    total=0.
    for row,m,t in zip(logits,mask,targets):
        x=row[m].astype(np.float64);p=t[m].astype(np.float64)
        logz=float(x.max())+math.log(sum(math.exp(float(a-x.max())) for a in x))
        total+=sum(float(a)*(logz-float(b)) for a,b in zip(p,x) if a)
    return total/len(logits)

def train(mode):
    s=sources()
    random.seed(101);np.random.seed(102);torch.manual_seed(103)
    restored=restore_training(OUT/('fresh-step-1.pt' if mode=='resume' else 'fresh-step-0.pt'),expected_config=CONFIG,expected_provenance=provenance(),mixed_sources=s,expected_mixed_settings=SETTINGS)
    model,opt,state=restored.model,restored.optimizer,restored.mixed_state
    assert restored.data_order is None
    captured=[]
    def hook(module,inputs,outputs):
        for tensor in outputs: tensor.retain_grad()
        captured[:]=outputs
    handle=model.register_forward_hook(hook)
    records=[];expected=read('full.json')['steps']
    stop=1 if mode=='prefix' else 3
    while state.updates<stop:
        si,ri,candidate=state.next_selection();sb,rb=s.batches(si,ri)
        metrics,state=mixed_step(model,opt,s,state)
        assert state==candidate
        e=expected[state.updates-1]
        policy,wdl=[t.detach().numpy() for t in captured]
        mask=np.concatenate((sb.legal_mask,rb.legal_mask))
        assert not torch.count_nonzero(captured[0].grad[torch.tensor(~mask)])
        classes=np.eye(4352,dtype=np.float32)[sb.policy_classes]
        losses=[cross_entropy(policy[:2],sb.legal_mask,classes),cross_entropy(wdl[:2],np.ones((2,3),bool),sb.wdl),cross_entropy(policy[2:],rb.legal_mask,rb.policy_distributions),cross_entropy(wdl[2:],np.ones((2,3),bool),rb.wdl)]
        actual=[metrics.supervised.policy_loss,metrics.supervised.wdl_loss,metrics.selfplay.policy_loss,metrics.selfplay.wdl_loss]
        assert max(abs(a-b) for a,b in zip(losses,actual))<2e-6
        observed=dict(metrics=asdict(metrics),supervised_references=sb.references,selfplay_sample_ids=rb.sample_ids,
            batch_sha256=tree([torch.tensor(sb.features),torch.tensor(rb.features)]),logits_sha256=tree(captured),logit_gradients_sha256=tree([t.grad for t in captured]),
            gradients_sha256=tree({name:v.grad for name,v in model.named_parameters()}),model_sha256=tree(model.state_dict()),optimizer_sha256=tree(opt.state_dict()),state_sha256=tree(asdict(state)),rng_sha256=tree(_rng_state()))
        for key,v in observed.items():
            assert digest(v)==digest(e[key]), (mode,state.updates,key)
        assert list(rb.sample_ids)==[s.replay_membership[i] for i in ri]
        assert np.array_equal(rb.policy_distributions,np.array([s._samples[i]['position']['visit_policy'] for i in ri],dtype=np.float32))
        assert np.array_equal(rb.wdl,np.array([s._samples[i]['wdl'] for i in ri],dtype=np.float32))
        observed.update(update=state.updates,independent_loss_max_error=max(abs(a-b) for a,b in zip(losses,actual)))
        records.append(observed)
    handle.remove()
    name='fresh-step-1.pt' if mode=='prefix' else 'fresh-'+mode+'-step-3.pt'
    identifier='R04-plumbing-'+dict(full='full-step-3',prefix='step-1',resume='updated-step-3')[mode]
    save_checkpoint(OUT/name,model=model,optimizer=opt,checkpoint_id=identifier,provenance=provenance(),weights=state.settings.weights,progress=Progress(state.updates,0,state.updates*4),mixed_state=state)
    payload=torch.load(OUT/name,weights_only=True)
    original=torch.load(read(mode+'.json')['checkpoint']['path'],weights_only=True)
    assert tree(payload)==tree(original)
    before=tree(_rng_state()); inference=load_inference(OUT/name,expected_config=CONFIG)
    assert tree(inference.model.state_dict())==tree(model.state_dict()) and tree(_rng_state())==before
    save(mode+'.json',dict(pid=os.getpid(),steps=records,exact_complete_saved_payload=True,inference_exact_rng_preserved=True,checkpoint=identity(OUT/name)))

def evaluate():
    # Preserve original byte-bound identities while independently rerunning E01.
    players=[]
    paths=[P['checkpoint_path'],read('resume.json')['checkpoint']['path']]
    for label,role,path in zip(('r04-starting-T03','r04-updated'),('P','R'),paths):
        evaluator=RealEvaluator(path,expected_config=CONFIG)
        players.append(SearchPlayer(label,role,evaluator.checkpoint.checkpoint_id,evaluator,SearchSettings(simulation_budget=8),Checkpoint.from_file(evaluator.checkpoint.checkpoint_id,path,kind='fixture')))
    book=OpeningSets('r04-diagnostic-book',tuple(Opening(**v) for v in P['evaluation']['validation']),tuple(Opening(**v) for v in P['evaluation']['test']))
    record=run_match(*players,book,split='validation',match_id=P['evaluation']['match_id'],comparison='P-versus-R',seed=P['evaluation']['seed'],ply_cap=8,protocol_only=True).to_dict()
    original=json.loads((EVIDENCE/'matches.jsonl').read_text())
    assert record==original
    groups={};complete=0;plies=0;sims=0
    for game in record['games']:
        groups.setdefault(game['pair_id'],[]).append(game)
        state=make_state(game['initial_fen'],game['moves_uci'],history_complete=True)
        terminal=rule_outcome(state.board)
        assert (terminal is not None)==(game['status']=='completed')
        if terminal is not None: complete+=1
        else: assert game['outcome'] is None and game['winner'] is None
        for attempt in game['attempts']:
            assert attempt['status']=='completed' and attempt['applied']
            plies+=1;sims+=attempt['details']['completed_simulations']
    complete_pairs=0
    for games in groups.values():
        assert {g['leg'] for g in games}=={0,1} and len(games)==2
        a,b=sorted(games,key=lambda g:g['leg'])
        assert a['opening']==b['opening']
        assert a['players']['white']==b['players']['black'] and a['players']['black']==b['players']['white']
        complete_pairs+=all(g['status']=='completed' for g in games)
    assert complete==2 and complete_pairs==1 and plies==18 and sims==144
    assert record['summary']['completed_game_denominator']==complete
    assert record['summary']['complete_pairs']==complete_pairs and record['summary']['paired_uncertainty'] is None
    assert record['summary']['completed_game_score']==record['summary']['complete_pair_score']==.5
    save('evaluation.json',dict(exact_full_match=True,completed_denominator=complete,complete_pairs=complete_pairs,ordinary_truncated=2,fixture_draws=2,generated_plies=plies,simulations=sims,protocol_only=True,no_learning_strength_or_P_claim=True))

def reload():
    s=sources(); results=[]
    for mode in ('boundary','prefix','full','resume'):
        if mode=='boundary':
            original=read('new-boundary.json')['path']; fresh=OUT/'fresh-step-0.pt'
        else:
            original=read(mode+'.json')['checkpoint']['path']
            fresh=OUT/('fresh-step-1.pt' if mode=='prefix' else 'fresh-'+mode+'-step-3.pt')
        for path in (Path(original),fresh):
            payload=torch.load(path,weights_only=True,map_location='cpu')
            random.seed(301);np.random.seed(302);torch.manual_seed(303)
            rng=tree(_rng_state());loaded=load_inference(path,expected_config=CONFIG)
            assert tree(loaded.model.state_dict())==tree(payload['model_state']) and tree(_rng_state())==rng
            restored=restore_training(path,expected_config=CONFIG,expected_provenance=provenance(),mixed_sources=s,expected_mixed_settings=SETTINGS)
            assert restored.data_order is None and restored.model.training==payload['model_training']
            assert tree(restored.model.state_dict())==tree(payload['model_state'])
            assert tree(asdict(restored.mixed_state))==tree(payload['training']['mixed_state'])
            assert asdict(restored.progress)==payload['training']['progress'] and tree(_rng_state())==tree(payload['training']['rng'])
            expected=read('new-boundary.json') if mode=='boundary' else read(mode+'.json')['final']
            assert tree(restored.optimizer.state_dict())==expected['optimizer_sha256']
            results.append(dict(mode=mode,path=str(path),complete_training_state_exact=True,inference_rng_unchanged=True))
    save('reload.json',dict(pid=os.getpid(),checkpoints=results))

def final():
    baseline=json.loads((OUT/'preservation-baseline.json').read_text())
    for path,expected in baseline['branch_files'].items():
        assert identity(ROOT/path)==expected, path
    assert not git('status','--porcelain').strip()
    for mode in ('full','prefix','resume'):
        assert json.loads((OUT/(mode+'.json')).read_text())['exact_complete_saved_payload']
    assert len(json.loads((OUT/'reload.json').read_text())['checkpoints'])==8
    before=read('preservation-before.json')
    for path, expected in before['accepted_external'].items(): assert identity(path)==expected, path
    for path, expected in read('summary.json')['external_artifacts'].items(): assert identity(path)==expected, path
    for name, expected in before['primary']['tracked'].items(): assert identity(PRIMARY/name)==expected, name
    assert identity(PRIMARY/'.git/index')==before['primary']['raw_index']
    import xml.etree.ElementTree as ET
    tests={}
    for name in ('model','adapter'):
        suite=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
        tests[name]={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
        assert tests[name]['failures']==tests[name]['errors']==0
    save('summary.json',dict(verdict='ACCEPT bounded CPU Gate I',actionable_findings=[],all_requested_checks_passed=True,
        reviewed_head=git('rev-parse','HEAD').decode().strip(),accepted_base=git('rev-parse','2a41e27').decode().strip(),
        complete_sequence=['cec350b','bbd2e40','01f3c3f'],
        checks=[
            dict(check='Both freezes precede their governed execution',passed=True,source='scripts/run_r04.py',line=141,evidence='audit.json',detail='Plan 15:37:29 IST before generation 15:37:39; membership 15:38:34 before boundary 15:38:45 and optimization 15:39:37. Frozen bytes match both original commits and HEAD.'),
            dict(check='Ordinary truncations unlabeled; all 20 roots excluded',passed=True,source='runs/r04/eligible-membership.json',line=3,evidence='audit.json',detail='12 standard-start roots and 8 opening-prefix roots produce no replay targets, winner or outcome.'),
            dict(check='Declared setup history and real search',passed=True,source='scripts/run_r04.py',line=84,evidence='regeneration.json',detail='Actual declared initial FEN, no preceding moves, complete history with one real snapshot. Native chess checks verify valid nonterminal roots and every legal child fifty-move draw. All generated configurations, roots, visits, moves and RNG exactly reproduce.'),
            dict(check='Completed roots preserve visits, distributions, exact WDL',passed=True,source='src/chess_rl/replay.py',line=95,evidence='audit.json',detail='Both roots retain 8 visits, multi-action distributions (4 and 3 supported actions) and exact [0,1,0]. Fresh replay export and admission receipts match the original.'),
            dict(check='Held-out filtering and frozen membership before optimization',passed=True,source='src/chess_rl/training/mixed_state.py',line=201,evidence='audit.json',detail='Independent canonical whole-game and native rule-position filtering confirms two eligible roots, no held-out overlaps; reopened 12 training labels and membership hashes match the immutable freeze.'),
            dict(check='T03 weights initialize new Adam and mixed state without legacy cursor',passed=True,source='scripts/run_r04.py',line=239,evidence='boundary.json',detail='Independently rebuilt complete step-zero payload equals the original: exact T03 model, empty Adam, zero mixed source progress; no legacy data_order field.'),
            dict(check='Actual roots feed recorded mixed updates',passed=True,source='scripts/run_r04.py',line=275,evidence='full.json',detail='All three updates consume both eligible sample IDs; batches retain the actual visit policies and outcome WDL. References, metrics, features, logits, gradients, model, Adam, sampling state and global RNG reproduce exactly; independent CE agrees within 2e-6.'),
            dict(check='Checkpoint reload and fresh-process continuation preserve complete state',passed=True,source='scripts/run_r04.py',line=259,evidence='reload.json',detail='Full/prefix/resume run in distinct processes with scrambled RNG and reproduce every recorded step and complete saved payload. All four original and four fresh checkpoints pass inference and complete training restoration: mode, model, Adam, both source cursors/permutations/epochs/counts, sampling RNG, global RNG and progress.'),
            dict(check='E01 pairing, settings and denominators',passed=True,source='scripts/run_r04.py',line=312,evidence='evaluation.json',detail='Fresh four-game match equals the entire original record. Identical openings and settings, reversed player colors, no noise; 18 plies and 144 simulations. Two ordinary truncations excluded from score; 2 completed fixture draws, denominator 2, 1 complete pair, score 0.5, no interval.'),
            dict(check='No learning, strength or pretrained-P claim',passed=True,source='docs/r04-local-cycle.md',line=118,evidence='evaluation.json',detail='Both checkpoints kind=fixture, protocol_only=true, reserved test opening unused. Documentation explicitly limits the results to plumbing and makes no learning, strength, improvement or pretrained-P assertion.')
        ],
        scope_limitations=['Deterministic pinned CPU float32 only','Ordinary games have zero completed outcomes','Completed replay targets and completed evaluation outcomes are setup-fixture draws only','No evidence of learning or strength; no P selected; no pilot or integration authorized by this verdict'],
        retained_review_harness_failure=dict(command='boundary.command.json',reason='Review omitted explicit weight_decay=0.0; strict mixed optimizer schema rejected PyTorch default integer 0 before save or optimization',correction='Exact frozen Adam options; successful boundary-retry.command.json; candidate sources unchanged'),
        branch_and_historical_evidence_unchanged=True,pilot_started=False,integration_performed=False,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        regressions=tests,
        review_file_identities={p.name:identity(p) for p in OUT.iterdir() if p.is_file()}))

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('full','prefix','resume'): train(mode)
    else: globals()[mode]()
    print(mode+' passed',flush=True)
