"""Append-only R04 command receipts and independent end-to-end/preservation audit."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/r04'
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
PYTHON = PRIMARY/'.venv/bin/python'
ADAPTER = PRIMARY/'.venv-acceptance/bin/python'


def save(name,value):
    with (OUT/name).open('x') as stream: stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')


def identity(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): sha.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=sha.hexdigest())


def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd)


def primary_snapshot():
    files = git('ls-files','-z',cwd=PRIMARY).decode().split('\0')[:-1]
    return dict(head=git('rev-parse','HEAD',cwd=PRIMARY).decode().strip(),
        branch=git('branch','--show-current',cwd=PRIMARY).decode().strip(),
        status=git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode(),
        patch_base64=base64.b64encode(git('diff','--binary',cwd=PRIMARY)).decode(),
        staged_patch_base64=base64.b64encode(git('diff','--cached','--binary',cwd=PRIMARY)).decode(),
        raw_index=identity(PRIMARY/'.git/index'),
        tracked={name:identity(PRIMARY/name) for name in files})


def prepare():
    files = git('ls-files','-z').decode().split('\0')[:-1]
    external = ROOT.parent/'external-artifacts'
    save('preservation-before.json',dict(primary=primary_snapshot(),
        accepted_checkout={name:identity(ROOT/name) for name in files},
        accepted_external={str(p):identity(p) for p in sorted(external.rglob('*')) if p.is_file()},
        isolation=dict(checkout=str(ROOT),branch=git('branch','--show-current').decode().strip(),
            base_commit=git('rev-parse','HEAD').decode().strip(),
            setup_note='Initial sandbox hardlink clone failed before checkout; approved no-hardlink local clone succeeded',
            agents='No applicable repository or ancestor AGENTS.md found')))
    print('Preservation baseline saved',flush=True)


def run(name,argv):
    overrides=dict(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code={str(p.relative_to(ROOT)):identity(p) for d in ('src','scripts','tests')
        for p in sorted((ROOT/d).rglob('*.py'))}
    code['runs/r04/verify.py']=identity(OUT/'verify.py')
    stamp=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    result=subprocess.run([str(v) for v in argv],cwd=ROOT,env=os.environ|overrides,capture_output=True)
    for label,raw in (('stdout',result.stdout),('stderr',result.stderr)):
        with (OUT/(name+'.'+label+'.txt')).open('xb') as stream:stream.write(raw)
    save(name+'.command.json',dict(argv=[str(v) for v in argv],cwd=str(ROOT),environment_overrides=overrides,
        started_at_utc=stamp,ended_at_utc=datetime.now(timezone.utc).isoformat(),duration_seconds=time.monotonic()-tick,
        exit_code=result.returncode,tested_sources=code,
        outputs={label:identity(OUT/(name+'.'+label+'.txt')) for label in ('stdout','stderr')}))
    assert all(identity(ROOT/k)==v for k,v in code.items()),'code changed during execution'
    print(name+': exit='+str(result.returncode),flush=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)
    assert result.returncode==0,name


def audit():
    from chess_rl.adapter import apply_move, encode_position, legal_mask, move_to_id, terminal_result
    from chess_rl.canonical_games import canonical_payload,game_id
    from chess_rl.evaluation.matches import digest
    from chess_rl.evaluation.replay import audit_match
    from chess_rl.position_sampling import rule_position
    from chess_rl.replay import ReplayBuffer, read_source
    import chess
    plan=json.loads((OUT/'frozen-plan.json').read_text())
    games=json.loads((OUT/'games.json').read_text())
    eligible=json.loads((OUT/'eligible-membership.json').read_text())
    exported=ReplayBuffer(OUT/'replay.sqlite').export()
    assert digest(exported)==digest(json.loads((OUT/'replay-export.json').read_text()))
    assert digest(exported)==eligible['replay_id']
    targets=[json.loads(line) for line in (OUT/'targets.jsonl').read_text().splitlines()]
    assert targets==exported['samples']
    samples={s['sample_id']:s for s in targets}
    held={json.loads(v)['game_id'] for v in Path(plan['source_config']['heldout_games']).read_text().splitlines()}
    protected={json.loads(v)['rule_position'] for v in Path(plan['source_config']['protected_positions']).read_text().splitlines()}
    links=[];unfinished_roots=0;root_count=0
    for snapshot in games:
        config,game=snapshot['configuration'],snapshot['game']
        from chess_rl.evaluation.matches import Opening
        opening=Opening(**{k:config['opening'][k] for k in ('opening_id','initial_fen','moves_uci','history_complete','source')})
        assert read_source(OUT/opening.opening_id)==snapshot
        assert config['checkpoint']['sha256']==plan['checkpoint_identity']['sha256']
        assert config['checkpoint']['checkpoint_id']==plan['checkpoint_id']
        state=opening.state()
        final=opening.state()
        for uci in game['moves_uci'][len(opening.moves_uci):]:apply_move(final,move_to_id(final,chess.Move.from_uci(uci)))
        terminal=terminal_result(final)
        assert (terminal is not None)==(game['status']=='completed')
        for position in game['positions']:
            root_count+=1
            assert position['root_fen']==state.board.fen(en_passant='fen')
            assert position['legal_mask']==legal_mask(state).tolist()
            visits=position['root_visits'];n=sum(visits)
            assert n==config['settings']['simulation_budget']==position['completed_simulations']
            assert all((not v) or legal for v,legal in zip(visits,position['legal_mask']))
            assert position['visit_policy']==[v/n for v in visits]
            matching=[s for s in targets if s['source_id']==digest(snapshot) and s['position']['ply']==position['ply']]
            if game['status']!='completed':
                assert matching==[] and game['outcome'] is None and game['winner'] is None
                unfinished_roots+=1
            else:
                assert len(matching)==1
                sample=matching[0]
                expected=[0,1,0] if terminal.winner is None else [1,0,0] if terminal.winner==state.board.turn else [0,0,1]
                assert sample['wdl']==expected and sample['position']==position
                assert sample['moves_uci']==[m.uci() for m in state.board.move_stack]
                assert sample['available_snapshots']==encode_position(state).available_snapshots
                exclusion=(game_id(canonical_payload(game['initial_fen'],game['moves_uci'])) in held or rule_position(state.board) in protected)
                assert (sample['sample_id'] in eligible['excluded_heldout'])==exclusion
                assert (sample['sample_id'] in eligible['replay_membership'])== (not exclusion)
                links.append(dict(opening_id=opening.opening_id,configuration_sha256=digest(config),
                    source_id=sample['source_id'],game_id=sample['game_id'],sample_id=sample['sample_id'],
                    root_fen=position['root_fen'],root_player=position['side_to_move'],ply=position['ply'],
                    visits_sha256=digest(visits),visit_policy_sha256=digest(position['visit_policy']),
                    nonzero_visits=sum(v>0 for v in visits),completed_simulations=n,
                    played_uci=position['move_uci'],outcome=game['outcome'],ending_reason=game['ending_reason'],
                    replay_wdl=expected,eligible=not exclusion))
            apply_move(state,move_to_id(state,chess.Move.from_uci(position['move_uci'])))
    full,prefix,resume=[json.loads((OUT/(v+'.json')).read_text()) for v in ('full','prefix','resume')]
    assert full['steps']==prefix['steps']+resume['steps'] and full['final']==resume['final']
    consumed=[]
    for step in full['steps']:
        assert step['metrics']['supervised']['batch_size']==step['metrics']['selfplay']['batch_size']==2
        assert step['metrics']['supervised_contribution']>0 and step['metrics']['selfplay_contribution']>0
        for sample_id in step['selfplay_sample_ids']:
            assert sample_id in eligible['replay_membership'];consumed.append(sample_id)
        for sample_id,wdl in zip(step['selfplay_sample_ids'],step['final_selfplay_wdl']):assert wdl==samples[sample_id]['wdl']
    for link in links:
        link['consumed_by_updates']=[s['update'] for s in full['steps'] if link['sample_id'] in s['selfplay_sample_ids']]
        link['updated_checkpoint']=resume['checkpoint']
    matches=[json.loads(v) for v in (OUT/'matches.jsonl').read_text().splitlines()]
    match=matches[0];match_audit=audit_match(match)
    config=match['configuration']
    assert config['left_player']['checkpoint']['sha256']==plan['checkpoint_identity']['sha256']
    assert config['right_player']['checkpoint']['sha256']==resume['checkpoint']['identity']['sha256']
    for player in (config['left_player'],config['right_player']):
        assert player['checkpoint']['kind']=='fixture' and player['settings']['exploration_noise'] is False
    assert config['protocol_only'] is True and config['left_player']['settings']==config['right_player']['settings']
    # Reopen and restore the final mixed checkpoint as well as its inference use.
    from dataclasses import asdict
    import random
    import numpy as np
    import torch
    from chess_rl.supervised import open_dataset
    from chess_rl.training.mixed_state import MixedSources,MixedSettings
    from chess_rl.training.checkpoints import ModelConfig,restore_training,_rng_state
    from scripts.run_r03 import tree_hash
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source_config=plan['source_config']
    reopened=MixedSources(open_dataset(**{k:Path(v) for k,v in source_config['dataset'].items()},split='train'),
        ReplayBuffer(source_config['replay']),heldout_games_path=source_config['heldout_games'],
        protected_positions_path=source_config['protected_positions'])
    expected_provenance=dict(task='R04 local plumbing only; never pretrained P',plan_sha256=digest(plan),
        starting_checkpoint=plan['checkpoint_identity'],starting_checkpoint_id=plan['checkpoint_id'],
        boundary='new Adam and mixed state at zero; no supervised optimizer/cursor restore',
        membership_sha256=digest(eligible),sources=full['sources'])
    random.seed(21);np.random.seed(22);torch.manual_seed(23)
    restored=restore_training(resume['checkpoint']['path'],expected_config=ModelConfig(**plan['model_config']),
        expected_provenance=expected_provenance,mixed_sources=reopened,
        expected_mixed_settings=MixedSettings(**plan['mixed_settings']))
    assert restored.data_order is None
    assert tree_hash(restored.model.state_dict())==resume['final']['model_sha256']
    assert tree_hash(restored.optimizer.state_dict())==resume['final']['optimizer_sha256']
    assert digest(asdict(restored.mixed_state))==digest(resume['final']['mixed_state'])
    assert asdict(restored.progress)==resume['final']['progress']
    assert tree_hash(_rng_state())==resume['final']['rng_sha256']
    save('chain-audit.json',dict(final_training_restore_exact=True,
        actual_root_to_target_links=links,generated_roots=root_count,
        unfinished_roots_excluded=unfinished_roots,eligible_sample_ids=eligible['replay_membership'],
        consumed_selfplay_ids=consumed,unique_consumed_selfplay_ids=sorted(set(consumed)),
        logical_consumed_supervised_samples=6,logical_consumed_selfplay_samples=6,
        physical_verification_samples_per_source=12,continuation_exact=True,match_audit=match_audit,
        evaluation_by_opening=[dict(opening_id=g['opening']['opening_id'],leg=g['leg'],status=g['status'],outcome=g['outcome']) for g in match['games']],
        reserved_protocol_test_played=False,source_metrics=[s['metrics'] for s in full['steps']],
        no_strength_claim=True,no_pretrained_P_selection=True))
    print('Independent roots/targets/updates/checkpoints/evaluation audit passed')


def preserve():
    before=json.loads((OUT/'preservation-before.json').read_text())
    current=primary_snapshot()
    assert current==before['primary'],'primary checkout or user edits drift'
    assert all(identity(p)==v for p,v in before['accepted_external'].items())
    allowed={'README.md','coding-plan.md'}
    assert all(identity(ROOT/p)==v for p,v in before['accepted_checkout'].items() if p not in allowed)
    for name in allowed:
        original=git('show',before['isolation']['base_commit']+':'+name)
        assert (ROOT/name).read_bytes().startswith(original),'only append scoped R04 documentation'
    save('preservation-after.json',dict(primary_exact=True,user_patch_and_index_exact=True,
        primary_tracked_files=len(current['tracked']),accepted_checkout_unchanged_except_documentation_appends=True,
        accepted_checkout_files=len(before['accepted_checkout']),accepted_external_files=len(before['accepted_external']),
        historical_external_bytes=sum(v['bytes'] for v in before['accepted_external'].values()),
        original_plan_identity=current['tracked']['coding-plan.md'],base_commit=before['isolation']['base_commit']))
    files=[p for p in Path(json.loads((OUT/'frozen-plan.json').read_text())['external_directory']).rglob('*') if p.is_file()]
    save('new-external-artifact-index.json',{str(p):identity(p) for p in files})
    print('Accepted artifacts, primary index and user edits preserved exactly')


if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('prepare','audit','preserve'):globals()[mode]()
    elif mode=='audit-receipt':run('chain-audit',[PYTHON,'-B','runs/r04/verify.py','audit'])
    elif mode=='audit-final':run('chain-audit-final',[PYTHON,'-B','runs/r04/verify.py','audit'])
    elif mode=='preserve-receipt':run('preservation-check',[PYTHON,'-B','runs/r04/verify.py','preserve'])
    elif mode=='model':run('regression-model',[PYTHON,'-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider','-p','runs.m06.diagnostic_plugin','--junitxml=runs/r04/regression-model.xml'])
    elif mode=='adapter':run('regression-adapter',[ADAPTER,'-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider','--junitxml=runs/r04/regression-adapter.xml'])
    else:run(mode,[PYTHON,'-B','scripts/run_r04.py',mode])
