"""Derive archive expectations directly from chess history, without replay.finalize/audit."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import chess
from checks import ROOT, OUT, PRIMARY, run, save, identity

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()
def move_id(board,move):
    origin=move.from_square ^ (0 if board.turn else 56)
    dest=move.to_square ^ (0 if board.turn else 56)
    return 64*origin+dest if not move.promotion else 4096+4*(8*(origin%8)+dest%8)+[5,4,3,2].index(move.promotion)
def database(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        return {table:db.execute('SELECT * FROM '+table+' ORDER BY '+key).fetchall()
            for table,key in [('metadata','id'),('sources','ordinal'),('games','game_id'),('samples','sample_id')]}

archive=ROOT/'runs/r01-integration/real/games.json'
records=json.loads(archive.read_bytes())
assert records==json.loads((ROOT/'runs/r01/real/games.json').read_bytes())
assert len(records)==20
expected_samples=[]; expected_sources=[]; expected_games={}; expected_receipts=[]
for record in records:
    cfg=record['configuration']; game=record['game']
    snapshot=dict(configuration=cfg,game=game)
    source_id=digest(snapshot); game_id=digest([game['initial_fen'],game['moves_uci']])
    assert game['configuration_sha256']==digest(cfg)
    board=chess.Board(game['initial_fen']); boards=[board.copy(stack=True)]
    for uci in game['moves_uci']:
        move=chess.Move.from_uci(uci); assert move in board.legal_moves
        board.push(move); boards.append(board.copy(stack=True))
    assert board.fen(en_passant='fen')==game['final_fen']
    assert int(board.promoted)==game['final_promoted']
    finished=board.outcome(claim_draw=False)
    reason=finished.termination.name if finished else 'THREEFOLD_REPETITION' if board.is_repetition(3) else 'FIFTY_MOVES' if board.is_fifty_moves() else None
    if game['status']!='completed':
        assert not reason and game['winner'] is None and game['outcome'] is None
        disposition='excluded-'+game['status']; samples=[]
    else:
        assert reason==game['ending_reason']
        winner=finished.winner if finished else None
        winner_name=None if winner is None else 'white' if winner else 'black'
        outcome='1/2-1/2' if winner is None else '1-0' if winner else '0-1'
        assert game['winner']==winner_name and game['outcome']==outcome
        disposition='duplicate' if game_id in expected_games else 'admitted'
        samples=[]
        for position in game['positions']:
            ply=position['ply']; root=boards[ply]
            assert not root.outcome(claim_draw=False) and not root.is_repetition(3) and not root.is_fifty_moves()
            mask=[False]*4352
            for move in root.legal_moves: mask[move_id(root,move)]=True
            assert position['legal_mask']==mask
            assert position['root_fen']==root.fen(en_passant='fen') and position['root_promoted']==int(root.promoted)
            assert position['side_to_move']==('white' if root.turn else 'black')
            visits=position['root_visits']; policy=position['visit_policy']; total=sum(visits)
            assert total==position['completed_simulations']>0
            assert all(type(v) is int and v>=0 and (v==0 or mask[i]) for i,v in enumerate(visits))
            assert policy==[v/total for v in visits]
            assert move_id(root,chess.Move.from_uci(position['move_uci']))==position['move_id']
            assert game['moves_uci'][ply]==position['move_uci']
            wdl=[0,1,0] if winner is None else [1,0,0] if winner==root.turn else [0,0,1]
            samples.append(dict(schema='r02-sample-v1',sample_id=digest(['r02-sample-v1',game_id,digest(cfg),ply]),
                source_id=source_id,game_id=game_id,configuration_sha256=digest(cfg),contracts=cfg['contracts'],
                checkpoint=cfg['checkpoint'],opening=cfg['opening'],search_settings=cfg['settings'],
                source='completed-R01-self-play',policy_target_type='root-visit-distribution',
                wdl_target_type='exact-final-root-side-WDL',initial_fen=game['initial_fen'],
                moves_uci=game['moves_uci'][:ply],history_complete=True,available_snapshots=min(8,ply+1),
                position=copy.deepcopy(position),wdl=wdl,outcome=outcome,ending_reason=reason))
        if disposition=='admitted':
            expected_games[game_id]=(source_id,len(samples)); expected_samples.extend(samples)
    expected_sources.append(dict(source_id=source_id,game_id=game_id,disposition=disposition,snapshot=snapshot))
    expected_receipts.append(dict(fixture=record.get('fixture'),source_id=source_id,disposition=disposition,
        repeated=False,added_samples=len(samples) if disposition=='admitted' else 0))
expected_samples.sort(key=lambda s:s['sample_id'])
assert len(expected_samples)==4
assert {(s['position']['side_to_move'],tuple(s['wdl'])) for s in expected_samples}=={
    ('white',(1,0,0)),('black',(1,0,0)),('white',(0,1,0)),('black',(0,1,0))}
historical=json.loads((ROOT/'runs/r02/summary.json').read_bytes())
assert expected_samples==[json.loads(line) for line in (ROOT/'runs/r02/samples.jsonl').read_text().splitlines()]
assert digest(expected_samples)==historical['samples_sha256']=='921335a46de1efec6f6fa3a4155d54fe9956d978405fa22dfcdcf9a60b5cbd22'
counts=dict(Counter(s['disposition'] for s in expected_sources))
assert counts=={'admitted':7,'duplicate':1,'excluded-truncated':11,'excluded-interrupted':1}
dbpath=OUT/'fresh-replay.sqlite'; before_db=None
for name,records_path,create in [('archive-admission',archive,True),('archive-restart',archive,False),
    ('archive-original-repeat',ROOT/'runs/r01/real/games.json',False)]:
    argv=[PRIMARY/'.venv/bin/python','-B','scripts/run_r02.py','--records',records_path,'--buffer',dbpath,
        '--report',OUT/(name+'.json'),'--samples',OUT/(name+'.jsonl')]
    if create: argv+=['--create']
    run(name,argv)
    report=json.loads((OUT/(name+'.json')).read_bytes())
    assert report['receipts']==(expected_receipts if create else [dict(r,repeated=True,added_samples=0) for r in expected_receipts])
    assert report['summary']==dict(source_counts=counts,training_positions=4)
    assert [json.loads(line) for line in (OUT/(name+'.jsonl')).read_text().splitlines()]==expected_samples
    assert (OUT/(name+'.jsonl')).read_bytes()==(ROOT/'runs/r02/samples.jsonl').read_bytes()
    rows=database(dbpath)
    assert [(s['source_id'],s['game_id'],s['disposition'],canonical(s['snapshot'])) for s in expected_sources]==[
        (source,game,disposition,payload) for _,source,game,disposition,payload in rows['sources']]
    assert {game:(source,count) for game,source,count in rows['games']}==expected_games
    assert [json.loads(payload) for _,_,payload in rows['samples']]==expected_samples
    exported=dict(manifest=json.loads(rows['metadata'][0][1]),sources=expected_sources,samples=expected_samples,
        summary=dict(source_counts=counts,training_positions=4))
    assert digest(exported)==report['audited_export_sha256']==historical['audited_export_sha256']=='4bdd6f656e159cfe8fb46db65264e876fdee5abe6abb1fc6f85bec50cf6eedf4'
    if create: before_db=rows; original_db_identity=identity(dbpath)
    else: assert rows==before_db and identity(dbpath)==original_db_identity
save(OUT/'archive-verification.json',dict(derivation='Direct python-chess actual history and independently spelled canonical SHA256; no R02 finalize/audit for expected targets',
    accepted_archive=identity(archive),original_archive_exact=True,records=20,summary=dict(source_counts=counts,training_positions=4),
    samples_sha256=digest(expected_samples),audited_export_sha256=digest(exported),
    source_game_sample_identities_and_targets_exact=True,fresh_process_receipts_exact=True,
    restart_database_bytes_and_rows_exact=True,samples=expected_samples))
print('Independent four-sample derivation and three fresh-process ingestion checks passed.')
