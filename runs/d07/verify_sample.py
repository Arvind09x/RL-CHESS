"""Independent D07 sample verification against direct chess and frozen D05 hashes.

No worker replay/target helpers are used. This only reads accepted data; its
report goes to D07, never the teacher bundle. No engine is started.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import chess
import numpy as np

from chess_rl.adapter import encode_position, legal_mask, make_state
from chess_rl.supervised import open_dataset

ROOT=Path(__file__).resolve().parents[2]
bundle=ROOT/'runs/d05/tiny-final'
args=dict(bundle=bundle,manifest_path=ROOT/'data/manifests/elite-2025-01-d04-v1.json',
    games_path=Path('/private/tmp/rl-chess-d03-prepared/games.jsonl'),
    split_audit_path=Path('/private/tmp/rl-chess-d03-prepared/split-audit.json'),
    prepared=Path('/private/tmp/rl-chess-d04-prepared'),teacher_evidence=ROOT/'runs/d01/stockfish-provenance.json')
data=open_dataset(**args,split='train')
labels=[json.loads(r) for r in (bundle/'labels.jsonl').read_text().splitlines()]
needed={r['game_id'] for r in labels}
games={}
with args['games_path'].open() as handle:
    for raw in handle:
        game=json.loads(raw)
        if game['game_id'] in needed:games[game['game_id']]=game
frozen={(r['game_id'],r['sample_ply']):r for r in json.loads((bundle/'independent-replay.json').read_text())['record_checks']}
assert len(data)==len(labels)==12
checked=[];coverage=Counter()
for i,label in enumerate(labels):
    sample=data[i];game=games[label['game_id']]
    board=chess.Board(game['start_fen'])
    for uci in game['moves_uci'][:label['sample_ply']]:board.push_uci(uci)
    assert board.fen(en_passant='fen')==sample.state.board.fen(en_passant='fen')==label['root_fen']
    assert board.promoted==sample.state.board.promoted
    assert board.move_stack==sample.state.board.move_stack
    assert len(board.move_stack)==label['sample_ply'] and sample.state.history_complete is True
    payload=dict(start_fen=board.root().fen(en_passant='fen'),moves_uci=[m.uci() for m in board.move_stack],history_complete=True)
    serialized=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True)
    assert hashlib.sha256(serialized.encode()).hexdigest()==label['replay_sha256']
    fresh=make_state(game['start_fen'],game['moves_uci'][:label['sample_ply']],history_complete=True)
    assert np.array_equal(sample.features,encode_position(fresh).features)
    assert np.array_equal(sample.legal_mask,legal_mask(fresh))
    def independent_id(move):
        origin,target=move.from_square,move.to_square
        if not board.turn:origin^=56;target^=56
        return (4096+4*(8*(origin%8)+target%8)+(chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT).index(move.promotion)
            if move.promotion else 64*origin+target)
    expected_mask=np.zeros(4352,dtype=np.bool_)
    for move in board.legal_moves:expected_mask[independent_id(move)]=True
    assert np.array_equal(sample.legal_mask,expected_mask)
    move=chess.Move.from_uci(label['best_move_uci'])
    assert move in board.legal_moves and independent_id(move)==sample.policy_class==label['best_move_id']
    search=label['returned_search'];counts=search['wdl']['relative']
    if search['wdl']['turn']!=('white' if board.turn else 'black'):counts=counts[::-1]
    assert counts==label['teacher_wdl_counts'] and sum(counts)==1000
    assert np.array_equal(sample.wdl,np.array(counts,dtype=np.float32)/np.float32(1000))
    features_sha=hashlib.sha256(sample.features.astype('<f4').tobytes()).hexdigest()
    mask_sha=hashlib.sha256(sample.legal_mask.tobytes()).hexdigest()
    expected=frozen[sample.reference]
    assert features_sha==expected['features_sha256'] and mask_sha==expected['legal_mask_sha256']
    coverage[f"{label['phase']}:{'white' if board.turn else 'black'}"]+=1
    checked.append(dict(game_id=sample.reference[0],sample_ply=sample.reference[1],root_fen=label['root_fen'],
        features_sha256=features_sha,legal_mask_sha256=mask_sha,policy_class=sample.policy_class,wdl=sample.wdl.tolist()))
refs=[data[i].reference for i in range(len(data))]
batch_report=[]
for size in (1,5,12,20):
    batches=list(data.batches(size))
    assert [r for b in batches for r in b.references]==refs
    for b in batches:
        assert b.features.dtype==np.float32 and b.legal_mask.dtype==np.bool_
        assert b.policy_classes.dtype==np.int64 and b.wdl.dtype==np.float32
        assert np.all(b.legal_mask[np.arange(len(b.references)),b.policy_classes])
    batch_report.append(dict(requested_size=size,sizes=[len(b.references) for b in batches],
        shapes=[dict(features=list(b.features.shape),mask=list(b.legal_mask.shape),
            policy=list(b.policy_classes.shape),wdl=list(b.wdl.shape)) for b in batches]))
assert [r for b in data.batches(5,order=list(reversed(range(12)))) for r in b.references]==refs[::-1]
assert [len(b.references) for b in data.batches(5,drop_last=True)]==[5,5]
for split in ('validation','test'):
    held=open_dataset(**args,split=split)
    assert len(held)==0 and list(held.batches(5))==[]
report=dict(records=checked,coverage=dict(coverage),batches=batch_report,split_selection_verified=True,
    direct_board_and_history_agree=True,frozen_D05_feature_mask_hashes_agree=True,
    independent_move_ids_and_soft_targets_agree=True,torch_imported='torch' in sys.modules,
    limitations='12 existing training roots, CPU only; no new annotation/training/scale claims')
(ROOT/sys.argv[1]).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('records','batches')},indent=2))
