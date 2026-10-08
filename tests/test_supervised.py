"""D07 sealed synthetic bundles; no executable or teacher searches."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys

import chess
import numpy as np
import pytest

from chess_rl.adapter import encode_position, legal_mask, make_state, move_to_id
from chess_rl.canonical_games import assign_split, canonical_json, canonical_payload, game_id
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.position_sampling import phase, rule_position, sample_record
from chess_rl.stockfish_worker import replay_identity
from chess_rl.supervised import BUNDLE_FILES, open_dataset

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'runs/d01/stockfish-provenance.json'


def write_rows(path, rows):
    path.write_text(''.join(json_line(r) for r in rows))


def read_rows(path):
    return [json.loads(r) for r in path.read_text().splitlines()]


def make_game(manifest, split, fen=chess.STARTING_FEN, moves=()):
    for counter in range(1, 10000):
        start = fen.rsplit(' ', 1)[0] + f' {counter}'
        identity = game_id(canonical_payload(start, list(moves)))
        if assign_split(identity, manifest.split_seed) == split:
            break
    return dict(game_id=identity, split=split, start_fen=start, moves_uci=list(moves),
        dataset_id=manifest.dataset_id, encoding_version=manifest.encoding_version,
        move_vocab_version=manifest.move_vocab_version, ending_policy=manifest.ending_policy,
        history_complete=True, history_origin='declared-pgn-start', record_version='replay-jsonl-v1',
        wdl_order=list(manifest.wdl_order))


def target_fixture(game, ref, manifest, provenance, best=None):
    """Spelled serialized fixture, never produced by annotation machinery."""
    state = make_state(game['start_fen'], game['moves_uci'][:ref['sample_ply']], history_complete=True)
    move = chess.Move.from_uci(best) if best else next(iter(state.board.legal_moves), chess.Move.null())
    counts = [700, 200, 100] if state.board.turn else [100, 200, 700]
    info = dict(multipv=1, wdl=dict(relative=[700, 200, 100], turn='white'),
        pv=[move.uci()], score=dict(relative_cp=125, relative_mate=None, text='+125', turn='white'),
        depth=12, seldepth=17, nodes=100123, nps=90000, hashfull=1, tbhits=0, time=1.1)
    options = dict(Threads=1, Hash=128, MultiPV=1, UCI_ShowWDL=True, UCI_LimitStrength=False,
        SyzygyPath='', Ponder=False, UCI_Chess960=False, EvalFile=manifest.engine.embedded_nnue[0].name,
        EvalFileSmall=manifest.engine.embedded_nnue[1].name)
    options['Skill Level'] = 20
    attempt = dict(reference=ref, attempt=1, started_at_utc='2026-10-08T00:00:00+00:00',
        finished_at_utc='2026-10-08T00:00:01+00:00', requested_settings=asdict(manifest.teacher),
        status='success', returned=dict(completed=True, hash_reset_completed=True,
            principal_response=info, returned_rows=[info], bestmove=move.uci(), returned_bestmove=move.uci(),
            engine_id=dict(name=manifest.engine.uci_name, author=manifest.engine.uci_author), effective_options=options))
    label = dict(ref, record_version='stockfish-label-v1', root_fen=state.board.fen(en_passant='fen'),
        history_complete=True, available_snapshots=min(8, ref['sample_ply']+1), replay_sha256=replay_identity(state),
        dataset_id=manifest.dataset_id, encoding_version=manifest.encoding_version,
        move_vocab_version=manifest.move_vocab_version, ending_policy=manifest.ending_policy,
        wdl_order=list(manifest.wdl_order), teacher_config_id=manifest.teacher.config_id,
        requested_settings=asdict(manifest.teacher), successful_attempt=1, target_provenance=provenance,
        best_move_uci=move.uci(), best_move_id=move_to_id(state, move) if move else 0,
        teacher_wdl_counts=counts, teacher_wdl=[n/1000 for n in counts], returned_search=info,
        returned_bestmove_uci=move.uci())
    return label, attempt


def seal_bundle(out):
    selected, labels, attempts, excluded = [read_rows(out/name) for name in BUNDLE_FILES[:4]]
    summary = dict(status='complete', pending_references=[], selected=len(selected),
        successful_labels=len(labels), attempts=len(attempts), excluded=len(excluded),
        failed_attempts=sum(a['status'] != 'success' for a in attempts),
        output_identities={name:file_identity(out/name) for name in BUNDLE_FILES})
    (out/'summary.json').write_text(json.dumps(summary))


def fixture_bundle(tmp_path, specifications=None):
    manifest = DatasetManifest.load(ROOT/'data/manifests/elite-2025-01-d04-v1.json')
    if specifications is None:
        specifications = [
            ('train', chess.STARTING_FEN, ['e2e4','e7e5','g1f3','b8c6'], [1,2,3], None),
            ('validation', chess.STARTING_FEN, ['d2d4','d7d5','c1f4','g8f6'], [1,2], None),
            ('test', chess.STARTING_FEN, ['c2c4','e7e5','b1c3','f8b4'], [1,2], None)]
    prepared, out = tmp_path/'prepared', tmp_path/'bundle'
    prepared.mkdir(); out.mkdir()
    games, refs, moves = [], [], []
    for split, fen, prefix, plies, best in specifications:
        game = make_game(manifest, split, fen, prefix)
        games.append(game)
        for ply in plies:
            state = make_state(game['start_fen'], prefix[:ply], history_complete=True)
            refs.append(sample_record(game['game_id'], split, (ply,phase(state.board,ply),rule_position(state.board))))
            moves.append(best)
    games_path, split_path, manifest_path = tmp_path/'games.jsonl', tmp_path/'split.json', tmp_path/'manifest.json'
    write_rows(games_path, games)
    split_audit = dict(complete=True, audit_version='canonical-split-audit-v1', dataset_id=manifest.dataset_id,
        split_seed=manifest.split_seed, split_algorithm=manifest.preparation.split_algorithm,
        archive=dict(sha256=manifest.archive.downloaded_sha256, bytes=manifest.archive.downloaded_bytes),
        outputs={'games.jsonl':file_identity(games_path)}, assignments_sha256='fixture assignments')
    split_path.write_text(json.dumps(split_audit))
    write_rows(prepared/'samples.jsonl', refs)
    write_rows(prepared/'heldout-games.jsonl', [dict(game_id=g['game_id'],split=g['split']) for g in games if g['split']!='train'])
    write_rows(prepared/'protected-heldout.jsonl', [r for r in refs if r['split']!='train'])
    write_rows(prepared/'exclusions.jsonl', [])
    inputs = {'games.jsonl':file_identity(games_path), 'split-audit.json':file_identity(split_path)}
    audit = dict(audit_version='position-overlap-audit-v1', complete=True, dataset_id=manifest.dataset_id,
        sampling_version='phase-spread-sha256-v1', sampling_seed=manifest.sampling_seed,
        sampling_algorithm=manifest.preparation.sampling_algorithm, phase_boundaries=manifest.preparation.phase_boundaries,
        overlap_identity=manifest.overlap_policy, frozen_registry_version='protected-heldout-v1',
        cross_split_overlaps_after=0, labels_produced=0, assignments_sha256=split_audit['assignments_sha256'],
        inputs=inputs, outputs={name:file_identity(prepared/name) for name in
            ('samples.jsonl','heldout-games.jsonl','protected-heldout.jsonl','exclusions.jsonl')})
    (prepared/'overlap-audit.json').write_text(json.dumps(audit))
    manifest = replace(manifest, preparation=replace(manifest.preparation,
        split_audit_sha256=file_identity(split_path)['sha256'],
        overlap_audit_sha256=file_identity(prepared/'overlap-audit.json')['sha256'],
        protected_heldout_registry_sha256=file_identity(prepared/'protected-heldout.jsonl')['sha256']))
    manifest_path.write_text(manifest.to_json())
    identities = dict(inputs, **{name:file_identity(prepared/name) for name in
        ('samples.jsonl','heldout-games.jsonl','protected-heldout.jsonl','exclusions.jsonl','overlap-audit.json')},
        manifest=file_identity(manifest_path))
    evidence = json.loads(EVIDENCE.read_text())
    teacher = dict(executable_identity=dict(bytes=evidence['executable_bytes'],sha256=evidence['executable_sha256']),
        embedded_nnue=evidence['nnue'], build_evidence=file_identity(EVIDENCE), python_chess='1.999',chess='1.11.2')
    provenance = dict(kind='stockfish-root-wdl-and-pv',manifest_sha256=identities['manifest']['sha256'],
        games_sha256=identities['games.jsonl']['sha256'], samples_sha256=identities['samples.jsonl']['sha256'],
        overlap_audit_sha256=identities['overlap-audit.json']['sha256'],
        executable_sha256=manifest.engine.executable_sha256, embedded_nnue=evidence['nnue'])
    labels, attempts = [], []
    for ref, best in zip(refs, moves):
        game = next(g for g in games if g['game_id']==ref['game_id'])
        label, attempt = target_fixture(game, ref, manifest, provenance, best)
        labels.append(label); attempts.append(attempt)
    write_rows(out/'selected.jsonl', refs); write_rows(out/'labels.jsonl', labels)
    write_rows(out/'attempts.jsonl', attempts); write_rows(out/'exclusions.jsonl', [])
    (out/'settings.json').write_text(json.dumps(dict(artifact_identities=identities,
        teacher=teacher,requested_settings=asdict(manifest.teacher),max_attempts=2)))
    seal_bundle(out)
    return dict(bundle=out,manifest_path=manifest_path,games_path=games_path,split_audit_path=split_path,
        prepared=prepared,teacher_evidence=EVIDENCE)


@pytest.fixture
def bundle(tmp_path):
    return fixture_bundle(tmp_path)


def test_reloads_splits_and_batches(bundle):
    memberships = []
    for split, count in [('train',3),('validation',2),('test',2)]:
        data = open_dataset(**bundle,split=split)
        assert data.split == split and len(data)==count
        memberships.append({data[i].reference for i in range(count)})
        for i in range(count):
            sample = data[i]
            assert np.array_equal(sample.features,encode_position(sample.state).features)
            assert np.array_equal(sample.legal_mask,legal_mask(sample.state))
            assert sample.legal_mask[sample.policy_class]
            assert np.array_equal(sample.wdl,np.array([.7,.2,.1] if sample.state.board.turn else [.1,.2,.7],dtype=np.float32))
        batches = list(data.batches(2))
        assert [len(b.references) for b in batches] == ([2,1] if count==3 else [2])
        for b in batches:
            size = len(b.references)
            assert b.features.shape == (size,64,128) and b.features.dtype == np.float32
            assert b.legal_mask.shape == (size,4352) and b.legal_mask.dtype == np.bool_
            assert b.policy_classes.shape == (size,) and b.policy_classes.dtype == np.int64
            assert b.wdl.shape == (size,3) and b.wdl.dtype == np.float32
        assert [r for b in data.batches(1,order=list(reversed(range(count)))) for r in b.references] == [data[i].reference for i in reversed(range(count))]
        assert len(list(data.batches(2,drop_last=True)))==count//2
    assert all(not memberships[i]&memberships[j] for i in range(3) for j in range(i))


def test_input_ownership_and_empty_split(bundle):
    d = open_dataset(**bundle,split='train')
    a = d[0]; expected = a.features.copy()
    a.state.board.push(next(iter(a.state.board.legal_moves))); a.features[:]=0; a.legal_mask[:]=False; a.wdl[:]=0
    assert np.array_equal(d[0].features,expected)
    b = next(d.batches(2)); b.features[:]=0
    assert np.array_equal(d[0].features,expected)
    labels = read_rows(bundle['bundle']/'labels.jsonl')
    assert d[0].state.history_complete and len(d[0].state.board.move_stack)==labels[0]['sample_ply']


@pytest.mark.parametrize('split',[None,'all','Train',''])
def test_explicit_split(bundle,split):
    with pytest.raises(ValueError,match='split'):
        open_dataset(**bundle,split=split)
    with pytest.raises(TypeError):
        open_dataset(**bundle)


@pytest.mark.parametrize('kwargs',[dict(batch_size=0),dict(batch_size=True),dict(batch_size=1.5),
    dict(batch_size=2,drop_last=1),dict(batch_size=2,order=[0,0,1]),dict(batch_size=2,order=[0,1]),
    dict(batch_size=2,order=[0,1,True]),dict(batch_size=2,order=[0,1,3])])
def test_invalid_batch_options(bundle,kwargs):
    data = open_dataset(**bundle,split='train')
    with pytest.raises(ValueError): list(data.batches(**kwargs))


@pytest.mark.parametrize('field,value',[
    ('record_version','stockfish-label-v2'),('encoding_version','chess128-v1'),('move_vocab_version','other'),
    ('ending_policy','old'),('dataset_id','other'),('wdl_order',['loss','draw','win']),
    ('teacher_config_id','other'),('root_fen',chess.STARTING_FEN),('replay_sha256','0'*64),
    ('history_complete',1),('available_snapshots',True),('sample_ply',True),('sample_ply',99999),
    ('best_move_uci','e2e5'),('best_move_id',True),('best_move_id',4352),('best_move_id',796),
    ('teacher_wdl_counts',[100,200,699]),('teacher_wdl_counts',[True,200,799]),
    ('teacher_wdl_counts',[-1,200,801]),('teacher_wdl',[.7,.2,.1]),('teacher_wdl',[float('nan'),.2,.7]),
    ('teacher_wdl',[.1,.2]),('successful_attempt',2),('successful_attempt',True),
    ('target_provenance',{}),('requested_settings',{}),('extra',1)])
def test_invalid_label(bundle,field,value):
    path = bundle['bundle']/'labels.jsonl'
    labels=read_rows(path); labels[0][field]=value
    # NaN cannot go through canonical writer; deliberately malformed JSON bytes.
    path.write_text(''.join(json.dumps(r)+'\n' for r in labels)); seal_bundle(bundle['bundle'])
    with pytest.raises(ValueError): open_dataset(**bundle,split='train')


@pytest.mark.parametrize('part', ['completed','hash_reset_completed','bestmove','principal_response',
    'engine_id','effective_options','returned_rows','transport_errors'])
def test_bad_success_linkage(bundle,part):
    path=bundle['bundle']/'attempts.jsonl'; rows=read_rows(path)
    rows[0]['returned'][part] = (False if part in ('completed','hash_reset_completed') else
        'e2e5' if part=='bestmove' else ['malformed'] if part=='transport_errors' else [] if part=='returned_rows' else {})
    write_rows(path,rows); seal_bundle(bundle['bundle'])
    with pytest.raises(ValueError): open_dataset(**bundle,split='train')


@pytest.mark.parametrize('part',['manifest_sha256','games_sha256','samples_sha256','overlap_audit_sha256','executable_sha256','embedded_nnue'])
def test_bad_target_provenance(bundle,part):
    path=bundle['bundle']/'labels.jsonl'; rows=read_rows(path)
    rows[0]['target_provenance'][part] = [] if part=='embedded_nnue' else '0'*64
    write_rows(path,rows); seal_bundle(bundle['bundle'])
    with pytest.raises(ValueError,match='provenance'): open_dataset(**bundle,split='train')


def test_bad_heldout_label_is_checked_before_filter(bundle):
    path=bundle['bundle']/'labels.jsonl'; rows=read_rows(path)
    rows[-1]['encoding_version']='old'; write_rows(path,rows); seal_bundle(bundle['bundle'])
    with pytest.raises(ValueError,match='encoding_version'): open_dataset(**bundle,split='train')


@pytest.mark.parametrize('mutation',['interrupted','pending','missing_attempt','duplicate_label','duplicate_reference',
    'unselected_attempt','missing_label','failed_success','unfinished_time','bad_settings','bad_asset','bad_teacher'])
def test_unfinished_or_corrupt_bundle(bundle,mutation):
    out=bundle['bundle']
    if mutation in ('interrupted','pending'):
        summary=json.loads((out/'summary.json').read_text())
        if mutation=='interrupted': summary['status']='interrupted'
        else: summary['pending_references']=[{}]
        (out/'summary.json').write_text(json.dumps(summary))
    else:
        name = ('attempts.jsonl' if mutation in ('missing_attempt','unselected_attempt','failed_success','unfinished_time')
            else 'settings.json' if mutation in ('bad_settings','bad_asset','bad_teacher')
            else 'selected.jsonl' if mutation=='duplicate_reference' else 'labels.jsonl')
        if name=='settings.json':
            settings=json.loads((out/name).read_text())
            if mutation=='bad_settings': settings['requested_settings']['nodes']=1
            elif mutation=='bad_asset': settings['teacher']['embedded_nnue'][0]['sha256']='0'*64
            else: settings['teacher']['executable_identity']['sha256']='0'*64
            (out/name).write_text(json.dumps(settings))
        else:
            rows=read_rows(out/name)
            if mutation in ('missing_attempt','missing_label'): rows.pop(0)
            elif mutation.startswith('duplicate'): rows.append(deepcopy(rows[0]))
            elif mutation=='unselected_attempt': rows[0]['reference']['sample_ply']=100
            elif mutation=='failed_success': rows[0]['status']='failed'
            elif mutation=='unfinished_time': rows[0].pop('finished_at_utc')
            write_rows(out/name,rows)
        seal_bundle(out)
    with pytest.raises(ValueError): open_dataset(**bundle,split='train')


@pytest.mark.parametrize('name',['labels.jsonl','settings.json','samples.jsonl','games.jsonl','protected-heldout.jsonl','heldout-games.jsonl'])
def test_tamper_rejected(bundle,name):
    path = bundle['games_path'] if name=='games.jsonl' else (
        bundle['bundle']/name if name in BUNDLE_FILES else bundle['prepared']/name)
    with path.open('a') as f: f.write('tamper')
    with pytest.raises(ValueError): open_dataset(**bundle,split='train')


@pytest.mark.parametrize('fen,best',[
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1'),
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1c1'),
    ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1','e8g8'),
    ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1','e8c8'),
    ('k7/8/8/3pP3/8/8/8/4K3 w - d6 0 2','e5d6'),
    ('4k3/8/8/8/3Pp3/8/8/K7 b - d3 0 2','e4d3'),
    *[('k7/4P3/8/8/8/8/8/4K3 w - - 0 1','e7e8'+p) for p in 'qrbn'],
    *[('4k3/8/8/8/8/8/4p3/K7 b - - 0 1','e2e1'+p) for p in 'qrbn']])
def test_special_moves(tmp_path,fen,best):
    data=open_dataset(**fixture_bundle(tmp_path,[('train',fen,[],[0],best)]),split='train')
    sample=data[0]; board=chess.Board(fen); move=chess.Move.from_uci(best)
    origin,target=move.from_square,move.to_square
    if not board.turn: origin^=56; target^=56
    expected = 4096+4*(8*(origin%8)+target%8)+'qrbn'.index(best[-1]) if move.promotion else 64*origin+target
    assert sample.policy_class==expected and sample.legal_mask[expected]
    assert sample.state.board.fen(en_passant='fen')==chess.Board(sample.state.initial_fen).fen(en_passant='fen')


@pytest.mark.parametrize('fen,moves,ply',[
    ('4k3/8/8/8/8/8/8/4K3 w - - 0 1',[],0),
    ('4k3/8/8/8/8/8/8/R3K3 w - - 100 51',[],0),
    (chess.STARTING_FEN,['g1f3','g8f6','f3g1','f6g8']*2,8),
    (chess.STARTING_FEN,['f2f3','e7e5','g2g4','d8h4'],4)])
def test_terminal_rejected(tmp_path,fen,moves,ply):
    with pytest.raises(ValueError,match='Terminal'):
        open_dataset(**fixture_bundle(tmp_path,[('train',fen,moves,[ply],None)]),split='train')


def test_long_history_and_repetition(tmp_path):
    moves=['g1f3','g8f6','f3g1','f6g8','e2e4','e7e5','g1f3','b8c6','f1b5','a7a6']
    data=open_dataset(**fixture_bundle(tmp_path,[('train',chess.STARTING_FEN,moves,[4,10],None)]),split='train')
    early,later=data[0],data[1]
    assert len(later.state.board.move_stack)==10
    assert early.features[0,12]==1
    assert np.array_equal(later.features,encode_position(make_state(early.state.initial_fen,moves,history_complete=True)).features)


def test_completed_exclusion_and_retry(bundle):
    out=bundle['bundle']; labels=read_rows(out/'labels.jsonl'); attempts=read_rows(out/'attempts.jsonl')
    failed=deepcopy(attempts[0]); failed['status']='failed'; failed['error']={'type':'TimeoutError','message':'fixture'}
    succeeded=deepcopy(attempts[0]); succeeded['attempt']=2; labels[0]['successful_attempt']=2
    excluded=labels.pop(-1); last=attempts.pop(-1); last['status']='failed'; retry=deepcopy(last); retry['attempt']=2
    attempts=[failed,succeeded]+attempts[1:]+[last,retry]
    write_rows(out/'attempts.jsonl',attempts); write_rows(out/'labels.jsonl',labels)
    write_rows(out/'exclusions.jsonl',[dict(reference=last['reference'],reason='attempts_exhausted',attempts=2)])
    seal_bundle(out)
    assert len(open_dataset(**bundle,split='train'))==3
    assert len(open_dataset(**bundle,split='test'))==1


def test_empty_dataset(tmp_path):
    args=fixture_bundle(tmp_path,[('train',chess.STARTING_FEN,['e2e4'],[1],None)])
    data=open_dataset(**args,split='validation')
    assert len(data)==0 and list(data.batches(2))==[]


def test_optional_torch(bundle):
    torch=pytest.importorskip('torch')
    batch=next(open_dataset(**bundle,split='train').batches(2))
    tensors=batch.to_torch()
    for name in ('features','legal_mask','policy_classes','wdl'):
        assert np.array_equal(tensors[name].numpy(),getattr(batch,name))
    assert tensors['policy_classes'].dtype==torch.int64
    assert tensors['wdl'].dtype==torch.float32 and tensors['legal_mask'].dtype==torch.bool
    tensors['features'].zero_()
    assert np.any(batch.features)


def test_import_does_not_load_torch_or_model():
    code='import sys; import chess_rl.supervised; assert "torch" not in sys.modules; assert not any(k.startswith("chess_rl.model") for k in sys.modules)'
    subprocess.run([sys.executable,'-c',code],check=True)


def reseal_intake(args):
    """Adversarial self-consistent audit fixture: checks beyond SHA-256 alone."""
    prepared,out=args['prepared'],args['bundle']
    manifest=DatasetManifest.load(args['manifest_path'])
    split=json.loads(args['split_audit_path'].read_text())
    split['outputs']['games.jsonl']=file_identity(args['games_path'])
    args['split_audit_path'].write_text(json.dumps(split))
    audit=json.loads((prepared/'overlap-audit.json').read_text())
    audit['inputs']={'games.jsonl':file_identity(args['games_path']),'split-audit.json':file_identity(args['split_audit_path'])}
    audit['outputs']={name:file_identity(prepared/name) for name in audit['outputs']}
    (prepared/'overlap-audit.json').write_text(json.dumps(audit))
    manifest=replace(manifest,preparation=replace(manifest.preparation,
        split_audit_sha256=file_identity(args['split_audit_path'])['sha256'],
        overlap_audit_sha256=file_identity(prepared/'overlap-audit.json')['sha256'],
        protected_heldout_registry_sha256=file_identity(prepared/'protected-heldout.jsonl')['sha256']))
    args['manifest_path'].write_text(manifest.to_json())
    identities=dict(audit['inputs'],manifest=file_identity(args['manifest_path']),
        **{name:file_identity(prepared/name) for name in (*audit['outputs'],'overlap-audit.json')})
    settings=json.loads((out/'settings.json').read_text());settings['artifact_identities']=identities
    (out/'settings.json').write_text(json.dumps(settings))
    labels=read_rows(out/'labels.jsonl')
    for row in labels:
        for field,name in [('manifest_sha256','manifest'),('games_sha256','games.jsonl'),
            ('samples_sha256','samples.jsonl'),('overlap_audit_sha256','overlap-audit.json')]:
            row['target_provenance'][field]=identities[name]['sha256']
    write_rows(out/'labels.jsonl',labels);seal_bundle(out)


@pytest.mark.parametrize('mutation',['missing_sample','duplicate_sample','wrong_sample','protected_train_position',
    'heldout_train_game','missing_heldout_game','missing_heldout_position','missing_game','duplicate_game','wrong_game_history'])
def test_audited_join_and_split_isolation(bundle,mutation):
    prepared=bundle['prepared']
    refs=read_rows(prepared/'samples.jsonl')
    if mutation.endswith('sample'):
        path=prepared/'samples.jsonl'
        if mutation=='missing_sample': refs.pop(0)
        elif mutation=='duplicate_sample': refs.append(deepcopy(refs[0]))
        else: refs[0]['phase']='endgame'
        write_rows(path,refs)
    elif mutation in ('protected_train_position','missing_heldout_position'):
        path=prepared/'protected-heldout.jsonl';rows=read_rows(path)
        if mutation=='protected_train_position': rows.append(dict(refs[0],split='test'))
        else: rows.pop(0)
        write_rows(path,rows)
    elif mutation in ('heldout_train_game','missing_heldout_game'):
        path=prepared/'heldout-games.jsonl';rows=read_rows(path)
        if mutation=='heldout_train_game': rows.append(dict(game_id=refs[0]['game_id'],split='test'))
        else: rows.pop(0)
        write_rows(path,rows)
    else:
        path=bundle['games_path'];rows=read_rows(path)
        if mutation=='missing_game': rows.pop(0)
        elif mutation=='duplicate_game': rows.append(deepcopy(rows[0]))
        else: rows[0]['history_complete']=False
        write_rows(path,rows)
    reseal_intake(bundle)
    with pytest.raises(ValueError):open_dataset(**bundle,split='train')


@pytest.mark.parametrize('mutation',['failed_status','wrong_number','invalid_counts','invalid_perspective','illegal_pv','wdl_bool','missing_field','duplicate_json_field'])
def test_additional_record_corruption(bundle,mutation):
    out=bundle['bundle'];name='attempts.jsonl' if mutation in ('failed_status','wrong_number') else 'labels.jsonl'
    rows=read_rows(out/name)
    if mutation=='failed_status':rows[0]['status']='interrupted'
    elif mutation=='wrong_number':rows[0]['attempt']=True
    elif mutation=='invalid_counts':rows[0]['returned_search']['wdl']['relative']=[-1,200,801]
    elif mutation=='invalid_perspective':rows[0]['returned_search']['wdl']['turn']='unknown'
    elif mutation=='illegal_pv':rows[0]['returned_search']['pv'].append('e2e5')
    elif mutation=='wdl_bool':rows[0]['returned_search']['wdl']['relative']=[True,200,799]
    elif mutation=='missing_field':rows[0].pop('best_move_uci')
    if mutation=='duplicate_json_field':
        raw=[json.dumps(r) for r in rows];raw[0]=raw[0][:-1]+',"record_version":"stockfish-label-v1"}'
        (out/name).write_text('\n'.join(raw)+'\n')
        # Seal hashes manually: seal_bundle intentionally uses ordinary json,
        # loader uses duplicate-key rejecting parser.
    else:write_rows(out/name,rows)
    seal_bundle(out)
    with pytest.raises(ValueError):open_dataset(**bundle,split='train')


@pytest.mark.parametrize('mutation',[None,'history','reset','budget','exit','partial_shutdown'])
def test_raw_attempt_history(bundle,mutation):
    out=bundle['bundle'];rows=read_rows(out/'attempts.jsonl');ref=rows[0]['reference']
    game=next(g for g in read_rows(bundle['games_path']) if g['game_id']==ref['game_id'])
    start=game['start_fen']
    position='position startpos' if start==chess.STARTING_FEN else 'position fen '+start
    position+=' moves '+' '.join(game['moves_uci'][:ref['sample_ply']])
    returned=rows[0]['returned'];returned['uci_sent']=['setoption name Clear Hash',position,'go nodes 100000','quit'];returned['engine_exit_code']=0
    if mutation=='history':returned['uci_sent'][1]='position startpos'
    elif mutation=='reset':returned['uci_sent'].pop(0)
    elif mutation=='budget':returned['uci_sent'][2]='go nodes 10'
    elif mutation=='exit':returned['engine_exit_code']=1
    elif mutation=='partial_shutdown':returned['uci_sent'].pop()
    write_rows(out/'attempts.jsonl',rows);seal_bundle(out)
    if mutation:
        with pytest.raises(ValueError,match='Attempt'):open_dataset(**bundle,split='train')
    else:assert len(open_dataset(**bundle,split='train'))==3


@pytest.mark.parametrize('mutation',['numeric_best','object_pv','returned_not_object','numeric_row','numeric_command'])
def test_malformed_shapes_raise_value_error(bundle,mutation):
    out=bundle['bundle'];labels=read_rows(out/'labels.jsonl');attempts=read_rows(out/'attempts.jsonl')
    if mutation=='numeric_best':labels[0]['best_move_uci']=123
    elif mutation=='object_pv':labels[0]['returned_search']['pv']=[{}]
    elif mutation=='returned_not_object':attempts[0]['returned']=[]
    elif mutation=='numeric_row':attempts[0]['returned']['returned_rows']=[123]
    else:attempts[0]['returned']['uci_sent']=[123,'quit']
    write_rows(out/'labels.jsonl',labels);write_rows(out/'attempts.jsonl',attempts);seal_bundle(out)
    with pytest.raises(ValueError,match='Malformed|Unfinished|mismatch'):
        open_dataset(**bundle,split='train')
