"""D04 history, rule identity, deterministic coverage, refill and freezing."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import zlib

import chess
import pytest

from chess_rl.adapter import encode_position, make_state, terminal_result
from chess_rl.canonical_games import assign_split, canonical_payload, game_id
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl import position_sampling as d04

ROOT = Path(__file__).resolve().parents[1]


def game(moves=(), fen=chess.STARTING_FEN):
    manifest = DatasetManifest.load(ROOT / 'data/manifests/elite-2025-01-v1.json')
    identity = game_id(canonical_payload(fen, list(moves)))
    return dict(game_id=identity, split=assign_split(identity, manifest.split_seed),
                start_fen=fen, moves_uci=list(moves), history_complete=True,
                history_origin='declared-pgn-start', record_version='replay-jsonl-v1',
                dataset_id=manifest.dataset_id, encoding_version=manifest.encoding_version,
                move_vocab_version=manifest.move_vocab_version, ending_policy=manifest.ending_policy,
                wdl_order=list(manifest.wdl_order), sources=[{'pgn_result': 'provenance only'}])


def pool(size=30):
    return [(i, d04.PHASES[min(i//10, 2)], f'position-{i}') for i in range(size)]


def test_rule_identity_and_history_are_separate():
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
    first, second, third = [make_state(moves_uci=moves*n) for n in (0, 1, 2)]
    assert len({d04.rule_position(s.board) for s in (first, second, third)}) == 1
    assert terminal_result(first) is None and terminal_result(second) is None
    assert terminal_result(third).ending_reason == 'THREEFOLD_REPETITION'
    assert not (encode_position(first).features == encode_position(second).features).all()
    assert game(moves)['game_id'] != game(moves*2)['game_id']


@pytest.mark.parametrize('fen,available', [
    ('k7/8/8/3pP3/8/8/8/4K3 w - d6 0 2', True),
    ('k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', False),
    ('rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2', False)])
def test_legal_and_pinned_ep(fen, available):
    board = chess.Board(fen)
    no_ep = chess.Board(fen.replace(fen.split()[3], '-'))
    assert (d04.rule_position(board) != d04.rule_position(no_ep)) == available


@pytest.mark.parametrize('field,value,equal', [(4, '99', True), (5, '43', True),
                                              (1, 'b', False), (2, '-', False)])
def test_rule_counter_turn_castling(field, value, equal):
    fields = chess.STARTING_FEN.split()
    fields[field] = value
    assert (d04.rule_position(chess.Board(' '.join(fields))) == d04.rule_position(chess.Board())) == equal


@pytest.mark.parametrize('fen,ply,name', [(chess.STARTING_FEN, 19, 'opening'),
    (chess.STARTING_FEN, 20, 'middlegame'), ('4k3/8/8/8/8/8/8/R3K3 w - - 0 1', 0, 'endgame'),
    ('r3k3/8/8/8/8/8/8/R3K3 w - - 0 1', 40, 'endgame'),
    ('r3k3/8/8/8/8/8/8/3QK3 w - - 0 1', 40, 'endgame'),
    ('r3k3/8/8/8/8/8/8/RR2K3 w - - 0 1', 40, 'middlegame')])
def test_phase_boundary(fen, ply, name):
    assert d04.phase(chess.Board(fen), ply) == name


@pytest.mark.parametrize('fen', ['4k3/8/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/8/R3K3 w - - 100 51', '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1',
    '7k/5Q2/5K2/8/8/8/8/8 b - - 0 1'])
def test_terminal_start_excluded(fen):
    candidates, counts = d04.candidates_for_game(game(fen=fen))
    assert candidates == [] and counts['terminal_roots'] == 1


def test_counter_terminal_difference_does_not_hide_overlap():
    first = make_state('4k3/8/8/8/8/8/8/R3K3 w - - 0 1', history_complete=True)
    later = make_state('4k3/8/8/8/8/8/8/R3K3 w - - 100 51', history_complete=True)
    assert d04.rule_position(first.board)==d04.rule_position(later.board)
    assert terminal_result(first) is None
    assert terminal_result(later).ending_reason=='FIFTY_MOVES'
    assert game(fen=first.initial_fen)['game_id'] != game(fen=later.initial_fen)['game_id']


def test_repetition_excludes_suffix_but_replays_source():
    cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
    candidates, counts = d04.candidates_for_game(game(cycle*2 + ['e2e4', 'e7e5']))
    assert [c[0] for c in candidates] == [0, 1, 2, 3]
    assert counts == dict(roots=11, terminal_roots=1, post_terminal_roots=2, repeated_roots=4)
    # A next-move claim is not actual occurrence: prefix 7 remains nonterminal.
    assert terminal_result(make_state(moves_uci=cycle + cycle[:3])) is None
    with pytest.raises(ValueError):
        d04.candidates_for_game(game(cycle*2 + ['e2e5']))


def test_checkmate_final_root_excluded():
    candidates, counts = d04.candidates_for_game(game(['f2f3', 'e7e5', 'g2g4', 'd8h4']))
    assert [c[0] for c in candidates] == [0, 1, 2, 3]
    assert counts['terminal_roots'] == 1


def test_spread_coverage_eight_limit_reordering_and_refill():
    candidates = pool()
    selected = d04.select('fixed', candidates)
    assert len(selected) == len({c[0] for c in selected}) == len({c[2] for c in selected}) == 8
    assert {c[1] for c in selected} == set(d04.PHASES)
    assert min(c[0] for c in selected) <= 3 and max(c[0] for c in selected) >= 27
    random.Random(12).shuffle(candidates)
    assert selected == d04.select('fixed', candidates)
    blocked = {selected[0][2], selected[-1][2]}
    survivors = [c for c in selected if c[2] not in blocked]
    refill = d04.select('fixed', candidates, blocked, survivors)
    assert len(refill) == 8 and all(c in refill for c in survivors)
    assert all(c[2] not in blocked for c in refill)
    assert d04.select('fixed', candidates, {c[2] for c in candidates}) == []


def corpus(tmp_path, rows):
    tmp_path.mkdir()
    db = sqlite3.connect(tmp_path / 'pools.sqlite')
    db.execute('CREATE TABLE pools(game_id TEXT, split TEXT, candidates BLOB)')
    for identity, split, candidates in rows:
        db.execute('INSERT INTO pools VALUES (?,?,?)', (identity, split, zlib.compress(json_line(candidates).encode())))
    db.commit()
    report = d04.select_corpus(db, tmp_path)
    db.close()
    return report, [json.loads(line) for line in (tmp_path/'samples.jsonl').read_text().splitlines()]


def test_ownership_pairwise_separation_refill_and_order(tmp_path):
    rows = [('c', 'train', pool()), ('b', 'validation', pool()), ('a', 'test', pool())]
    a, samples = corpus(tmp_path/'a', rows)
    b, again = corpus(tmp_path/'b', list(reversed(rows)))
    assert samples == again and a == b
    keys = {s: {r['rule_position'] for r in samples if r['split'] == s} for s in d04.SPLITS}
    assert not keys['test'] & keys['validation'] and not keys['train'] & (keys['test'] | keys['validation'])
    assert a['splits']['validation']['removed'] > 0 and a['splits']['validation']['refilled'] > 0
    assert a['splits']['train']['retained'] == 8
    for name in ('samples.jsonl', 'protected-heldout.jsonl', 'heldout-games.jsonl', 'exclusions.jsonl'):
        assert (tmp_path/'a'/name).read_bytes() == (tmp_path/'b'/name).read_bytes()


def test_no_possible_refill_records_empty_games(tmp_path):
    report, samples = corpus(tmp_path/'a', [('a','test',pool(3)), ('b','validation',pool(3)), ('c','train',pool(3))])
    assert len(samples) == 3
    assert report['splits']['train']['games_empty'] == 1
    assert report['splits']['validation']['retained'] == 0


def test_expansion_is_append_only_and_protects_all_histories(tmp_path):
    _, old = corpus(tmp_path/'a', [('a','test',pool(3)), ('b','validation',pool(6)), ('c','train',pool())])
    heldout = [r for r in old if r['split'] != 'train']
    assignments = dict(a='test', b='validation', c='train', d='train', e='validation')
    new = dict(c=pool(), d=pool()+[(40,'endgame','new-40')], e=pool())
    result = d04.expand_training(new, old, heldout, assignments)
    assert all(r in result for r in old)
    assert [r for r in result if r['split'] != 'train'] == heldout
    assert any(r['game_id'] == 'd' for r in result)
    assert not any(r['game_id'] == 'e' for r in result)
    assert max(Counter(r['game_id'] for r in result).values()) <= 8
    assert result == d04.expand_training(dict(reversed(list(new.items()))), list(reversed(old)), heldout, assignments)
    assert result == d04.expand_training(new, result, heldout, assignments)
    with pytest.raises(ValueError, match='assignment'):
        d04.expand_training(new, old, heldout, dict(assignments, a='train'))
    with pytest.raises(ValueError, match='Protected'):
        d04.expand_training(new, old, heldout[:-1], assignments)


def intake(tmp_path, games):
    manifest = DatasetManifest.load(ROOT/'data/manifests/elite-2025-01-v1.json')
    path = tmp_path/'games.jsonl'
    path.write_text(''.join(json_line(g) for g in games))
    audit = json.loads((ROOT/'tests/fixtures/provenance/split-audit.json').read_text())
    audit['outputs']['games.jsonl'] = file_identity(path)
    audit['actual_split_counts'] = {s: dict(games=sum(g['split']==s for g in games),
        plies=sum(len(g['moves_uci']) for g in games if g['split']==s),
        sources=sum(len(g['sources']) for g in games if g['split']==s)) for s in d04.SPLITS}
    audit['assignments_sha256'] = hashlib.sha256(''.join(json_line(dict(game_id=g['game_id'],split=g['split'])) for g in sorted(games,key=lambda g:g['game_id'])).encode()).hexdigest()
    audit_path = tmp_path/'split-audit.json'
    audit_path.write_text(json.dumps(audit))
    digest = file_identity(audit_path)['sha256']
    from dataclasses import replace
    manifest = replace(manifest, preparation=replace(manifest.preparation, split_audit_sha256=digest))
    return path, audit_path, digest, manifest


def test_preparation_reproducible_reordered_and_tamper_closed(tmp_path):
    games = [game(['e2e4','e7e5']), game(['d2d4','d7d5']), game(['g1f3','g8f6'])]
    inputs = intake(tmp_path, games)
    a = d04.prepare_positions(*inputs, tmp_path/'a', workers=1)
    b = d04.prepare_positions(*inputs, tmp_path/'b', workers=2)
    assert a == b
    for name in a['outputs']:
        assert (tmp_path/'a'/name).read_bytes() == (tmp_path/'b'/name).read_bytes()
    reordered = tmp_path/'reordered'
    reordered.mkdir()
    c = d04.prepare_positions(*intake(reordered, list(reversed(games))), tmp_path/'c', workers=1)
    assert a['outputs'] == c['outputs'] and a['splits'] == c['splits']
    inputs[0].write_text(inputs[0].read_text()+'\n')
    with pytest.raises(ValueError, match='identity'):
        d04.prepare_positions(*inputs, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('field,value', [('history_complete',False), ('ending_policy','bad'),
    ('game_id','g:'+'a'*64), ('split','invalid'), ('record_version','bad')])
def test_game_contract_fail_closed(field, value):
    record = game(['e2e4'])
    record[field] = value
    with pytest.raises(ValueError):
        d04.validate_game(record, DatasetManifest.load(ROOT/'data/manifests/elite-2025-01-v1.json'))


def find_game(split, start=0, moves=None):
    for i in range(start, start+1000):
        fields = chess.STARTING_FEN.split()
        fields[5] = str(i+1)
        g = game(moves or ['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6'], ' '.join(fields))
        if g['split']==split:
            return g
    raise AssertionError('Fixture split not found')


def expansion_setup(tmp_path):
    from dataclasses import replace
    baseline_games = [find_game(s) for s in d04.SPLITS]
    before = tmp_path/'before'
    before.mkdir()
    inputs = intake(before, baseline_games)
    prior = tmp_path/'prior'
    d04.prepare_positions(*inputs, prior, workers=1)
    prior_audit_hash = file_identity(prior/'overlap-audit.json')['sha256']
    training = find_game('train', 1000,
        ['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6','b5a4','g8f6'])
    training['sources'].append({'additional': 'source multiplicity retained'})
    heldout = find_game('validation', 1000)
    new_games = baseline_games+[training,heldout]
    after = tmp_path/'after'
    after.mkdir()
    new_inputs = list(intake(after,new_games))
    new_inputs[3] = replace(new_inputs[3], preparation=replace(new_inputs[3].preparation,
        protected_heldout_registry_sha256=file_identity(prior/'protected-heldout.jsonl')['sha256']))
    return inputs, prior, prior_audit_hash, new_inputs, baseline_games, new_games


def test_expansion_file_pipeline_frozen_bytes_reorder_and_no_targets(tmp_path):
    from chess_rl.position_expansion import expand_positions
    old_inputs, prior, prior_hash, new_inputs, _, games = expansion_setup(tmp_path)
    report = expand_positions(*new_inputs,prior,old_inputs[0],prior_hash,tmp_path/'expanded')
    assert report['protected_registry_unchanged'] and report['existing_game_assignments_unchanged']
    assert report['counts']['quarantined_new_heldout_games']==1
    assert report['counts']['new_training_samples'] > 0
    for name in ('protected-heldout.jsonl','heldout-games.jsonl'):
        assert (prior/name).read_bytes()==(tmp_path/'expanded'/name).read_bytes()
    old = (prior/'samples.jsonl').read_text().splitlines()
    new = (tmp_path/'expanded'/'samples.jsonl').read_text().splitlines()
    assert set(old) <= set(new)
    assert all(not {'wdl','target','best_move_uci','result'} & set(json.loads(r)) for r in new)
    reordered = tmp_path/'reorder'
    reordered.mkdir()
    again = list(intake(reordered,list(reversed(games))))
    from dataclasses import replace
    again[3] = replace(again[3],preparation=replace(again[3].preparation,
        protected_heldout_registry_sha256=file_identity(prior/'protected-heldout.jsonl')['sha256']))
    second = expand_positions(*again,prior,old_inputs[0],prior_hash,tmp_path/'expanded-again')
    assert report['outputs']==second['outputs']
    repeated = expand_positions(*new_inputs,tmp_path/'expanded',new_inputs[0],
        file_identity(tmp_path/'expanded'/'overlap-audit.json')['sha256'],tmp_path/'expanded-twice')
    assert repeated['counts']['new_training_samples']==0
    assert repeated['outputs']==report['outputs']


@pytest.mark.parametrize('mutation', ['registry', 'samples', 'previous_hash', 'removed_game', 'source'])
def test_expansion_pipeline_fails_closed(tmp_path, mutation):
    from chess_rl.position_expansion import expand_positions
    old_inputs, prior, prior_hash, new_inputs, baseline, games = expansion_setup(tmp_path)
    if mutation in ('registry','samples'):
        path = prior/('protected-heldout.jsonl' if mutation=='registry' else 'samples.jsonl')
        path.write_bytes(path.read_bytes()+b'\n')
    elif mutation=='previous_hash':
        prior_hash='0'*64
    else:
        if mutation=='removed_game':
            games = [g for g in games if g['game_id'] != baseline[0]['game_id']]
        else:
            games[0] = deepcopy(games[0])
            games[0]['sources']=[]
        changed = tmp_path/'changed'
        changed.mkdir()
        replacement = list(intake(changed,games))
        from dataclasses import replace
        replacement[3] = replace(replacement[3],preparation=replace(replacement[3].preparation,
            protected_heldout_registry_sha256=file_identity(prior/'protected-heldout.jsonl')['sha256']))
        new_inputs = replacement
    with pytest.raises(ValueError):
        expand_positions(*new_inputs,prior,old_inputs[0],prior_hash,tmp_path/'rejected')
    assert not (tmp_path/'rejected'/'overlap-audit.json').exists()
