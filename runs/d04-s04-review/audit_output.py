"""Full record/selection audit, with separately spelled D04 selector and identity.

Every game pool and final reference is checked. Legal/terminal pool membership
comes from the full preparation replay; a deterministic subset is independently
replayed again through make_state. No production D04 selector is imported.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

from chess_rl.adapter import make_state, terminal_result
from chess_rl.pgn_replay import file_identity, json_line

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
audit = json.loads((out/'overlap-audit.json').read_text())
assert audit['complete']
for name, identity in audit['outputs'].items():
    assert file_identity(out/name) == identity, name
assert file_identity('/private/tmp/rl-chess-d03-prepared/games.jsonl') == audit['inputs']['games.jsonl']
assert file_identity('/private/tmp/rl-chess-d03-prepared/split-audit.json') == audit['inputs']['split-audit.json']
phases = ('opening','middlegame','endgame')


def choose(identity, pool, blocked=(), fixed=()):
    result = list(fixed)
    options = [x for x in pool if x not in result and x[2] not in blocked]
    def digest(x):
        raw = json.dumps(dict(game_id=identity,sample_ply=x[0],sampling_version='phase-spread-sha256-v1',seed=20261009),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
        return hashlib.sha256(raw).digest(),x[0]
    hashes = {x[0]: digest(x) for x in options}
    def tie(x):
        return hashes[x[0]]
    for phase in phases:
        candidates = [x for x in options if x[1]==phase]
        if len(result)<8 and candidates and not any(x[1]==phase for x in result):
            twice_center = min(x[0] for x in candidates)+max(x[0] for x in candidates)
            item = min(candidates,key=lambda x:(abs(2*x[0]-twice_center),tie(x)))
            result.append(item)
            options.remove(item)
    while options and len(result)<8:
        item = min(options,key=lambda x:(-min(abs(x[0]-y[0]) for y in result),tie(x)))
        result.append(item)
        options.remove(item)
    return sorted(result)


db = sqlite3.connect(f'file:{out}/candidate-pools.sqlite?mode=ro',uri=True)
assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
protected, baseline_keys = set(), {}
ownership = {}
exclusion_digest = hashlib.sha256()
covered = Counter()
for split in ('test','validation','train'):
    keys = set()
    initial_keys = set()
    c = Counter(candidate_positions=0,initial_samples=0,removed=0,refilled=0,retained=0,
                games_covered=0,games_empty=0,blocked_candidate_positions=0)
    before, after, pc = Counter(),Counter(),Counter()
    excluded_owners = Counter()
    for identity, blob in db.execute('SELECT game_id,candidates FROM pools WHERE split=? ORDER BY game_id',(split,)):
        pool = [tuple(x) for x in json.loads(zlib.decompress(blob))]
        assert pool == sorted(pool)
        assert len({x[0] for x in pool}) == len({x[2] for x in pool}) == len(pool)
        initial = choose(identity,pool)
        surviving = [x for x in initial if x[2] not in protected]
        expected = choose(identity,pool,protected,surviving)
        for ply, name, key in initial:
            if key in protected:
                owner = ownership[key]
                excluded_owners[owner] += 1
                exclusion_digest.update(json_line(dict(record_version='position-reference-v1',
                    game_id=identity,split=split,sample_ply=ply,phase=name,
                    rule_position=key,owner=owner)).encode())
        actual = [tuple(x) for x in db.execute('SELECT sample_ply,phase,rule_position FROM selected WHERE game_id=? ORDER BY sample_ply',(identity,))]
        assert actual == expected, identity
        assert not {x[2] for x in actual} & protected
        keys.update(x[2] for x in actual)
        initial_keys.update(x[2] for x in initial)
        c.update(candidate_positions=len(pool),initial_samples=len(initial),removed=len(initial)-len(surviving),
                 refilled=len(actual)-len(surviving),retained=len(actual),games_covered=bool(actual),games_empty=not actual,
                 blocked_candidate_positions=sum(x[2] in protected for x in pool))
        before.update(x[1] for x in initial)
        after.update(x[1] for x in actual)
        pc.update(x[1] for x in pool)
    s = audit['splits'][split]
    assert all(s[k]==v for k,v in c.items())
    assert s['phase_before']=={p:before[p] for p in phases}
    assert s['phase_after']=={p:after[p] for p in phases}
    assert s['phase_candidates']=={p:pc[p] for p in phases}
    assert s['distinct_rule_positions']==len(keys)
    assert s['exclusions_by_owner']==dict(excluded_owners)
    baseline_keys[split] = initial_keys
    covered[split] = c['games_covered']
    if split != 'train':
        protected.update(keys)
        ownership.update({key:split for key in keys})
    print('Audited',split,c['retained'],flush=True)
# Check every output reference against the audited selection, exact ordering and schema.
expected_digest = hashlib.sha256()
heldout_digest = hashlib.sha256()
for identity,split,ply,phase,key in db.execute('SELECT * FROM selected ORDER BY game_id,sample_ply'):
    record = dict(record_version='position-reference-v1',game_id=identity,split=split,sample_ply=ply,phase=phase,rule_position=key)
    raw = json_line(record).encode()
    expected_digest.update(raw)
    if split != 'train':
        heldout_digest.update(raw)
assert expected_digest.hexdigest()==audit['outputs']['samples.jsonl']['sha256']
assert heldout_digest.hexdigest()==audit['outputs']['protected-heldout.jsonl']['sha256']
assert exclusion_digest.hexdigest()==audit['outputs']['exclusions.jsonl']['sha256']
game_registry = hashlib.sha256()
for identity,split in db.execute("SELECT game_id,split FROM pools WHERE split!='train' ORDER BY game_id"):
    game_registry.update(json_line(dict(game_id=identity,split=split)).encode())
assert game_registry.hexdigest()==audit['outputs']['heldout-games.jsonl']['sha256']
assert audit['protected_rule_positions']==len(protected)
assert db.execute('SELECT count(*) FROM (SELECT rule_position FROM selected GROUP BY rule_position HAVING count(DISTINCT split)>1)').fetchone()[0]==0
# Every exclusion and owner was independently derived in priority order above,
# then the complete canonical exclusion stream was compared by SHA-256.
# Fresh adapter root replay: every 991st canonical game, including terminal policy history.
replayed_games = replayed_roots = 0
with Path('/private/tmp/rl-chess-d03-prepared/games.jsonl').open() as handle:
    for index,raw in enumerate(handle):
        if index % 991:
            continue
        g = json.loads(raw)
        for ply,name,key in db.execute('SELECT sample_ply,phase,rule_position FROM selected WHERE game_id=? ORDER BY sample_ply',(g['game_id'],)):
            state = make_state(g['start_fen'],g['moves_uci'][:ply],history_complete=True)
            assert terminal_result(state) is None
            assert ' '.join(state.board.fen(en_passant='legal').split()[:4])==key
            material = sum(v*len(state.board.pieces(piece,color)) for piece,v in ((2,3),(3,3),(4,5),(5,9)) for color in (True,False))
            expected_phase = 'endgame' if material<=14 else 'opening' if ply<20 else 'middlegame'
            assert expected_phase==name
            replayed_roots += 1
        replayed_games += 1
before_pairs = {a+'-'+b:len(baseline_keys[a]&baseline_keys[b]) for a,b in
                (('train','validation'),('train','test'),('validation','test'))}
report = dict(all_game_pools_and_sample_references_verified=True, separately_spelled_selection_matched=True,
    pairwise_distinct_overlap_identities_before=before_pairs,cross_split_overlaps_after=0,
    actual_splits=audit['splits'],distinct_game_coverage=dict(covered),
    replayed_games=replayed_games,replayed_sample_roots=replayed_roots,
    independent_replay_scope='every 991st ID-sorted game; full legal replay and policy checks for all roots were performed by preparation',
    initial_10000_capacity_sufficient=all(audit['splits'][s]['retained']>=n for s,n in audit['initial_10000_quota'].items()),
    available_capacity=audit['capacity_records'], protected_registry_sha256=audit['outputs']['protected-heldout.jsonl']['sha256'],
    candidate_cache_identity=file_identity(out/'candidate-pools.sqlite'),
    inputs_preserved=True,outputs=audit['outputs'])
(ROOT/'runs/d04-s04-review/output-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
