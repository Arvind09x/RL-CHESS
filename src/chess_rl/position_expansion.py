"""Verified append-only D04 expansion with immutable protected registries.

A new D03 table/audit and its accepted digest are prerequisites. Expansion
never reassigns old games or selects samples from newly assigned heldout games.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from chess_rl.canonical_games import canonical_json, load_json
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.position_sampling import (PHASES, RECORD_VERSION, candidates_for_game,
    PHASE_DESCRIPTION, SAMPLING_DESCRIPTION, SEED, sample_record, select,
    validate_game, verify_intake)


def expand_positions(games, split_audit, split_sha256, manifest, previous_dir,
                     previous_games, previous_audit_sha256, output_dir):
    """Verify all prior artifacts and replay roots before publishing an audit.

    Retain prior training samples too. Same-game capacity stays eight. Additions
    are sorted deterministically; registry bytes and assignments cannot change.
    All source occurrences must survive with their multiplicity. The candidate
    SQLite cache is never trusted as expansion evidence.
    """
    if (manifest.sampling_seed != SEED
            or manifest.preparation.sampling_algorithm not in (None, SAMPLING_DESCRIPTION)
            or manifest.preparation.phase_boundaries not in (None, PHASE_DESCRIPTION)):
        raise ValueError('Expansion sampling decisions changed')
    prior = Path(previous_dir)
    previous_audit = prior / 'overlap-audit.json'
    if file_identity(previous_audit)['sha256'] != previous_audit_sha256:
        raise ValueError('Previous overlap audit identity mismatch')
    old_audit = load_json(previous_audit.read_bytes())
    if old_audit.get('complete') is not True or old_audit.get('audit_version') not in (
            'position-overlap-audit-v1', 'position-expansion-audit-v1'):
        raise ValueError('Previous audit incomplete or unsupported')
    if file_identity(previous_games) != old_audit['inputs']['games.jsonl']:
        raise ValueError('Previous games identity mismatch')
    required = ('samples.jsonl', 'protected-heldout.jsonl', 'heldout-games.jsonl')
    identities = {name: file_identity(prior/name) for name in required}
    if any(identities[name] != old_audit['outputs'][name] for name in required):
        raise ValueError('Previous sample/registry identity mismatch')
    if identities['protected-heldout.jsonl']['sha256'] != manifest.preparation.protected_heldout_registry_sha256:
        raise ValueError('Protected registry differs from manifest freeze')
    new_audit, inputs = verify_intake(games, split_audit, split_sha256, manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protected = set()
    counts = Counter(existing_samples=0, new_training_samples=0, quarantined_new_heldout_games=0)
    phases = {s: Counter() for s in ('train','validation','test')}
    with tempfile.TemporaryDirectory(prefix='rl-chess-d04-expand-') as scratch:
        with sqlite3.connect(str(Path(scratch)/'expansion.sqlite')) as db:
            db.execute('CREATE TABLE old_games(id TEXT PRIMARY KEY, split TEXT, sources TEXT, seen INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE samples(id TEXT, ply INTEGER, split TEXT, record TEXT, PRIMARY KEY(id,ply))')
            with Path(previous_games).open('rb') as handle:
                for raw in handle:
                    g = load_json(raw)
                    validate_game(g, manifest)
                    db.execute('INSERT INTO old_games(id,split,sources) VALUES (?,?,?)',
                        (g['game_id'],g['split'],canonical_json(g['sources'])))
            heldout_digest = hashlib.sha256()
            with (prior/'samples.jsonl').open('rb') as handle:
                for raw in handle:
                    r = load_json(raw)
                    if r['record_version'] != RECORD_VERSION:
                        raise ValueError('Sample version mismatch')
                    split = db.execute('SELECT split FROM old_games WHERE id=?',(r['game_id'],)).fetchone()
                    if split is None or split[0] != r['split']:
                        raise ValueError('Previous sample assignment mismatch')
                    db.execute('INSERT INTO samples VALUES (?,?,?,?)',
                        (r['game_id'],r['sample_ply'],r['split'],canonical_json(r)))
                    counts['existing_samples'] += 1
                    if r['split'] != 'train':
                        heldout_digest.update(raw)
                        protected.add(r['rule_position'])
            if heldout_digest.hexdigest() != identities['protected-heldout.jsonl']['sha256']:
                raise ValueError('Protected samples differ from frozen registry')
            observed = hashlib.sha256()
            assignments = []
            game_counts = Counter()
            with Path(games).open('rb') as handle:
                for raw in handle:
                    observed.update(raw)
                    g = load_json(raw)
                    validate_game(g, manifest)
                    identity, split = g['game_id'], g['split']
                    game_counts[split] += 1
                    assignments.append((identity,split))
                    old = db.execute('SELECT split,sources,seen FROM old_games WHERE id=?',(identity,)).fetchone()
                    if old:
                        if old[0] != split or old[2]:
                            raise ValueError('Existing game assignment changed or repeated')
                        before = Counter(canonical_json(s) for s in json.loads(old[1]))
                        after = Counter(canonical_json(s) for s in g['sources'])
                        if before - after:
                            raise ValueError('Original source provenance removed')
                        db.execute('UPDATE old_games SET seen=1 WHERE id=?',(identity,))
                    elif split != 'train':
                        counts['quarantined_new_heldout_games'] += 1
                    # Revalidate all previous roots through full-history replay.
                    pool, _ = candidates_for_game(g)
                    prior_samples = [json.loads(row[0]) for row in db.execute(
                        'SELECT record FROM samples WHERE id=? ORDER BY ply',(identity,))]
                    locked = [(r['sample_ply'],r['phase'],r['rule_position']) for r in prior_samples]
                    if any(c not in pool for c in locked) or len(locked)>8 or len({c[2] for c in locked}) != len(locked):
                        raise ValueError('Prior roots changed, terminal or over limit')
                    if split == 'train':
                        chosen = select(identity, pool, protected, locked)
                        for c in chosen:
                            if c not in locked:
                                r = sample_record(identity, split, c)
                                db.execute('INSERT INTO samples VALUES (?,?,?,?)',
                                    (identity,c[0],split,canonical_json(r)))
                                counts['new_training_samples'] += 1
                    elif not old and prior_samples:
                        raise ValueError('New heldout game has samples')
            if observed.hexdigest() != inputs['games.jsonl']['sha256']:
                raise ValueError('Expansion games changed during processing')
            if db.execute('SELECT count(*) FROM old_games WHERE seen=0').fetchone()[0]:
                raise ValueError('Original games removed')
            digest = hashlib.sha256(''.join(json_line(dict(game_id=i,split=s)) for i,s in sorted(assignments)).encode()).hexdigest()
            if digest != new_audit['assignments_sha256'] or any(game_counts[s] != new_audit['actual_split_counts'][s]['games'] for s in game_counts):
                raise ValueError('Expansion assignment census mismatch')
            train_keys = set()
            with (out/'samples.jsonl').open('w') as handle:
                for split, raw in db.execute('SELECT split,record FROM samples ORDER BY id,ply'):
                    r = json.loads(raw)
                    phases[split][r['phase']] += 1
                    if split == 'train':
                        train_keys.add(r['rule_position'])
                    handle.write(raw+'\n')
            if train_keys & protected:
                raise ValueError('Training collides with protected heldout')
    for name in ('protected-heldout.jsonl','heldout-games.jsonl'):
        shutil.copyfile(prior/name,out/name)
    if (any(file_identity(prior/name) != identities[name] for name in required)
            or file_identity(previous_audit)['sha256'] != previous_audit_sha256
            or file_identity(previous_games) != old_audit['inputs']['games.jsonl']):
        raise ValueError('Previous artifacts changed during expansion')
    if file_identity(games) != inputs['games.jsonl'] or file_identity(split_audit) != inputs['split-audit.json']:
        raise ValueError('Expansion intake changed before completion')
    report = dict(audit_version='position-expansion-audit-v1', complete=True,
                  previous_audit_sha256=previous_audit_sha256, inputs=inputs,
                  assignments_sha256=digest, counts=dict(counts), cross_split_overlaps_after=0,
                  phase_after={s:{p:c[p] for p in PHASES} for s,c in phases.items()},
                  outputs={name:file_identity(out/name) for name in required},
                  protected_registry_unchanged=True, existing_game_assignments_unchanged=True,
                  source_provenance_preserved=True, labels_produced=0)
    (out/'overlap-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report
