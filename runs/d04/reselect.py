"""Repeat complete-corpus selection from replayed candidate pools, read-only input.

This verifies selection reproducibility at corpus scale; it is not a second
complete replay. Authoritative samples, registry, freeze and exclusions must
match byte-for-byte, including deterministic refill.
"""
import json
from pathlib import Path
import sqlite3
import sys
from chess_rl.pgn_replay import file_identity
from chess_rl.position_sampling import select_corpus

ROOT = Path(__file__).resolve().parents[2]
source = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True,exist_ok=False)
prior = json.loads((source/'overlap-audit.json').read_text())
source_db = sqlite3.connect(f'file:{source}/candidate-pools.sqlite?mode=ro',uri=True)
db = sqlite3.connect(out/'candidate-pools.sqlite')
db.execute('CREATE TABLE pools(game_id TEXT PRIMARY KEY,split TEXT,candidates BLOB)')
db.executemany('INSERT INTO pools VALUES (?,?,?)',source_db.execute('SELECT game_id,split,candidates FROM pools ORDER BY game_id DESC'))
db.execute('CREATE INDEX by_split ON pools(split,game_id)')
db.commit()
report = select_corpus(db,out,progress=lambda n:print(n,flush=True))
assert report['splits']==prior['splits']
identities = {}
for name in prior['outputs']:
    identities[name]=file_identity(out/name)
    assert identities[name]==prior['outputs'][name],name
result = dict(full_corpus_selection_repeated=True,full_replay_repeated=False,
              insertion_order='reverse game_id',outputs_byte_identical=True,
              splits_identical=True,outputs=identities)
(ROOT/'runs/d04/reproduction.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
db.close()
source_db.close()
