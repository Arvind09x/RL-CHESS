"""Independently check stored source locations/counts against all raw member frames."""
from collections import Counter
import hashlib,json,sys,zipfile
from pathlib import Path
from itertools import zip_longest
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import iter_frames,file_identity,replay_state
bulk=Path(sys.argv[1])
root=Path(__file__).resolve().parents[2]
summary=json.loads((bulk/'summary.json').read_text())
retrieval=json.loads((root/'runs/d02/archive-provenance.json').read_text())
manifest=DatasetManifest.load(root/'data/manifests/elite-2025-01-v1.json')
counts=Counter()
rows=[]
# Source ordinals in separate accepted/rejected tables must each be monotonic.
def read_rows(name):
 with (bulk/name).open() as h:
  previous=None
  for line in h:
   row=json.loads(line)
   key=(row['source']['member_index'],row['source']['game_ordinal'])
   assert previous is None or previous < key
   previous=key
   yield key,row
# Merge by source ordinal without sorting or accumulating bulk rows in RAM.
accepted=iter(read_rows('replays.jsonl'))
rejected=iter(read_rows('rejections.jsonl'))
a=next(accepted,None); r=next(rejected,None)
def merged():
 global a,r
 while a is not None or r is not None:
  take_a=r is None or (a is not None and a[0]<r[0])
  if take_a:
   yield a[0],True,a[1]
   a=next(accepted,None)
  else:
   yield r[0],False,r[1]
   r=next(rejected,None)
def raw_frames():
 with zipfile.ZipFile('/private/tmp/lichess_elite_2025-01.zip') as z:
  for index,info in enumerate(z.infolist(),1):
   if info.filename.lower().endswith('.pgn'):
    with z.open(info) as h:
     for ordinal,frame in enumerate(iter_frames(h),1):
      yield (index,ordinal),info.filename,frame
last_accepted=None
for raw,stored in zip_longest(raw_frames(),merged()):
 assert raw is not None and stored is not None
 key,name,frame=raw
 saved_key,is_accepted,row=stored
 assert key==saved_key
 s=row['source']
 assert s['archive_sha256']==manifest.archive.downloaded_sha256
 assert s['member']==name
 assert (s['byte_start'],s['byte_end'],s['line_start'],s['line_end'])==(frame.byte_start,frame.byte_end,frame.line_start,frame.line_end)
 assert s['source_sha256']==hashlib.sha256(frame.raw).hexdigest()
 counts['games']+=1
 counts['accepted' if is_accepted else 'rejected']+=1
 if is_accepted:
  assert not ({'game_id','split','sample_ply','teacher','wdl'} & row.keys())
  assert row['dataset_id']==manifest.dataset_id and row['history_complete'] is True
  counts['plies']+=len(row['moves_uci'])
  counts['unfinished']+=row['pgn_result']['unfinished']
  counts['result_discrepancies']+=row['pgn_result']['discrepancy']
  counts['declared_setups']+=row['declared_setup']
  last_accepted=row
  if counts['accepted'] in (1,1000,10000,100000,200000):
   state=replay_state(row)
   rows.append({'source':s,'plies':len(state.board.move_stack),'final_fen':state.board.fen(en_passant='fen'),'history_complete':state.history_complete})
for key,value in summary['counts'].items(): assert counts[key]==value,(key,counts[key],value)
for name,identity in summary['outputs'].items(): assert file_identity(bulk/name)==identity
for member in summary['members']:
 match=next(m for m in retrieval['members'] if m['member_index']==member['member_index'])
 for field in match: assert member[field]==match[field]
if last_accepted:
 state=replay_state(last_accepted)
 rows.append({'source':last_accepted['source'],'plies':len(state.board.move_stack),'final_fen':state.board.fen(en_passant='fen'),'history_complete':state.history_complete})
assert summary['complete'] and summary['source_game_count_exact']
print(json.dumps({'bulk_path':str(bulk),'counts':dict(counts),'all_source_locations_and_output_hashes_matched':True,'member_identity_matches_retrieval':True,'selected_record_roundtrips':rows,'annotation_blockers':list(manifest.annotation_blockers())},indent=2,sort_keys=True))
