"""Join completed corpus/run evidence; no dataset processing algorithms."""
from pathlib import Path
import hashlib,json,subprocess
from chess_rl.pgn_replay import file_identity
root=Path(__file__).resolve().parents[2]
bulks=[Path('/private/tmp/rl-chess-d02-prepared'),Path('/private/tmp/rl-chess-d02-final-prepared'),Path('/private/tmp/rl-chess-d02-accepted-prepared')]
reports=[json.loads((b/'summary.json').read_text()) for b in bulks]
assert all(r==reports[-1] for r in reports)
for bulk,report in zip(bulks,reports):
 assert report['complete'] and report['source_game_count_exact']
 for name,expected in report['outputs'].items():
  assert file_identity(bulk/name)==expected
census=json.loads((root/'runs/d02/source-census.log').read_text())
assert reports[-1]['counts']['games']==census['raw_census']['event_headers']
assert reports[-1]['members'][0]['sha256']==census['member_sha256']
(root/'runs/d02/replay-summary.json').write_text(json.dumps(reports[-1],indent=2,sort_keys=True)+'\n')
evidence={'authoritative_bulk_path':str(bulks[-1]),'comparison_bulk_paths':[str(b) for b in bulks],'all_full_preparation_outputs_byte_identical':True,'preliminary_passes':'first began before move-number checks; second before orphan/empty variation checks; final strict pass uses the final package code','counts':reports[-1]['counts'],'archive':reports[-1]['archive'],'outputs':reports[-1]['outputs'],'independent_raw_event_census_matches_source_frame_count':True,'package_code_sha256':hashlib.sha256((root/'src/chess_rl/pgn_replay.py').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((root/'data/manifests/elite-2025-01-v1.json').read_bytes()).hexdigest(),'annotation_blockers':reports[-1]['annotation_blockers'],'D02_independent_review_accepted':False}
(root/'runs/d02/preparation-evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
print(json.dumps(evidence,indent=2,sort_keys=True))
