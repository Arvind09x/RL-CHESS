"""Check pinned teacher/source/embedded NNUE bytes without building or searching."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'data/manifests/elite-2025-01-v1.json').read_text())
provenance=json.loads((root/'runs/d01/stockfish-provenance.json').read_text())
binary=Path(provenance['executable']).read_bytes()
assert hashlib.sha256(binary).hexdigest()==manifest['engine']['executable_sha256']
source=Path('/private/tmp/rl-chess-stockfish-17.1-source.tar.gz')
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['engine']['package_sha256']
report={'executable':provenance['executable'],'executable_sha256':hashlib.sha256(binary).hexdigest(),'source_package':str(source),'source_package_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'embedded_nnue':[],'rebuilt':False,'search_commands_sent':False}
for asset in provenance['nnue']:
 path=Path('/private/tmp/Stockfish-sf_17.1/src')/asset['name']
 blob=path.read_bytes()
 assert hashlib.sha256(blob).hexdigest()==asset['sha256']
 assert binary[asset['embedded_offset']:asset['embedded_offset']+len(blob)]==blob
 report['embedded_nnue'].append(asset)
print(json.dumps(report,indent=2,sort_keys=True))
