"""Copy the small authoritative audit and resolve only the two D03 manifest fields."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT/'data/manifests/elite-2025-01-v1.json'
raw = Path('/private/tmp/rl-chess-d03-prepared/split-audit.json').read_bytes()
audit = json.loads(raw)
assert audit['complete'] is True
manifest = json.loads(path.read_text())
manifest['preparation']['split_algorithm'] = audit['split_algorithm']
manifest['preparation']['split_audit_sha256'] = hashlib.sha256(raw).hexdigest()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
(ROOT/'runs/d03/split-audit.json').write_bytes(raw)
print(json.dumps(dict(split_algorithm=audit['split_algorithm'], split_audit_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
