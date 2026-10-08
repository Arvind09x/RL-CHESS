"""Compare both complete D03 outputs byte-for-byte without loading bulk tables."""
import hashlib
import json
from pathlib import Path

directories = [Path('/private/tmp/rl-chess-d03-prepared'), Path('/private/tmp/rl-chess-d03-reproduced')]
report = {}
for name in ('games.jsonl', 'split-audit.json'):
    digest = hashlib.sha256()
    size = 0
    with (directories[0]/name).open('rb') as a, (directories[1]/name).open('rb') as b:
        while True:
            left, right = a.read(1024*1024), b.read(1024*1024)
            assert left == right, name
            if not left:
                break
            digest.update(left)
            size += len(left)
    report[name] = dict(byte_identical=True, bytes=size, sha256=digest.hexdigest())
assert (directories[0]/'split-audit.json').read_bytes() == (Path(__file__).parent/'split-audit.json').read_bytes()
(Path(__file__).parent/'reproduction.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2, sort_keys=True))
