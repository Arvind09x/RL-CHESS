"""Archive exact M06 stdout as a JSON string, including its EOF blank line.

The initial whitespace check flagged that raw log's extra EOF newline. JSON
preserves every captured byte without normalizing evidence to pass Git checks.
"""
import hashlib
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
path = directory/'m06.command.json'
command = json.loads(path.read_text())
output = command['outputs']['stdout']
if output.get('encoding') != 'utf8-json-string':
    raw = (directory/output['path']).read_bytes()
    assert len(raw) == output['bytes'] and hashlib.sha256(raw).hexdigest() == output['sha256']
    container = directory/'m06.stdout.json'
    container.write_text(json.dumps(raw.decode('utf-8'), ensure_ascii=True) + '\n')
    output.update(path=container.name, encoding='utf8-json-string',
                  container_bytes=container.stat().st_size,
                  container_sha256=hashlib.sha256(container.read_bytes()).hexdigest())
    command['output_storage_note'] = 'stdout bytes/sha256 refer to decoded JSON string; exact captured bytes retained'
    path.write_text(json.dumps(command, indent=2, sort_keys=True) + '\n')
