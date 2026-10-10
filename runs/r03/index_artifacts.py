"""Index reviewable R03 evidence and implementation without hashing this index."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/r03'

def main():
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != 'artifact-index.json']
    paths += [ROOT/p for p in ('README.md','coding-plan.md','docs/r03-mixed-update.md',
        'src/chess_rl/supervised.py','src/chess_rl/training/checkpoints.py',
        'src/chess_rl/training/losses.py','src/chess_rl/training/mixed.py',
        'src/chess_rl/training/mixed_state.py','tests/test_mixed_update.py','scripts/run_r03.py')]
    values = {}
    for path in sorted(paths):
        data = path.read_bytes()
        values[str(path.relative_to(ROOT))] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    (OUT/'artifact-index.json').write_text(json.dumps(values,sort_keys=True,indent=2)+'\n')
    print(f'Indexed {len(values)} R03 files')

if __name__ == '__main__': main()
