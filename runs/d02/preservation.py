"""Assert all baseline implementation, planning contracts and historical evidence bytes."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
base='0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f'
allowed={'README.md','coding-plan.md','data/manifests/elite-2025-01-v1.json','tests/test_data_manifest.py'}
names=subprocess.check_output(['git','ls-tree','-r','--name-only',base],cwd=root,text=True).splitlines()
records=[]
for name in names:
 if name in allowed: continue
 original=subprocess.check_output(['git','show',f'{base}:{name}'],cwd=root)
 current=(root/name).read_bytes()
 assert original==current,name
 records.append({'path':name,'sha256':hashlib.sha256(current).hexdigest()})
print(json.dumps({'base':base,'protected_files':records,'protected_count':len(records),'allowed_modifications':sorted(allowed),'all_matched':True},indent=2,sort_keys=True))
