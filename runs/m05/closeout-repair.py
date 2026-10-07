"""Resume authorized M05 closeout after the evidence-log EOF whitespace finding."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
OUT = ROOT/'runs/m05'
spec = importlib.util.spec_from_file_location('m05_audit', OUT/'final-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == audit.BASELINE
allowed = audit.ALLOWED | audit.NEW | {str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file()}
staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines())
assert staged <= allowed
raw = (OUT/'01-m05-check.log').read_text()
assert raw.endswith('\n\n')
(OUT/'01-m05-raw-output.json').write_text(json.dumps({
    'command_record': 'commands.json entry 0', 'stdout_verbatim': raw,
    'reason': 'Preserve exact process output while text log removes a redundant EOF blank line for git diff --check'},
    indent=2)+'\n')
(OUT/'01-m05-check.log').write_text(raw.rstrip('\n')+'\n')
commands = json.loads((OUT/'commands.json').read_text())
commands[0]['text_log_transform'] = 'Removed one redundant EOF newline; complete stdout preserved verbatim in 01-m05-raw-output.json'
commands[0]['raw_output'] = 'runs/m05/01-m05-raw-output.json'
(OUT/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
shutil.copyfile(__file__, OUT/'closeout-repair.py')
context = json.loads((OUT/'execution-context.json').read_text())
context['closeout_repair_command'] = ['python3', '/private/tmp/rl-chess-m05-closeout-repair.py']
context['closeout_repair'] = 'Initial staged diff check rejected redundant EOF blank in M05 checker log. Exact raw output saved in JSON; text log EOF normalized; no model code or numerical result changed.'
(OUT/'execution-context.json').write_text(json.dumps(context, indent=2)+'\n')
protected = audit.audit(ROOT)
files = audit.manifest(ROOT, protected)
subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
audit.recorded(['git', 'diff', '--cached', '--check'], ROOT, '18-final-staged-diff-check.log')
files = audit.manifest(ROOT, protected)
subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
actual = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
assert actual == files, (actual, files)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', audit.COMMIT_MESSAGE], cwd=ROOT, check=True)
status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=ROOT, text=True)
assert status == '', status
print('Clean working tree. Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
