"""Preserve checker stdout, fix log EOF whitespace and finish the M06 commit.

The initial final-audit stopped before committing. This continuation verifies
its exact focused tree and preserves its failed command/log. Model/checker/test
sources and acceptance results are unchanged.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
STAGE = Path('/private/tmp/rl-chess-m06-change')
BASELINE = '2e6316dd0c49b0c22ff10ef72b55793856189a72'


def main():
    spec = importlib.util.spec_from_file_location('m06_audit', STAGE/'runs/m06/final-audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.git('rev-parse', 'HEAD').decode().strip() == BASELINE
    protected = audit.audit(SOURCE)
    for file in audit.ALLOWED | audit.NEW:
        assert (SOURCE/file).read_bytes() == (STAGE/file).read_bytes(), file
    # Check the initial focused archive exactly; permit only its three closeout logs.
    initial = set(json.loads((STAGE/'runs/m06/change-manifest.json').read_text())['changed_files'])
    expected = initial | {f'runs/m06/{n}' for n in ('16-repository-diff-check.log',
                                                  '17-repository-import.log', '18-staged-diff-check.log')}
    actual = set(audit.git('diff', 'HEAD', '--name-only').decode().splitlines())
    actual |= set(audit.git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    assert actual == expected, actual.symmetric_difference(expected)
    commands = json.loads((SOURCE/'runs/m06/commands.json').read_text())
    assert commands[-1]['exit_code'] != 0 and commands[-1]['output'] == 'runs/m06/18-staged-diff-check.log'
    raw = {c['output'].removeprefix('runs/m06/'): (STAGE/c['output']).read_text()
           for c in commands[:14]}
    # First 14 entries came from the verification runner and are exact raw stdout.
    assert all((SOURCE/'runs/m06'/name).read_text() == value for name, value in raw.items())
    (SOURCE/'runs/m06/checker-raw-output.json').write_text(json.dumps(raw, indent=2)+'\n')
    for name, value in raw.items():
        (SOURCE/'runs/m06'/name).write_text(value.rstrip('\n')+'\n')
    for file in ('verification-runner.py', 'closeout-repair.py'):
        shutil.copyfile(STAGE/'runs/m06'/file, SOURCE/'runs/m06'/file)
    report = {'reason': 'Initial staged whitespace check found an extra EOF blank line in two raw checker logs',
              'initial_failed_check': 'runs/m06/18-staged-diff-check.log',
              'exact_stdout_preserved_in': 'runs/m06/checker-raw-output.json',
              'readable_log_transform': "stdout.rstrip('\\n')+'\\n'",
              'model_checker_test_sources_acceptance_results_unchanged': True,
              'commit_created_before_repair': False,
              'repair_command': [str(SOURCE/'.venv/bin/python'), str(STAGE/'runs/m06/closeout-repair.py')]}
    (SOURCE/'runs/m06/log-preservation-repair.json').write_text(json.dumps(report, indent=2)+'\n')
    with (SOURCE/'docs/m06-relative.md').open('a') as document:
        document.write('''\n\n## Closeout log preservation\n\nThe initial staged whitespace check stopped before committing because two\nchecker stdout logs had an extra EOF blank line. Exact verification stdout is\nretained in [checker-raw-output.json](../runs/m06/checker-raw-output.json);\nreadable logs normalize only trailing newlines. The failed check and command\nremain in runs/m06. [Repair procedure](../runs/m06/closeout-repair.py) and\n[record](../runs/m06/log-preservation-repair.json) document the continuation.\nModel/checker/test sources and numerical results were unchanged. Final repaired\nrepository/staged whitespace, preservation, manifest and clean-tree checks passed.\n''')
    protected = audit.audit(SOURCE)
    audit.recorded(['git', 'diff', 'HEAD', '--check'], SOURCE, '19-repaired-repository-diff-check.log')
    files = audit.manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    audit.recorded(['git', 'diff', '--cached', '--check'], SOURCE, '20-repaired-staged-diff-check.log')
    files = audit.manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    assert audit.git('diff', '--cached', '--name-only').decode().splitlines() == files
    manifest = json.loads((SOURCE/'runs/m06/change-manifest.json').read_text())
    import hashlib
    assert all(hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() == h
               for f, h in manifest['sha256_excluding_self'].items())
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=SOURCE, check=True)
    subprocess.run(['git', 'commit', '-m', audit.MESSAGE], cwd=SOURCE, check=True)
    assert audit.git('status', '--porcelain=v1') == b''
    assert audit.git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines() == files
    print('Exact manifest verified. Clean working tree. Commit:', audit.git('rev-parse', 'HEAD').decode().strip())

if __name__ == '__main__':
    main()
