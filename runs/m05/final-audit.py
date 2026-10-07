"""M05 preservation, focused archive and authorized commit closeout procedure.

Without --archive-and-commit, audits staging only. With it, archives just the
three intentional existing-file edits and new M05 files, audits repository
preservation/whitespace/import origins, stages that exact manifest and commits.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
STAGE = Path('/private/tmp/rl-chess-m05-change')
BASELINE = '0fdbd1eaf5e4d1701a2627d55f87b2150a4f252a'
ALLOWED = {'README.md', 'docs/m04-wdl.md', 'src/chess_rl/model/__init__.py'}
NEW = {'docs/m05-network.md', 'src/chess_rl/model/network.py', 'scripts/check_m05.py',
       'tests/network_checks.py', 'tests/test_network.py'}
COMMIT_MESSAGE = 'Assemble M05 intermediate GAB network and record M04 AI acceptance'


def baseline_files():
    return subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=SOURCE, text=True).splitlines()


def audit(root):
    checks = {}
    for file in baseline_files():
        old = subprocess.check_output(['git', 'show', BASELINE+':'+file], cwd=SOURCE)
        new = (root/file).read_bytes()
        if file not in ALLOWED:
            assert new == old, file
            checks[file] = hashlib.sha256(new).hexdigest()
    old = subprocess.check_output(['git', 'show', BASELINE+':docs/m04-wdl.md'], cwd=SOURCE)
    assert (root/'docs/m04-wdl.md').read_bytes().startswith(old)
    prior = json.loads((root/'runs/m01/original-plan-preservation.json').read_text())
    original = Path(prior['directory'])
    hashes = {f: hashlib.sha256((original/f).read_bytes()).hexdigest() for f in prior['sha256']}
    assert hashes == prior['sha256']
    report = {'baseline_commit': BASELINE, 'protected_repository_sha256': checks,
              'intentionally_changed_existing_files': sorted(ALLOWED),
              'm04_document_retained_as_exact_prefix': True,
              'original_planning_directory': str(original), 'original_plan_sha256': hashes,
              'all_original_hashes_match_historical_m01_evidence': True,
              'repository_coding_plan_interface_plan1_unchanged': True,
              'all_prior_runs_adapter_accepted_implementations_contracts_tests_scripts_source_pins_notices_dependencies_unchanged': True}
    (root/'runs/m05/preservation-check.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print('Preservation:', len(checks), 'baseline files and three original plans', flush=True)
    return len(checks)


def manifest(root, protected):
    run_files = {str(p.relative_to(root)) for p in (root/'runs/m05').iterdir() if p.is_file()}
    files = sorted(ALLOWED | NEW | run_files | {'runs/m05/change-manifest.json'})
    data = {'baseline_commit': BASELINE, 'milestone': 'M05 only', 'changed_files': files,
            'review_status': 'M05 automated acceptance only; stop for independent review',
            'prior_milestone_acceptance': 'M04 independent AI review; no human teammate review claimed',
            'protected_existing_files': protected, 'protected_original_plans': 3,
            'closeout_commands': [['git', 'add', '--', *files], ['git', 'diff', '--cached', '--check'],
                                  ['git', 'commit', '-m', COMMIT_MESSAGE],
                                  ['git', 'status', '--porcelain=v1'], ['git', 'rev-parse', 'HEAD']]}
    (root/'runs/m05/change-manifest.json').write_text(json.dumps(data, indent=2)+'\n')
    return files


def recorded(command, root, name, overrides=None):
    out = root/'runs/m05'
    commands = json.loads((out/'commands.json').read_text())
    env = os.environ.copy()
    overrides = {'PYTHONDONTWRITEBYTECODE': '1', **(overrides or {})}
    env.update(overrides)
    start = time.time()
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/name).write_text(result.stdout)
    commands.append({'command': command, 'cwd': str(root), 'environment_overrides': overrides,
                     'started_at_utc': started, 'exit_code': result.returncode,
                     'duration_seconds': round(time.time()-start, 3), 'output': 'runs/m05/'+name})
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    assert result.returncode == 0, result.stdout
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-and-commit', action='store_true')
    args = parser.parse_args()
    if not args.archive_and_commit:
        protected = audit(STAGE)
        for file in sorted(ALLOWED | NEW):
            old = str(SOURCE/file) if file in ALLOWED else '/dev/null'
            result = subprocess.run(['git', 'diff', '--no-index', '--check', old, str(STAGE/file)],
                                    capture_output=True, text=True)
            # --no-index reports 1 for a nonempty diff, even with --check.
            assert result.returncode in (0, 1) and not result.stdout and not result.stderr, result.stdout+result.stderr
        (STAGE/'runs/m05/14-staged-whitespace-check.log').write_text('All focused source/test/script/document diffs passed git diff --no-index --check.\n')
        manifest(STAGE, protected)
        return
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == BASELINE
    assert subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=SOURCE, text=True) == ''
    files = json.loads((STAGE/'runs/m05/change-manifest.json').read_text())['changed_files']
    # Only focused files: no historical runs or baseline planning/code copies.
    assert set(files) == ALLOWED | NEW | {str(p.relative_to(STAGE)) for p in (STAGE/'runs/m05').iterdir() if p.is_file()}
    for file in files:
        target = SOURCE/file
        assert file in ALLOWED or not target.exists(), file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STAGE/file, target)
    assert all((SOURCE/f).read_bytes() == (STAGE/f).read_bytes() for f in files)
    protected = audit(SOURCE)
    recorded(['git', 'diff', '--check'], SOURCE, '15-repository-diff-check.log')
    code = ("from pathlib import Path; import chess_rl.model.network as n; "
            "assert Path(n.__file__).resolve() == Path('src/chess_rl/model/network.py').resolve(); "
            "print('Repository network import verified:', n.__file__)")
    recorded([str(SOURCE/'.venv/bin/python'), '-c', code], SOURCE, '16-repository-import.log',
             {'PYTHONPATH': str(SOURCE/'src')+':'+str(SOURCE)})
    files = manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    recorded(['git', 'diff', '--cached', '--check'], SOURCE, '17-staged-diff-check.log')
    # Include the staged-check log/updated command record in the same focused commit.
    files = manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    actual = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=SOURCE, text=True).splitlines()
    assert actual == files, (actual, files)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=SOURCE, check=True)
    subprocess.run(['git', 'commit', '-m', COMMIT_MESSAGE], cwd=SOURCE, check=True)
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=SOURCE, text=True)
    assert status == '', status
    print('Clean working tree. Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip())


if __name__ == '__main__':
    main()
