"""M06 preservation/manifest audit, focused archive, one commit and clean check.

Default audits the fully tested staging tree. --archive-and-commit writes the
reviewable manifest into the repository and creates the user-requested commit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
STAGE = Path('/private/tmp/rl-chess-m06-change')
BASELINE = '2e6316dd0c49b0c22ff10ef72b55793856189a72'
ALLOWED = {'README.md', 'docs/m05-network.md', 'src/chess_rl/model/attention.py',
           'src/chess_rl/model/backbone.py', 'src/chess_rl/model/network.py'}
NEW = {'docs/m06-relative.md', 'src/chess_rl/model/relative.py', 'scripts/check_m06.py',
       'tests/relative_checks.py', 'tests/test_relative.py',
       'tests/fixtures/relative_reference.py', 'tests/fixtures/combined_reference.py'}
MESSAGE = 'Finalize M06 GAB and Q/K/V relative architecture with acceptance evidence'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=SOURCE)


def audit(root):
    protected = {}
    baseline_files = git('ls-tree', '-r', '--name-only', BASELINE).decode().splitlines()
    for file in baseline_files:
        previous = git('show', BASELINE+':'+file)
        current = (root/file).read_bytes()
        if file not in ALLOWED:
            assert current == previous, file
            protected[file] = hashlib.sha256(current).hexdigest()
    assert (root/'docs/m05-network.md').read_bytes().startswith(git('show', BASELINE+':docs/m05-network.md'))
    historical = json.loads((root/'runs/m01/original-plan-preservation.json').read_text())
    directory = Path(historical['directory'])
    original = {f: hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in historical['sha256']}
    assert original == historical['sha256']
    results = json.loads((root/'runs/m06/relative-acceptance.json').read_text())
    assert all(x['total'] == 5049411 for x in results['accounting'])
    report = {'baseline_commit': BASELINE, 'protected_repository_sha256': protected,
              'intentionally_changed_existing_files': sorted(ALLOWED),
              'm05_document_retained_as_exact_prefix': True,
              'original_planning_directory': str(directory), 'original_plan_sha256': original,
              'original_plan_hashes_match_m01': True,
              'planning_adapter_contracts_heads_gab_exports_dependencies_source_pins_notices_prior_runs_tests_checkers_unchanged': True}
    (root/'runs/m06/preservation-check.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print('Preserved', len(protected), 'baseline files, exact M05 prefix and three original plans', flush=True)
    return len(protected)


def manifest(root, protected):
    run_files = {str(p.relative_to(root)) for p in (root/'runs/m06').iterdir() if p.is_file()}
    files = sorted(ALLOWED | NEW | run_files | {'runs/m06/change-manifest.json'})
    hashes = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files
              if f != 'runs/m06/change-manifest.json'}
    report = {'baseline_commit': BASELINE, 'milestone': 'M06 only', 'changed_files': files,
              'sha256_excluding_self': hashes, 'protected_baseline_files': protected,
              'review_status': 'M06 automated acceptance; stop for independent review',
              'm05_acceptance': 'independent AI review, supplied by user; no human teammate review claimed',
              'commit_message': MESSAGE,
              'closeout_commands': [['git', 'add', '--', *files], ['git', 'diff', '--cached', '--check'],
                                    ['git', 'commit', '-m', MESSAGE],
                                    ['git', 'status', '--porcelain=v1'], ['git', 'rev-parse', 'HEAD']]}
    (root/'runs/m06/change-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return files


def recorded(argv, root, name, overrides=None):
    out = root/'runs/m06'
    commands = json.loads((out/'commands.json').read_text())
    env = os.environ.copy()
    overrides = {'PYTHONDONTWRITEBYTECODE': '1', **(overrides or {})}
    env.update(overrides)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    result = subprocess.run(argv, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/name).write_text(result.stdout)
    commands.append({'argv': argv, 'cwd': str(root), 'environment_overrides': overrides,
                     'started_at_utc': started, 'duration_seconds': round(time.monotonic()-start, 3),
                     'exit_code': result.returncode, 'output': 'runs/m06/'+name})
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    assert result.returncode == 0, result.stdout
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-and-commit', action='store_true')
    args = parser.parse_args()
    if not args.archive_and_commit:
        assert git('rev-parse', 'HEAD').decode().strip() == BASELINE
        assert git('status', '--porcelain=v1') == b''
        protected = audit(STAGE)
        for file in sorted(ALLOWED | NEW):
            old = str(SOURCE/file) if file in ALLOWED else '/dev/null'
            result = subprocess.run(['git', 'diff', '--no-index', '--check', old, str(STAGE/file)],
                                    capture_output=True, text=True)
            assert result.returncode in (0, 1) and not result.stdout and not result.stderr, result.stdout+result.stderr
        (STAGE/'runs/m06/15-staged-whitespace-check.log').write_text('All focused source/test/script/document diffs passed git diff --no-index --check.\n')
        context = {'source_root': str(SOURCE), 'staging_root': str(STAGE), 'baseline_head': BASELINE,
                   'initial_head_and_clean_tree_verified': True,
                   'initial_verification_command': 'pwd && git status --short && git rev-parse HEAD',
                   'archive_command': [str(SOURCE/'.venv/bin/python'), str(STAGE/'runs/m06/final-audit.py'), '--archive-and-commit'],
                   'scope': 'focused manifest only; existing interpreters; no network/dependency installation/subagents',
                   'exploratory_verification': {'argv': [str(SOURCE/'.venv/bin/python'), '-m', 'pytest',
                                                        'tests/test_relative.py', '-x', '-q', '-p', 'no:cacheprovider'],
                                               'cwd': str(STAGE), 'result': '36 passed in 19.18s before added seeded train/eval recovery assertions; final commands verify current bytes'}}
        (STAGE/'runs/m06/execution-context.json').write_text(json.dumps(context, indent=2)+'\n')
        files = manifest(STAGE, protected)
        baseline_files = set(git('ls-tree', '-r', '--name-only', BASELINE).decode().splitlines())
        actual_new = {str(p.relative_to(STAGE)) for p in STAGE.rglob('*') if p.is_file()} - baseline_files
        assert actual_new == set(files) - baseline_files, actual_new.symmetric_difference(set(files)-baseline_files)
        print('Staging manifest exact:', len(files), 'files')
        return
    assert git('rev-parse', 'HEAD').decode().strip() == BASELINE
    assert git('status', '--porcelain=v1') == b''
    files = json.loads((STAGE/'runs/m06/change-manifest.json').read_text())['changed_files']
    assert set(files) == ALLOWED | NEW | {str(p.relative_to(STAGE)) for p in (STAGE/'runs/m06').iterdir() if p.is_file()}
    for file in files:
        destination = SOURCE/file
        assert file in ALLOWED or not destination.exists(), file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STAGE/file, destination)
    assert all((SOURCE/f).read_bytes() == (STAGE/f).read_bytes() for f in files)
    protected = audit(SOURCE)
    recorded(['git', 'diff', '--check'], SOURCE, '16-repository-diff-check.log')
    code = ("from pathlib import Path; import chess_rl.model.relative as r; "
            "from chess_rl.model import ChessNetwork; "
            "assert Path(r.__file__).resolve() == Path('src/chess_rl/model/relative.py').resolve(); "
            "m=ChessNetwork(); assert all(b.self_attn.relative is not None for b in m.backbone.layers); "
            "assert sum(p.numel() for p in m.parameters())==5049411; "
            "print('Repository relative import/default/count verified:',r.__file__)")
    recorded([str(SOURCE/'.venv/bin/python'), '-c', code], SOURCE, '17-repository-import.log',
             {'PYTHONPATH': str(SOURCE/'src')+':'+str(SOURCE)})
    # Verify focused archive before recording the final manifest.
    changed = set(git('diff', '--name-only').decode().splitlines())
    untracked = set(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    expected = ALLOWED | NEW | {str(p.relative_to(SOURCE)) for p in (SOURCE/'runs/m06').iterdir() if p.is_file()}
    assert changed | untracked == expected, (changed | untracked).symmetric_difference(expected)
    files = manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    recorded(['git', 'diff', '--cached', '--check'], SOURCE, '18-staged-diff-check.log')
    files = manifest(SOURCE, protected)
    subprocess.run(['git', 'add', '--', *files], cwd=SOURCE, check=True)
    assert git('diff', '--cached', '--name-only').decode().splitlines() == files
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=SOURCE, check=True)
    subprocess.run(['git', 'commit', '-m', MESSAGE], cwd=SOURCE, check=True)
    assert git('status', '--porcelain=v1') == b''
    assert git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').decode().splitlines() == files
    print('Manifest verified. Clean working tree. Commit:', git('rev-parse', 'HEAD').decode().strip())

if __name__ == '__main__':
    main()
