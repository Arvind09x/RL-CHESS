"""Accept R01 only after fresh checks and preservation comparisons pass."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from run_checks import ROOT, OUT, PRIMARY, CHECKPOINT, identity, git, save

BEFORE = json.loads((OUT / 'preservation-before.json').read_text())
SOURCE = Path('/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/r01-self-play')
REVIEW = SOURCE.parent / 'independent-review-20261010'
ACCEPTED = BEFORE['source_head']

def suites(directory):
    result = {}
    for p in directory.glob('*.xml'):
        suite = ET.parse(p).getroot().find('testsuite')
        counts = {k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        counts['passed'] = counts['tests']-counts['skipped']-counts['errors']-counts['failures']
        result[p.name] = counts
    for p in directory.glob('*.xml.json'):
        suite = ET.fromstring(base64.b64decode(json.loads(p.read_text())['data'])).find('testsuite')
        counts = {k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        counts['passed'] = counts['tests']-counts['skipped']-counts['errors']-counts['failures']
        result[p.name.removesuffix('.json')] = counts
    return result

def main():
    commands = {}
    for p in sorted(OUT.glob('*.command.json')):
        value = json.loads(p.read_text())
        assert value['exit_code'] == 0, p
        for n, item in value['tested_sources'].items():
            assert identity(ROOT / n) == item, n
        outputs = {}
        for label in ('stdout', 'stderr'):
            output = json.loads(p.with_name(p.name.replace('.command.json', f'.{label}.json')).read_text())
            raw = base64.b64decode(output['data'])
            assert len(raw) == output['bytes'] and hashlib.sha256(raw).hexdigest() == output['sha256']
            outputs[label] = dict(bytes=len(raw),sha256=output['sha256'])
        commands[p.name.removesuffix('.command.json')] = dict(argv=value['argv'],
            exit_code=0, duration_seconds=value['duration_seconds'], outputs=outputs)
    assert len(commands) == 13, len(commands)
    results = suites(OUT)
    expected = {'regression-model.xml':(1453,0), 'regression-adapter.xml':(1023,12),
        'focused.xml':(308,0), 'independent-r01.xml':(11,0),
        'accepted-probes-model.xml':(84,0), 'accepted-probes-adapter.xml':(84,0)}
    for n, (passed, skipped) in expected.items():
        assert results[n]['passed'] == passed and results[n]['skipped'] == skipped, results[n]
        assert results[n]['errors'] == results[n]['failures'] == 0
    for mode in ('adapter','model'):
        new = json.loads((OUT / f'adapter-replay-{mode}.stdout.json').read_text())
        old = json.loads((ROOT / f'runs/r01/adapter-replay-{mode}.stdout.json').read_text())
        assert base64.b64decode(new['data']) == base64.b64decode(old['data'])
    real = json.loads((OUT / 'real/summary.json').read_text())
    plan = json.loads((OUT / 'real/frozen-plan.json').read_text())
    assert identity(CHECKPOINT) == {'bytes':60883331,'sha256':'d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d'}
    fresh = json.loads((OUT / 'real/games.json').read_text())
    historical = json.loads((ROOT / 'runs/r01/real/games.json').read_text())
    assert fresh == historical and len(fresh) == 20
    assert real['continuation']['reference_core_sha256'] == real['continuation']['resumed_core_sha256'] == '3e04ba91ae21b3fa895458e347eaa310ea754de971deed0b394d1378eaddbd58'
    assert real['continuation']['hard_kill_recovered_prior_atomic_boundary']
    assert real['continuation']['global_rng_untouched']
    assert real['checkpoint_unchanged']
    assert git('diff', '--name-only', ACCEPTED) == b''
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip() == ACCEPTED
    assert not git('status','--porcelain=v1','-uall',cwd=SOURCE)
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip() == BEFORE['head']
    assert git('branch','--show-current',cwd=PRIMARY).decode().strip() == 'main'
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode() == BEFORE['status']
    assert git('diff','--binary',cwd=PRIMARY) == base64.b64decode(BEFORE['patch_base64'])
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert (PRIMARY/'coding-plan.md').read_bytes() == base64.b64decode(BEFORE['original_plan_base64'])
    for n, item in BEFORE['tracked_files'].items():
        assert identity(PRIMARY/n) == item, n
    for n, item in BEFORE['external_files'].items():
        assert identity(n) == item, n
    for n, item in BEFORE['review_artifacts'].items():
        assert identity(REVIEW/n) == item, n
    assert identity(REVIEW/'artifact-index.json') == BEFORE['review_index_identity']
    # All accepted tracked bytes are verified before changing acceptance docs.
    tracked = git('ls-tree','-r','--name-only',ACCEPTED).decode().splitlines()
    accepted_inventory = {n:identity(ROOT/n) for n in tracked}
    save(OUT/'accepted-tree-inventory.json', accepted_inventory)
    save(OUT/'preservation-after-verification.json',dict(
        primary_head_branch_status_index_patch_and_all_tracked_bytes_exact=True,
        primary_original_file_and_binary_patch_exact=True, accepted_tracked_files=len(tracked),
        accepted_tree_exact=True, external_files_verified=len(BEFORE['external_files']),
        review_artifacts_verified=len(BEFORE['review_artifacts']), original_source_checkout_clean=True))
    verdict = dict(accepted_at_utc=datetime.now(timezone.utc).isoformat(),local_date='2026-10-10',
        base_commit=BEFORE['head'], accepted_implementation_commit=ACCEPTED,
        integration_merge=git('rev-parse','HEAD').decode().strip(), verdict='ACCEPTED',
        review_type='User-supplied independent AI review; no new reviewer commissioned',
        supplied_review_thread='01a12493-4a87-7680-aa7c-5e16b980bfa8', actionable_findings=[],
        scope='Pinned macOS arm64 sequential CPU self-play and durable move-boundary resume',
        commands=commands,suites=results,real_checkpoint=plan['checkpoint'], real_results=real,
        all_twenty_historical_snapshots_exact=True, adapter_replay_exact_in_both_environments=True,
        preserved_accepted_tracked_files=len(tracked),preserved_external_files=76,
        limitations=['No R02 replay finalization or R03/R04 training','No CUDA integration or GPU verification',
            'No production opening sampling or P selection','No scaling, strength or hardware/power-loss claim'],
        initial_harness_attempt=dict(path='harness-attempt-1',results=suites(OUT/'harness-attempt-1'),
            cause='Missing basetemp parent directory; fixed setup only; original failed receipts retained'))
    save(OUT/'summary.json',verdict)
    text = '''# R01 independent review acceptance and integration — 10 October 2026

R01 `2e31236c33a114490c59be837459803c7b4f7299` is accepted for sequential
deterministic CPU self-play and durable move-boundary resume. The user supplied
the independent AI review in chat “Review R01 and CUDA changes”; its R01 verdict
has no actionable findings. This integration reruns checks and records acceptance;
it does not commission a new reviewer. The supplied review also discusses CUDA,
but only R01 is integrated here.

Primary `/Users/arvindsrinivass/Desktop/RL-Chess` was on `main` at `ea0f8a1`,
with a clean index and only the user's D06/D07/E01 checkbox changes in
`coding-plan.md`. Its complete original bytes, binary patch and tracked inventory
were saved before work. Integration was built in an isolated clone on
`codex/r01-integration`. A merge commit retains the exact original implementation
commit. The merged tree equaled the reviewed tree before adding acceptance.
No production changes or conflict resolution were needed. Historical pending-review
statements and implementation evidence remain exact; this dated acceptance
supersedes their status. Only the R01 checkbox and new acceptance appends change
the accepted documentation.

Fresh verification used the existing pinned model `.venv` and Torch-free
`.venv-acceptance` interpreters, with the isolated source path, disabled bytecode
and pytest caches, and one numerical thread. Python 3.12.12, NumPy 2.5.3,
python-chess 1.999/chess 1.11.2, pytest 9.1.1 and model Torch 2.8.0 were checked.
No dependencies were installed. The documented M06 diagnostic plugin applies only
to the historical M05 constructor helper; production inference retains GAB and
all Q/K/V relative terms.

| Fresh check | Result |
|---|---|
| Full model regression | 1,453 passed, zero skips |
| Full Torch-free adapter regression | 1,023 passed, 12 expected optional-Torch skips |
| Self-play/search/game/evaluator/checkpoint focused cases | 308 passed |
| Supplied independent R01 probes rerun | 11 passed |
| Accepted earlier review probes | 84 passed in each environment |
| Frozen adapter replay | All 13 fixtures byte-identical in both environments |
| M06 numerical check | Passed |
| Real T03-checkpoint self-play and fresh-process crash/resume | Passed, exact continuation |

The real suite uses unchanged `T03-plumbing-step-26`, 60,883,331 bytes, SHA-256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`.
All 20 fresh game snapshots equal the committed implementation snapshots exactly.
They retain 36 generated plies and 258 committed simulations; 8 completed,
11 truncated and 1 interrupted snapshots include one deliberate duplicate
completed game. The 19 ordinary executions make 148 batch-of-one forwards.
Both-color final-ply mates and fifty-move draws confirm terminal-before-cap
precedence. Unfinished records retain null outcomes.

The continuation fixture preserves its 11-ply prefix, commits two additional
moves, then receives SIGKILL during the next search. A new process reloads the
prior whole boundary and continues with scrambled global RNGs. Histories,
settings, actions, visits, policies and owned RNG equal uninterrupted execution;
global RNGs remain unchanged. Both core SHA-256 values are
`3e04ba91ae21b3fa895458e347eaa310ea754de971deed0b394d1378eaddbd58`.
Independent probes also pass writer contention, file/rename/directory publication
failures, exact recovery, stale-session rejection, partial-creation rejection
and stored-record tampering checks.

The first regression attempt omitted pytest's temporary parent directory.
Its failed setup receipts are retained in `harness-attempt-1`; creating that
directory resolved the harness error. Identical commands then passed, without
production or acceptance-threshold changes.

[Acceptance summary](../runs/r01-integration/summary.json),
[original primary preservation](../runs/r01-integration/preservation-before.json),
[verification preservation](../runs/r01-integration/preservation-after-verification.json)
and [supplied review](../runs/r01-integration/supplied-review/review.json)
record exact provenance, commands, lossless output and JUnit results. All
accepted tracked bytes, 76 external artifacts and 213 original review artifacts
were verified. Fresh real evidence lives under `runs/r01-integration/real`;
original `runs/r01` is untouched.

Application refuses primary drift, retains a recovery stash, fast-forwards main
to the committed isolated result and reapplies the original binary patch.
It verifies the complete resulting tree, exact three user checkbox substitutions,
clean index, original stash inventory and retained external/review artifacts.
The primary application receipt is `runs/r01-integration/main-integration.json`.

For a fresh verification checkout, create `runs/r01-integration/tmp` before
using `run_checks.py`; receipt and real-suite paths must be new because writes
refuse overwrites. The command receipts contain exact executable paths and argv.

Stop after R01 integration. R02 remains unchecked. CUDA integration, RL updates,
P selection, production sampling and scaling are outside this acceptance.
'''
    (ROOT/'docs/r01-integration.md').write_text(text)
    plan_path = ROOT/'coding-plan.md'
    original = plan_path.read_bytes()
    old = b'| [ ] | R01 /'
    assert original.count(old) == 1
    plan_path.write_bytes(original.replace(old,b'| [x] | R01 /',1) + '''

## R01 independent review acceptance and integration — 10 October 2026

R01 implementation `2e31236` is accepted for documented CPU self-play and durable
move-boundary resume following the user-supplied independent AI review with no
actionable findings and fresh isolated integration verification. Model regression
passed 1,453 tests; Torch-free adapter regression passed 1,023 with 12 expected
skips; focused cases passed 308 and independent R01 probes passed 11. Earlier
accepted probes passed 84 in each environment; adapter replay and M06 checks pass.
All 20 actual T03-checkpoint snapshots equal historical evidence, and fresh-process
SIGKILL/resume exactly preserves histories, visits, policies, actions and owned RNG.
The original implementation commit and all historical evidence remain retained.
Primary's exact D06/D07/E01 user edits are restored after verified application.
See [R01 integration](docs/r01-integration.md) and
[acceptance evidence](runs/r01-integration/summary.json). R02 and CUDA integration
remain outside scope; no subsequent milestone is begun.
'''.encode())
    with (ROOT/'README.md').open('a') as f:
        f.write('''

## R01 accepted CPU self-play integration — 10 October 2026

R01 `2e31236` is independently AI review accepted and freshly verified in an
isolated integration checkout. All 20 real T03-checkpoint snapshots reproduce
exactly; fresh-process crash/resume matches uninterrupted histories, actions,
visits, policies and RNG. Historical implementation evidence remains intact.
See [integration and acceptance](docs/r01-integration.md) and
[fresh evidence](runs/r01-integration/summary.json). This accepts sequential CPU
execution and durable move-boundary recovery; R02 and CUDA integration are pending.
''')
    inventory = {str(p.relative_to(OUT)):identity(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and 'tmp' not in p.relative_to(OUT).parts and p.name != '.writer.lock'
        and p.name != 'artifact-index.json'}
    save(OUT/'artifact-index.json',inventory)
    print('ACCEPTED after 13 successful commands; original accepted tree, primary patch, external and review artifacts preserved')

if __name__ == '__main__':
    main()
