"""Attribute supplied acceptance separately from fresh integration evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from checks import ROOT,OUT,EXTERNAL,identity,save,git,external_preserved

assert (OUT/'preservation-after-verification.json').exists()
assert all(json.loads((OUT/'comparisons.json').read_text()).values())
review=json.loads((ROOT/'runs/r03-acceptance/review-summary.json').read_text())
merge=json.loads((OUT/'merge.json').read_text())
assert review['bounded_cpu_verdict']=='ACCEPT' and review['actionable_findings']==[]
assert review['reviewed']==merge['implementation'] and review['freeze']==merge['freeze']
assert git('rev-parse',merge['freeze']+'^').decode().strip()==merge['base']
assert git('rev-parse',merge['implementation']+'^').decode().strip()==merge['freeze']
assert git('diff-tree','--no-commit-id','--name-only','-r',merge['freeze']).decode().splitlines()==['runs/r03/freeze.py','runs/r03/frozen-plan.json']
freeze_time=datetime.fromisoformat(git('show','-s','--format=%cI',merge['freeze']).decode().strip())
plan=json.loads((ROOT/'runs/r03/frozen-plan.json').read_text())
first_mixed=min(datetime.fromisoformat(json.loads(p.read_text())['started_at_utc'])
    for p in (ROOT/'runs/r03').glob('*.command.json')
    if 'scripts/run_r03.py' in json.loads(p.read_text())['argv'])
assert datetime.fromisoformat(plan['frozen_at_utc'])<freeze_time<first_mixed
def counts(name):
    suite=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
    result={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    assert result['failures']==result['errors']==0
    result['passed']=result['tests']-result['skipped']; return result
suites={name:counts(name) for name in ('model','adapter','supplied-probes')}
assert suites['model']['passed']==1553 and suites['adapter']['passed']==1023 and suites['supplied-probes']['passed']==22
scope='Bounded deterministic CPU mixed-update and persistence correctness; no learning, chess strength, accelerator, concurrent-source or pilot/main-run claim'
acceptance=dict(verdict='ACCEPTED',scope=scope,base=merge['base'],freeze=merge['freeze'],reviewed=merge['implementation'],
    local_date='2026-10-10',timezone='Asia/Kolkata',supplied_by_user=True,
    independent_review_thread='01a124e6-875e-7090-9ff4-855e4566b4d2',
    independent_review_title='Review R03 mixed-update commits',new_independent_review_commissioned=False,
    actionable_findings=[],review_summary_identity=identity(ROOT/'runs/r03-acceptance/review-summary.json'),
    report_identity=identity(ROOT/'runs/r03-acceptance/review-report.txt'),
    independent_results=review['tests'],freeze_precedes_execution=True,
    freeze_committed_at=freeze_time.isoformat(),first_retained_mixed_execution_at=first_mixed.isoformat(),
    documented_failure_boundary='Rejected gradients leave parameters/cursors unchanged but may consume dropout RNG; partial optimizer failures require checkpoint restoration',
    integration_evidence_separate='runs/r03-integration',r04_started=False,cuda_integrated=False)
save(ROOT/'runs/r03-acceptance/acceptance.json',acceptance)
doc='''# Accepted R03 integration — 10 October 2026

The user supplied independent AI review acceptance of freeze `d223be2` and
implementation `9fefb55` against accepted R02 main `6691380`. The review found no
actionable defects and accepts bounded deterministic CPU update and persistence
correctness. The supplied [report](../runs/r03-acceptance/review-report.txt),
[review summary](../runs/r03-acceptance/review-summary.json) and
[acceptance attribution](../runs/r03-acceptance/acceptance.json) are separate
from fresh integration verification. No new independent review was commissioned.

The isolated checkout is `r03-integration` under RL Project 2, on
`codex/r03-integration`. Its merge retains both original commits and exactly
reproduces the reviewed tree. No production correction or dependency change was
needed. The freeze commit predates mixed execution: 13:30:29 IST versus the first
retained mixed command at 13:38:59 IST. Historical implementation and review
artifacts retain their original bytes and pending-review wording as dated records.

| Fresh check | Result |
|---|---|
| Full model regressions | 1,553 passed, including 56 R03 cases |
| Torch-free adapter regressions | 1,023 passed; 14 expected skips |
| Supplied independent reviewer probes rerun | 22 passed |
| Frozen adapter replay in both environments | All 13 records reproduce exactly |
| Legacy v1 real-source fresh-process continuation | Exact next update, model, Adam, gradients, D07 order and RNG |
| Mixed full/prefix/resume in distinct processes | Exact updates two and three and final state |
| Accepted source reopen | 12 audited train labels and four completed replay roots; no held-out overlap |
| Sampling audit | Both permutations, cursors, epochs, counts and dedicated RNG reproduced independently over 101 updates |

Mixed runs use the original frozen 50% supervised fraction, 2+2 batch, Adam at
`1e-5`, unit policy/WDL weights and one update per full batch. Supervised class
and soft teacher WDL targets remain separate from visit distributions and exact
final replay WDL. Each source contributes six consumed samples across three
updates. Mixed losses reproduce `4.1802921295`, `2.9184389114` and `4.1084284782`.
All 215 parameter gradients are finite; illegal logit gradients are exactly zero.

The fresh resume scrambles global RNG before reopening both sources. Complete
step/final fingerprints match the accepted historical run exactly, including
model, Adam, gradients, progress, both complete sampling states and global RNG.
The new mixed boundary checkpoint also matches the historical bytes exactly.
Replayed sampling crosses the replay epoch on update three. Synthetic regressions
and reviewer probes additionally exercise both epochs, held-out whole-game/root
exclusions, unfinished receipts, source drift and compatibility rejection.
The [source audit](../runs/r03-integration/source-audit.json) records protected
registry identities/counts and historical v1 checkpoints loaded through the
common inference API with caller RNG preserved. Sixteen frozen inputs were
rehashed. Every unfinished receipt stays unlabeled.

New mixed and legacy checkpoints are stored separately under
`external-artifacts/r03-integration-20261010`; original checkpoints are never
overwritten. Their paths and identities are recorded in the
[new artifact index](../runs/r03-integration/new-external-artifact-index.json).
Lossless command output and source hashes are in `runs/r03-integration`;
[the summary](../runs/r03-integration/summary.json) binds their results.

Before primary integration, the exact original `coding-plan.md` file, binary
patch, raw Git index, staged entries and empty staged patch were saved losslessly.
All primary tracked files, the reviewed checkout, supplied review and 93
historical external artifacts were rehashed. Primary application refuses drift,
saves and verifies a recovery stash, fast-forwards local main, and reapplies the
original patch. Only the original D06/D07/E01 checkbox edits remain unstaged;
their patch hunks and original-file projection are exact. Blob hash headers
necessarily change because accepted R03 content is added to the plan. The raw
pre-integration index stays exact until application and remains saved; the new
index must match the integrated HEAD while preserving the original empty staged
state. The application receipt records full-tree and artifact verification.

R03 is accepted. R04 remains unchecked and unstarted; CUDA is not integrated.
Rejected gradients may consume dropout RNG; arbitrary optimizer failures can
partly mutate parameters, so exact retry restores the last immutable checkpoint.
Acceptance makes no learning, strength or accelerator claim. Work stops here.
'''
(ROOT/'docs/r03-integration.md').write_text(doc)
for name,append in [('README.md','''

## Accepted R03 mixed-update integration — 10 October 2026

The independently reviewed freeze `d223be2` and implementation `9fefb55` are
accepted for bounded deterministic CPU mixed-update and persistence correctness.
Both original commits are retained. Fresh isolated verification passed 1,553
model tests, 1,023 Torch-free adapter tests (14 expected skips), 22 supplied
reviewer probes, both frozen adapter replays, legacy checkpoint continuation
and exact mixed continuation through updates two and three. Reopened accepted
D07/R02 sources, protected exclusions and both sampling states match the freeze.
See [acceptance and integration](docs/r03-integration.md),
[supplied acceptance](runs/r03-acceptance/acceptance.json) and
[fresh evidence](runs/r03-integration/summary.json). Historical artifacts and
the exact unstaged user plan edits are preserved. Scope stops before R04 and
CUDA integration; no learning or strength claim is made.
'''),('coding-plan.md','''

### R03 independent acceptance and integration — 10 October 2026

- **R03 accepted:** User-supplied independent AI review accepts freeze `d223be2`
  and implementation `9fefb55` for bounded deterministic CPU mixed-update and
  persistence correctness, with no actionable findings. Original commits are
  retained; [supplied attribution](runs/r03-acceptance/acceptance.json) is
  separate from [fresh integration evidence](runs/r03-integration/summary.json).
- Fresh isolated checks pass 1,553 model tests, 1,023 adapter tests with 14 expected
  skips, 22 supplied reviewer probes, both frozen replays, legacy compatibility
  and fresh-process continuation. Accepted D07/R02 identities, held-out
  exclusions and both sampling permutations/cursors/epochs/RNG match. Mixed
  full/prefix/resume reproduce exact historical steps and checkpoint bytes.
- [Integration and preservation](docs/r03-integration.md) records exact original
  plan/index backups, artifact preservation and guarded primary application.
  The original D06/D07/E01 edits remain unstaged. R04 remains unchecked and
  unstarted; CUDA integration and strength/learning claims are outside scope.
''')]:
    path=ROOT/name; data=path.read_bytes()
    if name=='coding-plan.md':
        assert data.count(b'| [ ] | R03 /')==1
        data=data.replace(b'| [ ] | R03 /',b'| [x] | R03 /',1)
    path.write_bytes(data+append.encode())
git('add','--','README.md','coding-plan.md','docs/r03-integration.md','runs/r03-acceptance')
git('diff','--cached','--check')
git('commit','-m','Accept independently reviewed R03 freeze and mixed-update implementation')
acceptance_commit=git('rev-parse','HEAD').decode().strip()
commands={p.name:json.loads(p.read_text()) for p in OUT.glob('*.command.json')}
assert len(commands)==10 and all(v['exit_code']==0 and v['sources_unchanged'] for v in commands.values())
audit=json.loads((OUT/'source-audit.json').read_text())
legacy=json.loads((EXTERNAL/'legacy-continuation/summary.json').read_text())
assert legacy['exact_match'] and audit['actual_heldout_overlaps_zero']
new_paths=[EXTERNAL/'mixed-boundary.pt',*sorted((EXTERNAL/'legacy-continuation').iterdir())]
save(OUT/'new-external-artifact-index.json',{str(p):identity(p) for p in new_paths if p.is_file()})
save(OUT/'summary.json',dict(verdict='ACCEPTED',scope=scope,base=merge['base'],freeze=merge['freeze'],
    implementation=merge['implementation'],merge=merge['merge'],acceptance_commit=acceptance_commit,
    verified_at_utc=datetime.now(timezone.utc).isoformat(),suites=suites,
    commands={n:{k:v[k] for k in ('argv','exit_code','sources_unchanged','duration_seconds')} for n,v in commands.items()},
    mixed_continuation=json.loads((OUT/'comparisons.json').read_text()),legacy_continuation=legacy,
    source_audit={k:audit[k] for k in ('frozen_inputs_verified','supervised_count','replay_id','replay_membership',
        'receipt_counts','unfinished_receipts_unlabeled','heldout_games','protected_rule_positions','excluded_heldout',
        'sampler_updates_independently_reproduced','actual_three_update_sampling_state_exact')},
    legacy_historical_checkpoints_verified=len(audit['legacy_historical_checkpoints']),
    preservation=external_preserved(),primary_verification=json.loads((OUT/'preservation-after-verification.json').read_text()),
    r03_marked_accepted=True,r04_started=False,cuda_integrated=False))
save(OUT/'artifact-index.json',{str(p.relative_to(OUT)):identity(p) for p in OUT.rglob('*') if p.is_file()})
git('add','--','runs/r03-integration'); git('diff','--cached','--check')
git('commit','-m','Record fresh R03 integration regressions source audits and exact CPU continuation')
assert not git('status','--porcelain=v1','-uall')
print(json.dumps(dict(acceptance_commit=acceptance_commit,verified_integration=git('rev-parse','HEAD').decode().strip(),suites=suites),indent=2))
