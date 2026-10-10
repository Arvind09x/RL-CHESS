"""Record supplied acceptance separately from fresh integration verification."""
from datetime import datetime,timezone
import json,xml.etree.ElementTree as ET
from checks import *
review=json.loads((ROOT/'runs/r04-acceptance/summary.json').read_text())
merge=json.loads((OUT/'merge.json').read_text())
assert review['verdict']=='ACCEPT bounded CPU Gate I' and review['actionable_findings']==[]
assert review['reviewed_head']==before()['source_head']
commands={p.name:json.loads(p.read_text()) for p in OUT.glob('*.command.json')}
assert len(commands)==12 and all(v['exit_code']==0 and v['sources_unchanged'] for v in commands.values())
suites={}
for name in ('model','adapter'):
    s=ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
    c={k:int(s.attrib[k]) for k in ('tests','failures','errors','skipped')}
    assert c['failures']==c['errors']==0
    c['passed']=c['tests']-c['skipped'];suites[name]=c
assert suites['model']['passed']==1553 and suites['adapter']['passed']==1023 and suites['adapter']['skipped']==14
full,prefix,resume=[json.loads((OUT/(n+'.json')).read_text()) for n in ('full','prefix','resume')]
assert len({v['pid'] for v in (full,prefix,resume)})==3
assert full['steps']==prefix['steps']+resume['steps']
assert all(v['exact_complete_saved_payload'] and v['inference_exact_rng_preserved'] for v in (full,prefix,resume))
reload=json.loads((OUT/'reload.json').read_text())
assert len(reload['checkpoints'])==8 and all(v['complete_training_state_exact'] and v['inference_rng_unchanged'] for v in reload['checkpoints'])
evaluation=json.loads((OUT/'evaluation.json').read_text())
assert evaluation['exact_full_match'] and evaluation['completed_denominator']==2 and evaluation['complete_pairs']==1
accepted_tree()
preservation=dict(primary=primary_unchanged(),historical=historical())
save(OUT/'preservation-after-verification.json',preservation)
scope='Bounded deterministic pinned CPU plumbing; no learning, strength, CUDA, pilot or pretrained-P selection claim'
acceptance=dict(verdict='ACCEPTED bounded CPU Gate I',scope=scope,supplied_by_user=True,
    independent_review_thread='01a12553-e821-7492-9a13-5c62faa6aa4c',
    independent_review_title='Review R04 local cycle evidence',new_independent_review_commissioned=False,
    base=merge['base'],sequence=merge['sequence'],reviewed_head=review['reviewed_head'],actionable_findings=[],
    review_summary=identity(ROOT/'runs/r04-acceptance/summary.json'),
    review_report=identity(ROOT/'runs/r04-acceptance/review-report.txt'),
    review_evidence_original_directory=str(REVIEW),fresh_integration_evidence='runs/r04-integration',
    local_date='2026-10-10',timezone='Asia/Kolkata',r04_accepted=True,gate_i_accepted=True)
save(ROOT/'runs/r04-acceptance/acceptance.json',acceptance)
doc="""# Accepted R04 / Gate I integration — 10 October 2026

The user-supplied independent review accepts the complete original sequence
`cec350b → bbd2e40 → 01f3c3f` against accepted R03 `2a41e27`, with no actionable
findings. R04 and Gate I are accepted for bounded deterministic CPU plumbing.
The [supplied report](../runs/r04-acceptance/review-report.txt),
[review summary](../runs/r04-acceptance/summary.json) and
[acceptance attribution](../runs/r04-acceptance/acceptance.json) are separate
from [fresh integration verification](../runs/r04-integration/summary.json).
No new independent review was commissioned.

Integration was prepared in isolated `r04-integration`, on
`codex/r04-integration`. The merge retains all three original commit identities
and exactly reproduces the reviewed tree. Original freezes, historical reports,
checkpoints, failed audit receipt and supplied review failure receipt are kept
unchanged. Historical pending-review wording remains a dated record; this
acceptance entry supplies the current status.

| Fresh integration verification | Result |
|---|---|
| Model regressions | 1,553 passed |
| Torch-free regressions | 1,023 passed, 14 expected optional-Torch skips |
| Plan and membership freeze chronology | Original committed bytes exact; both freezes precede governed execution |
| Native chess/root/target audit | 22 roots; 20 ordinary truncated roots excluded, two completed setup roots |
| Fresh real-search generation and R02 admission | All configurations, visits, moves, history, RNG, receipts and exported replay exact |
| Held-out/source membership | 12 audited supervised train labels, two eligible replay roots, no held-out overlap |
| New mixed boundary | T03 model exact, empty Adam, zero mixed progress, no legacy cursor |
| Full / prefix / resume in distinct processes | Exact steps and complete checkpoint payloads; scrambled RNG restored |
| Original and fresh checkpoint reloads | All eight inference and full training-state restores exact |
| E01 reproduction | Entire four-game record exact; two ordinary truncations, two fixture draws, one complete pair |

Fresh updates preserve the original visit distributions and exact draw WDL.
Independent float64 cross-entropy checks agree within `2e-6`; features, losses,
logits, logit and parameter gradients, model, Adam, both source sampling states,
sampling RNG, global RNG and progress match every recorded update.
The saved payloads also preserve model mode and provenance. E01 uses identical
search settings and reversed player colors with no noise. Its completed-game
denominator is two; only the one completed fixture pair supplies score 0.5.
These diagnostic fixture draws provide no evidence of learning or strength.
Both checkpoint roles remain fixture/protocol-only; pretrained P is unselected.

Starting T03 retains SHA256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`.
Original updated R04 retains SHA256
`967dde2cd8c2d3213c3297ae5dc5871f0e4db8faa11cb8145b8e55f919d3376b`.
Original serialized checkpoint identities are rehashed; fresh files have their
own byte identities and exactly matching complete payloads. Fresh checkpoints
are in `external-artifacts/r04-integration-20261010`, with a separate
[artifact index](../runs/r04-integration/new-external-artifact-index.json).
All command output is lossless; receipts bind argv, environment, timestamps,
exit status, source identities and the tested isolated commit. The verification
harness adapts supplied review probes to this isolated tree and new namespaces;
the original review harness remains exact.

Before application, the original primary plan file, binary patch, raw index,
staged entries, empty staged patch and stash inventory are saved losslessly.
Fresh verification preserves every original source/review file and 5,652
historical external/input files. Primary application refuses drift, verifies a
recovery stash, fast-forwards local main and reapplies the exact original patch.
Only the original D06/D07/E01 checkbox edits remain unstaged. Their patch hunks
and original-file projection remain exact; diff blob hashes change with the new
accepted plan content. The original raw index remains exact until application
and is retained in the backup. The current index must track integrated HEAD
while preserving the user's original empty staged state.
The [application receipt](../runs/r04-integration/main-integration.json) records
final main verification.

Work stops after R04 / Gate I acceptance. CUDA integration, pilot execution and
pretrained-P selection have not begun.
"""
(ROOT/'docs/r04-integration.md').write_text(doc)
readme="""

## Accepted R04 / Gate I — 10 October 2026

User-supplied independent review accepts original `cec350b`, `bbd2e40` and
`01f3c3f` with no actionable findings. Isolated integration retains all commits
and original evidence. Fresh verification passes 1,553 model tests, 1,023
Torch-free tests (14 expected skips), complete recorded-chain reproduction,
all checkpoint reloads, exact fresh-process continuation and the full E01 match.
See [acceptance and integration](docs/r04-integration.md),
[supplied acceptance](runs/r04-acceptance/acceptance.json) and
[fresh evidence](runs/r04-integration/summary.json). R04 and Gate I are accepted
for bounded CPU plumbing only. Historical artifacts and original user plan edits
are preserved. Stop before CUDA integration, pilot execution or pretrained-P
selection; fixture outcomes establish no learning or strength.
"""
with (ROOT/'README.md').open('a') as f:f.write(readme)
plan=(ROOT/'coding-plan.md').read_bytes()
assert plan.count(b'| [ ] | R04 /')==1
plan=plan.replace(b'| [ ] | R04 /',b'| [x] | R04 /',1)
plan+="""

### R04 independent acceptance and Gate I integration — 10 October 2026

- **R04 and Gate I accepted** for bounded deterministic CPU plumbing.
  User-supplied independent review accepts original `cec350b`, `bbd2e40` and
  `01f3c3f` against R03 `2a41e27`, with no actionable findings. Original
  commits, freezes and evidence remain intact.
- Fresh isolated verification passes 1,553 model tests and 1,023 Torch-free
  tests with 14 expected skips. The complete root/search/game/target/update/
  checkpoint/E01 chain reproduces exactly. All 20 truncated roots remain
  excluded. Actual completed setup roots feed three mixed updates; both source
  sampling states, RNG, gradients, Adam and complete checkpoint payloads match
  in distinct full/prefix/resume processes. All eight checkpoint reloads pass.
- [Supplied acceptance](runs/r04-acceptance/acceptance.json) remains distinct
  from [fresh integration evidence](runs/r04-integration/summary.json).
  [Integration/preservation](docs/r04-integration.md) records lossless original
  plan/index backups and guarded restoration of the original unstaged
  D06/D07/E01 edits.
- Ordinary E01 games remain truncated; the two completed setup draws and single
  complete pair establish protocol execution only. No learning or strength
  claim and no pretrained-P selection. Stop before CUDA integration, pilot
  execution or pretrained-P selection.
""".encode()
(ROOT/'coding-plan.md').write_bytes(plan)
expected_plan()
git('add','--','README.md','coding-plan.md','docs/r04-integration.md','runs/r04-acceptance')
git('diff','--cached','--check')
git('commit','-m','Accept independently reviewed R04 local CPU cycle and Gate I')
accepted=git('rev-parse','HEAD').decode().strip()
save(OUT/'new-external-artifact-index.json',{str(p):identity(p) for p in ART.iterdir() if p.is_file()})
save(OUT/'summary.json',dict(verdict='VERIFIED',scope=scope,base=merge['base'],sequence=merge['sequence'],
    reviewed_head=review['reviewed_head'],merge=merge['merge'],acceptance_commit=accepted,
    verified_at_utc=datetime.now(timezone.utc).isoformat(),suites=suites,
    commands={n:{k:v[k] for k in ('argv','exit_code','sources_unchanged','duration_seconds','head')} for n,v in commands.items()},
    audit=json.loads((OUT/'audit.json').read_text()),regeneration=json.loads((OUT/'regeneration.json').read_text()),
    fresh_process_continuation=dict(distinct_processes=[v['pid'] for v in (full,prefix,resume)],
        exact_complete_saved_payloads=True,exact_all_step_fingerprints=True,source_cursors_rng_adam_progress_exact=True),
    checkpoint_reloads=reload,evaluation=evaluation,preservation=preservation,
    r04_accepted=True,gate_i_accepted=True,cuda_integrated=False,pilot_executed=False,pretrained_P_selected=False))
save(OUT/'artifact-index.json',{str(p.relative_to(OUT)):identity(p) for p in OUT.rglob('*') if p.is_file()})
git('add','--','runs/r04-integration')
git('diff','--cached','--check')
git('commit','-m','Record fresh R04 integration chain checkpoint and continuation verification')
assert not git('status','--porcelain=v1','-uall')
print(json.dumps(dict(verified_integration=git('rev-parse','HEAD').decode().strip(),suites=suites),indent=2))

