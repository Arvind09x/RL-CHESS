"""Record supplied acceptance and passing fresh verification before primary writes."""
import json
import xml.etree.ElementTree as ET
from checks import ROOT,OUT,save,identity,preserved,git

preservation=preserved()
assert preservation==json.loads((OUT/'preservation-after-verification.json').read_text())
before=json.loads((OUT/'preservation-before.json').read_text())
review=json.loads((OUT/'supplied-review/summary.json').read_text())
assert review['commit']==before['source_head'] and review['accepted_base']==before['primary']['head']
assert review['verdict']=='accept for bounded CPU R02'
assert 'No actionable findings.' in (OUT/'supplied-review/review.txt').read_text()
commands={p.name.removesuffix('.command.json'):json.loads(p.read_text()) for p in OUT.glob('*.command.json')}
assert len(commands)==6 and all(c['exit_code']==0 for c in commands.values())
suites={}
for name in ('model','adapter','supplied-probes'):
    nodes=ET.parse(OUT/(name+'.xml')).getroot().findall('testsuite')
    totals={k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    totals['passed']=totals['tests']-totals['failures']-totals['errors']-totals['skipped']
    assert totals['failures']==totals['errors']==0
    suites[name]=totals
assert suites['model']['passed']==1497 and suites['model']['skipped']==0
assert suites['adapter']['passed']==1023 and suites['adapter']['skipped']==13
assert suites['supplied-probes']['passed']==17 and suites['supplied-probes']['skipped']==0
archive=json.loads((OUT/'archive-verification.json').read_text())
summary=dict(verdict='ACCEPTED',scope='bounded local CPU R02 completed-game replay',local_date='2026-10-10',timezone='Asia/Kolkata',
    accepted_implementation_commit=before['source_head'],accepted_r01_main_base=before['primary']['head'],
    integration_merge=json.loads((OUT/'merge.json').read_text())['merge'],review=dict(
        chat_title='Review R02 replay implementation',chat_id='01a124bd-bb57-7a92-b75f-4c11eaae3c37',
        verdict=review['verdict'],actionable_findings=0,exact_review_files=len(before['review']),
        source=str(ROOT.parent/'r02-replay/runs/r02-review-20261010'),review_sha256=identity(OUT/'supplied-review/review.txt')['sha256']),
    suites=suites,r02_cases_within_model_suite=44,commands=commands,preservation=preservation,
    archive_verification={k:v for k,v in archive.items() if k!='samples'},
    production_tree_exact_to_reviewed_candidate=True,implementation_retained_by_merge=True,
    index_policy='Original raw index preserved exactly until primary application and retained losslessly in preservation-before.json; updated main index must match new HEAD with the same empty staged patch.',
    limitations=review['limits'],r03_started=False,cuda_integrated=False)
save(OUT/'summary.json',summary)
plan=ROOT/'coding-plan.md'; data=plan.read_bytes()
assert data.count(b'| [ ] | R02 /')==1 and b'| [ ] | R03 /' in data
data=data.replace(b'| [ ] | R02 /',b'| [x] | R02 /',1)
data+='''

## R02 independent review acceptance and integration — 10 October 2026

Accepted `2e57474` from `codex/r02-replay` for bounded local CPU R02 against
accepted R01 main `48bce1e`. The supplied independent AI review in chat
"Review R02 replay implementation" has no actionable findings. This dated
acceptance supersedes the implementation entry's pending-review status.

An isolated `codex/r02-integration` clone retains the original implementation
commit through a merge whose tree exactly equals the reviewed candidate.
Fresh model regression: 1,497 passed, including all 44 R02 cases. Torch-free
adapter regression: 1,023 passed, 13 expected optional-Torch skips. The supplied
17 independent probes pass again. No production code or dependency changes.

Independent direct chess-history and canonical SHA256 derivation of accepted
R01's 20 snapshots yields the same four roots: White win, Black win, White draw
and Black draw. All original position masks/visits/policies and complete sample
payloads match. Three fresh CLI processes verify admission, restart and original
archive repetition; source/game/sample identities, targets, database rows and
restart database bytes remain exact. Counts: 7 admitted completions, 1 duplicate,
11 excluded truncations and 1 excluded interruption. Export SHA256 remains
`4bdd6f656e159cfe8fb46db65264e876fdee5abe6abb1fc6f85bec50cf6eedf4`.

Fresh evidence and exact copied review live only under `runs/r02-integration`;
original R01/R02 evidence remains unchanged. Primary's complete original plan,
binary patch and raw index are saved losslessly and remain exact throughout
verification. Application refuses drift, fast-forwards main to the verified
commit, restores the exact D06/D07/E01 user patch and verifies the clean staged
state against the new HEAD. The original raw index remains in the preservation
record; the updated index records the new integrated tree.

See [integration record](docs/r02-integration.md),
[acceptance summary](runs/r02-integration/summary.json) and the primary
application receipt `runs/r02-integration/main-integration.json`.
Stop after accepted R02. R03 remains unchecked; CUDA integration has not begun.
'''.encode('utf-8')
plan.write_bytes(data)
with (ROOT/'README.md').open('ab') as f:
    f.write('\n\n## Accepted R02 integration — 10 October 2026\n\nR02 `2e57474` is independently AI-review accepted for bounded CPU completed-game\nreplay and integrated with its original commit retained. Fresh regressions and\nindependent four-root archive/restart verification pass. See\n[the integration record](docs/r02-integration.md) and\n[separate evidence](runs/r02-integration/summary.json). R03 and CUDA integration\nremain pending.\n'.encode('utf-8'))
print(json.dumps(dict(verdict=summary['verdict'],suites=suites,export_sha256=archive['audited_export_sha256']),indent=2))
