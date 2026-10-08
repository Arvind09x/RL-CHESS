"""Compact D04 completion summary, command inventory and actual coverage tables."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = Path('/private/tmp/rl-chess-d04-prepared')
audit = json.loads((RUN/'overlap-audit.json').read_text())
verify = json.loads((RUN/'output-verification.json').read_text())
reproduction = json.loads((RUN/'reproduction.json').read_text())
assert audit['complete'] and verify['all_game_pools_and_sample_references_verified']
assert reproduction['outputs_byte_identical']
commands = []
for path in sorted(RUN.glob('*.command.json')):
    record = json.loads(path.read_text())
    for stream in ('stdout','stderr'):
        if stream not in record:
            continue
        data = (RUN/record[stream]['path']).read_bytes()
        assert len(data)==record[stream]['bytes'] and hashlib.sha256(data).hexdigest()==record[stream]['sha256']
    commands.append(dict(record=path.name,**record))
(RUN/'commands.json').write_text(json.dumps(commands,indent=2,sort_keys=True)+'\n')
results = {}
for name in ('focused-final','final-model-regression','final-adapter-regression','reviewer-adapter','reviewer-model'):
    suite = ET.parse(RUN/(name+'.xml')).getroot().find('testsuite')
    counts = {key:int(suite.get(key)) for key in ('tests','failures','errors','skipped')}
    assert counts['failures']==counts['errors']==0
    counts['passed']=counts['tests']-counts['skipped']
    results[name]=counts
capacity = {s:v['retained'] for s,v in audit['splits'].items()}
scales = {}
for total in (10000,100000,500000,1000000):
    allocation = dict(train=total*90//100,validation=total*5//100,test=total*5//100)
    scales[str(total)] = dict(proposed_success_allocation=allocation,
        fits_current_reference_capacity=all(capacity[s]>=n for s,n in allocation.items()),
        later_actual_label_cost_and_learning_evidence_required=True)
summary = dict(status='implemented; stop for independent review',completed_at_utc=datetime.now(timezone.utc).isoformat(),
    base_commit='37f286618642150b0ffa904abee0cfaac3ad75e8',branch='codex/d04-position-sampling',
    checkout=str(ROOT),bulk_output=str(OUT),repeated_selection_output='/private/tmp/rl-chess-d04-reselected',
    new_manifest='data/manifests/elite-2025-01-d04-v1.json',
    test_results=results,splits=audit['splits'],replay_counts=audit['replay_counts'],
    assignments_sha256=audit['assignments_sha256'],
    available_reference_capacity=capacity,capacity_total=audit['capacity_records'],scale_capacity=scales,
    distinct_rule_capacity={s:v['distinct_rule_positions'] for s,v in audit['splits'].items()},
    cross_split_overlap_before=verify['pairwise_distinct_overlap_identities_before'],cross_split_overlap_after=0,
    protected_registry_sha256=audit['outputs']['protected-heldout.jsonl']['sha256'],
    overlap_audit_sha256=hashlib.sha256((RUN/'overlap-audit.json').read_bytes()).hexdigest(),
    output_identities=audit['outputs'],labels_produced=0,review_accepted=False,merged=False,pushed=False,
    limitations=['CPU macOS arm64/Python 3.12.12 only',
        'One full legal replay and terminal-policy preparation; complete selection repeated from replayed pools',
        'Independent audit checks every pool/selection/reference; adapter root replay repeated on every 997th game only',
        'Input reorder/worker reproducibility and protected expansion are fixture tested; no real-corpus expansion run',
        'Candidate SQLite bytes are a cache, not stable output or trusted expansion evidence',
        'Reference counts allow same-split occurrences across games; unique rule capacity is reported separately',
        'No annotation, benchmark, training, GPU, cost, throughput or learning adequacy measurement',
        'New manifest metadata is complete; D04 review acceptance and D05 work remain pending'])
(RUN/'completion.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
# Append actual measurements once; never replace historical documentation.
lines = ['\n## Actual corpus results\n',
 '| Split | Eligible distinct candidates | Initial samples | Removed | Refilled | Retained | Covered games | Unique rule identities |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for split in ('train','validation','test'):
    s=audit['splits'][split]
    lines.append('| '+split+' | '+' | '.join(f'{s[k]:,}' for k in ('candidate_positions','initial_samples','removed','refilled','retained','games_covered','distinct_rule_positions'))+' |')
lines += ['','| Split | Opening before / after | Middlegame before / after | Endgame before / after |','|---|---:|---:|---:|']
for split in ('train','validation','test'):
    s=audit['splits'][split]
    lines.append('| '+split+' | '+' | '.join(f"{s['phase_before'][p]:,} / {s['phase_after'][p]:,}" for p in ('opening','middlegame','endgame'))+' |')
lines += ['',f"Full replay: **{audit['replay_counts']['roots']:,} roots** across **{sum(audit['assigned_games'].values()):,} games**. "
          f"Excluded {audit['replay_counts']['terminal_roots']:,} first terminal roots, "
          f"{audit['replay_counts']['post_terminal_roots']:,} suffix roots and "
          f"{audit['replay_counts']['repeated_roots']:,} repeated nonterminal rule occurrences.",
          '',f"Initial selection pairwise distinct overlap identities: {json.dumps(verify['pairwise_distinct_overlap_identities_before'],sort_keys=True)}. "
          'Final overlaps are zero across every pair of splits. Exclusion files record initial-sample removals; blocked candidate counts also include candidates bypassed during refill.',
          '',f"Total available retained reference capacity: **{audit['capacity_records']:,}**. "
          f"Fresh independent adapter replay: {verify['replayed_sample_roots']:,} retained roots from {verify['replayed_games']:,} games. "
          'Each reference remains unlabeled.',
          '', '| Target total successful labels | Train / validation / test proposal | Fits current reference capacity |','|---:|---|---|']
for total,entry in scales.items():
    a=entry['proposed_success_allocation']
    lines.append(f"| {int(total):,} | {a['train']:,} / {a['validation']:,} / {a['test']:,} | {'yes' if entry['fits_current_reference_capacity'] else 'no'} |")
lines += ['', 'These are capacity checks, not successful-label or cost claims. Larger scales remain optional and require later measured annotation and learning evidence.',
          '',f"Protected sample registry SHA-256: `{summary['protected_registry_sha256']}`. "
          f"Overlap audit SHA-256: `{summary['overlap_audit_sha256']}`.",
          '', 'Final checks: '+', '.join(f"{n}: {c['passed']} passed"+(f", {c['skipped']} expected skips" if c['skipped'] else '') for n,c in results.items())+'. Both accepted 13-fixture adapter reports match byte-for-byte; pinned upstream and M06 checks pass. All final success commands exit 0.',
          '', '[Complete compact result](../runs/d04/completion.json), [output audit](../runs/d04/output-verification.json), [repeated selection](../runs/d04/reproduction.json), [actual commands](../runs/d04/commands.json) and [preservation](../runs/d04/preservation.json).']
with (ROOT/'docs/d04-position-sampling.md').open('a') as handle:
    handle.write('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
