"""Validate fresh integration evidence without altering any milestone evidence."""
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/d08-t01-integration'
EXTERNAL = ROOT.parent / 'external-artifacts/d08-t01-integration-audits'


def load(path):
    return json.loads(path.read_text())


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    commands = [load(p) for p in sorted(OUT.glob('*.command.json'))]
    assert len(commands) == 22, len(commands)
    for c in commands:
        assert c['exit_code'] == 0 and c['source_unchanged_during_command']
        for v in c['outputs'].values():
            p = OUT / v['path']
            assert identity(p)['sha256'] == v['container_sha256']
            raw = base64.b64decode(load(p)['data'], validate=True)
            assert len(raw) == v['decoded_bytes'] and hashlib.sha256(raw).hexdigest() == v['decoded_sha256']
    suites = {}
    for kind, passes, skips in [('model', 1284, 0), ('adapter', 1023, 8)]:
        suite = ET.parse(OUT / f'regressions-{kind}.xml').getroot()[0]
        assert int(suite.get('failures')) == int(suite.get('errors')) == 0
        assert int(suite.get('tests')) == passes + skips and int(suite.get('skipped')) == skips
        probes = ET.parse(OUT / f'accepted-review-probes-{kind}.xml').getroot()[0]
        assert probes.get('tests') == '84' and probes.get('failures') == probes.get('errors') == probes.get('skipped') == '0'
        stdout = load(OUT / f'frozen-adapter-{kind}.stdout.json')
        assert stdout['sha256'] == '32c8da66c132f9ce87bbdb8b772b02acf3318e6d235538553c904556d6da3737'
        suites[kind] = dict(passed=passes, skipped=skips, accepted_reviewer_probes=84,
                            frozen_adapter_fixtures=13, frozen_output_matches_accepted=True)
    real = load(OUT / 't01-real-batch.json')
    previous = load(ROOT / 'runs/t01/real-batch.json')
    for field in ('config', 'config_identity', 'dataset_records', 'references', 'selected_policy_classes',
                  'selected_wdl', 'dataset_label_identity', 'metrics', 'independent_losses',
                  'loss_absolute_errors', 'parameter_checks'):
        assert real[field] == previous[field], field
    for field in ('independent_output_gradients_pass', 'illegal_policy_gradients_exact_zero',
                  'batch_arrays_and_reloaded_adapter_batch_unchanged', 'unique_shared_bank',
                  'first_step_adam_formula_pass'):
        assert real[field] is True
    assert real['changed_parameter_tensors'] == 214 and real['parameter_tensors'] == 215
    audit = load(EXTERNAL / 'd08-replay.json')
    old = load(ROOT.parent / 'external-artifacts/d08-bcb208d-production/independent-replay.json')
    assert audit['successful_labels_replayed'] == 10000 and audit['quota_reached']
    assert audit['successful_by_split'] == dict(train=9000, validation=500, test=500)
    for field in ('cross_split_rule_position_overlaps', 'coverage', 'sample_identities'):
        assert audit[field] == old[field]
    # Timings vary; compare all substantive per-label outputs with the frozen evidence.
    def substantive(row):
        return {k: v for k, v in row.items() if not k.endswith('_seconds')}
    assert list(map(substantive, audit['record_checks'])) == list(map(substantive, old['record_checks']))
    loader = load(EXTERNAL / 'd08-loader.json')
    assert loader['all_10000_individual_samples_and_three_batch_passes_verified']
    assert loader['independent_feature_mask_fingerprints_match']
    assert loader['explicit_split_selection_verified'] and not loader['torch_imported']
    summary = dict(regressions=suites, checks=len(commands), all_commands_exit_zero=True,
                   t01_exact_reproduction_of_frozen_numerical_results=True,
                   t01_real_batch=dict(metrics=real['metrics'], loss_absolute_errors=real['loss_absolute_errors'],
                       parameter_tensors=215, changed_parameter_tensors=214, optimizer_steps=1),
                   d08=dict(successful_labels=10000, split_counts=audit['successful_by_split'],
                       zero_cross_split_overlap=True, substantive_per_label_audit_matches_frozen=True,
                       all_individual_samples_and_three_complete_batch_passes=True,
                       fresh_replay_report=identity(EXTERNAL / 'd08-replay.json'),
                       fresh_loader_report=identity(EXTERNAL / 'd08-loader.json')),
                   scope='User-supplied independent AI review acceptance; this chat inspected and verified integration. No new independent reviewer, human review, subagents, T02 or later implementation.')
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2, sort_keys=True) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
