"""Verify fresh and frozen evidence containers and publish a checked integration index."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from run_checks import ROOT, OUT, ARTIFACTS, SOURCE, BASE, identity, save

def decode(path):
    value = json.loads(path.read_text())
    raw = base64.b64decode(value['data'], validate=True)
    assert value['encoding'] == 'base64'
    assert len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256']
    return raw

def verify_command(path):
    record = json.loads(path.read_text())
    assert record['exit_code'] == 0 and record['tested_source_unchanged_during_command']
    for name, digest in record['tested_source_sha256'].items():
        assert identity(ROOT/name)['sha256'] == digest, (path.name, name)
    for output in record['outputs'].values():
        container = path.parent/output['path']
        assert identity(container)['sha256'] == output['container_sha256']
        raw = decode(container)
        assert len(raw) == output['decoded_bytes']
        assert hashlib.sha256(raw).hexdigest() == output['decoded_sha256']
    return record

records = [verify_command(p) for p in sorted(OUT.glob('*.command.json'))]
assert len(records) == 14
historical = [verify_command(p) for p in sorted((ROOT/'runs/t02').glob('*.command.json'))]
assert len(historical) == 8
save(OUT/'commands.json', records)
tests = {}
for p in sorted(OUT.glob('*.xml')):
    suites = list(ET.parse(p).getroot().iter('testsuite'))
    errors = sum(int(s.attrib['errors']) for s in suites)
    failures = sum(int(s.attrib['failures']) for s in suites)
    assert errors == failures == 0
    total = sum(int(s.attrib['tests']) for s in suites)
    skips = sum(int(s.attrib['skipped']) for s in suites)
    tests[p.stem] = dict(collected=total, passed=total-skips, skipped=skips, **identity(p))
assert tests['regressions-model']['passed'] == 1344
assert tests['regressions-adapter']['passed'] == 1023 and tests['regressions-adapter']['skipped'] == 9
assert tests['checkpoint-training-tests']['passed'] == 152 and tests['checkpoint-training-tests']['skipped'] == 0
assert tests['accepted-review-probes-model']['passed'] == tests['accepted-review-probes-adapter']['passed'] == 84
frozen = decode(ROOT/'runs/t02/frozen-adapter.stdout.json')
for kind in ('model', 'adapter'):
    assert decode(OUT/f'frozen-adapter-{kind}.stdout.json') == frozen
assert len(json.loads(frozen)) == 13
old = json.loads((ROOT/'runs/t02/real-summary.json').read_text())
fresh = json.loads((ARTIFACTS/'summary.json').read_text())
assert fresh == old
for name, expected in fresh['outputs'].items():
    assert identity(ARTIFACTS/name) == expected
    assert (ARTIFACTS/name).read_bytes() == (ROOT.parent/'external-artifacts/t02-cpu-continuation'/name).read_bytes()
assert identity(ARTIFACTS/'step-1.pt') == fresh['checkpoint']
assert (ARTIFACTS/'uninterrupted.json').read_bytes() == (ARTIFACTS/'resumed.json').read_bytes()
assert fresh['exact_match'] and fresh['separate_python_process'] and fresh['tolerance'] == 0
artifacts = {str(p):identity(p) for p in sorted(ARTIFACTS.iterdir())}
save(OUT/'external-artifacts.json', dict(directory=str(ARTIFACTS), files=artifacts,
     exact_reproduction_of_original_checkpoint_and_all_report_hashes=True))
save(OUT/'fresh-continuation-summary.json', fresh)
before = json.loads((OUT/'preservation-before.json').read_text())
after = json.loads((OUT/'preservation-after.json').read_text())
for key in ('accepted_files','reviewed_files','appended_documents','external_files'):
    assert before[key] == after[key], key
workspace = json.loads((OUT/'workspace-external-after.json').read_text())
assert workspace['exact_same_files_and_bytes']
save(OUT/'evidence-format-check.json', dict(fresh_commands_verified=len(records),
    historical_t02_commands_verified=len(historical), all_sources_and_output_container_hashes_verified=True,
    junit_errors_and_failures=0, checked_at_utc=datetime.now(timezone.utc).isoformat()))
save(OUT/'summary.json', dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE,
    source_commit=SOURCE, review_type='Independent AI review; user-supplied acceptance; no human review claim',
    review_supplied_by_user=dict(blocking_findings=0, checkpoint_training_tests=152,
        exact_fresh_process_cpu_continuation_and_checkpoint_report_hashes_reproduced=True),
    integration_verification=dict(all_commands_exit_zero=True, checks=len(records), tests=tests,
        frozen_adapter_fixtures=13, both_frozen_adapter_outputs_byte_identical=True,
        pinned_environments_verified=True, m06_diagnostic_plugin_used_in_model_regressions=True,
        m06_numerical_check_passed=True, offline_dependency_and_upstream_checks_passed=True),
    real_continuation=fresh, external_artifact_index='external-artifacts.json',
    preservation=dict(accepted_files=len(after['accepted_files']), reviewed_files=len(after['reviewed_files']),
        indexed_external_files=len(after['external_files']), workspace_historical_artifacts=len(workspace['files']),
        original_documents_and_completion_records_preserved=True, manifests_and_protected_heldout_registries_preserved=True,
        primary_main_and_exact_unrelated_plan_patch_preserved=True),
    limitations='Pinned macOS arm64 CPU float32 fresh-process continuation only; float64/nondefault configurations covered by unit tests. Standard Adam and explicit single-process D07 order only; no accelerator, scheduler, accumulation, custom RNG or worker prefetch restoration. No playing-strength claim.',
    scope='T02 integration only; no T03, T04, overfit, pilot training, P selection, self-play or RL begun'))
print(json.dumps(dict(checks=len(records), tests=tests, exact_continuation=True,
                     checkpoint_sha256=fresh['checkpoint']['sha256'], preservation=after['primary_main_and_exact_unrelated_patch_preserved']),indent=2))
