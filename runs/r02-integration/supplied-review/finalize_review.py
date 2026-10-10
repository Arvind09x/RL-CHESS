"""Create the independent evidence index after all review checks finish."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import xml.etree.ElementTree as ET

from checks import ROOT, OUT, ENV, git, save, sha

BASE = '48bce1ecb469d07d9d5a57aff7d9cb4896a7e281'
COMMIT = '2e574740d980084f1dea2da59eae13c87e052d10'

def tree(ref):
    result = {}
    for line in git(ROOT, 'ls-tree', '-r', ref).decode().splitlines():
        info, path = line.split('\t')
        result[path] = info.split()[2]
    return result

def tests(name):
    suite = ET.parse(OUT/(name+'.xml')).getroot().find('testsuite')
    result = {k: int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    result['passed'] = result['tests']-result['failures']-result['errors']-result['skipped']
    assert result['failures'] == result['errors'] == 0
    assert json.loads((OUT/(name+'.command.json')).read_text())['exit_code'] == 0
    return result

if __name__ == '__main__':
    assert git(ROOT, 'rev-parse', 'HEAD').decode().strip() == COMMIT
    subprocess.run(['git','--no-optional-locks','diff','--exit-code', COMMIT], cwd=ROOT, env=ENV, check=True)
    base, candidate = tree(BASE), tree(COMMIT)
    assert set(base) <= set(candidate)
    modified = [p for p in base if base[p] != candidate[p]]
    assert sorted(modified) == ['README.md','coding-plan.md']
    for path in modified:
        assert git(ROOT, 'show', COMMIT+':'+path).startswith(git(ROOT, 'show', BASE+':'+path))
    historical = json.loads((ROOT/'runs/r02/summary.json').read_text())
    for path, value in historical['evidence'].items():
        assert sha(ROOT/path) == value['sha256']
        assert (ROOT/path).stat().st_size == value['bytes']
    for path, value in historical['new_sources'].items():
        assert sha(ROOT/path) == value
    before = json.loads((OUT/'preservation-before.json').read_text())
    after = json.loads((OUT/'preservation-after.json').read_text())
    assert before == after
    checks = {name: tests(name) for name in ('model','adapter','probes-final')}
    assert checks['model']['passed'] == 1497
    assert checks['adapter']['passed'] == 1023 and checks['adapter']['skipped'] == 13
    assert checks['probes-final']['passed'] == 17
    reports = {name: json.loads((OUT/'probes-final-artifacts'/(name+'.json')).read_text())
               for name in ('admission','restart','accepted-main')}
    for name, report in reports.items():
        assert report['audited_export_sha256'] == historical['audited_export_sha256']
        assert report['samples_sha256'] == historical['samples_sha256']
        assert report['summary'] == historical['admission']
        if name != 'admission':
            assert all(r['repeated'] and r['added_samples'] == 0 for r in report['receipts'])
    save('summary.json', dict(
        review='fresh independent AI review; no delegation', local_date='2026-10-10', timezone='Asia/Kolkata',
        commit=COMMIT, accepted_base=BASE, verdict='accept for bounded CPU R02', actionable_findings=[],
        tests=checks, r02_cases_within_full_model_suite=44,
        runtime=dict(python=platform.python_version(), sqlite=sqlite3.sqlite_version,
                     packages={p: metadata.version(p) for p in ('numpy','python-chess','chess','pytest','torch')}),
        recorded_export=dict(summary=reports['admission']['summary'],
            audited_export_sha256=historical['audited_export_sha256'], samples_sha256=historical['samples_sha256'],
            admission_report_exact=True, restart_report_exact=True, accepted_main_report_exact=True,
            sample_ids_payloads_and_jsonl_exact=True),
        preservation=dict(candidate_tracked_files=len(after['review_tracked']),
            original_r01_tracked_files=len(after['r01_tracked']),
            primary_tracked_files=len(after['primary']['tracked_files']), external_files=len(after['external_files']),
            candidate_tracked_bytes_unchanged=True, accepted_existing_tree_bytes_equal_except_document_appends=True,
            primary_patch_staged_patch_index_and_all_tracked_bytes_unchanged=True,
            historical_r02_evidence_hashes_match=True, integration_performed=False, r03_started=False),
        modified_existing_base_paths=modified,
        independent_probe_scope=[
            'known complete White/Black wins and losses plus both-color repetition draws',
            'exact masks, raw visits and soft policies from separately built records and move-ID calculation',
            'complete prefixes beyond eight plies and independently calculated snapshot ownership/reflection/repetition',
            'excluded unfinished receipts followed by completed admission',
            'first-admitted duplicate order including zero-root representative and different coverage',
            'concurrent identical and different-source writers; one committed representative',
            'WAL export consistency during interleaved writer commit',
            'post-open corruption rejection on export and ingest with original bytes preserved',
            'missing/invalid databases and exclusive create preservation',
            'read-only historical audit plus fresh subprocess CLI admission/restart/accepted-main identities'],
        harness_correction=dict(initial_run='probes.xml: 16 passed, 1 failed',
            reason='Review test assumed ValueError for non-SQLite bytes. Actual sqlite3.DatabaseError correctly refused the file; no public buffer error type was promised.',
            corrected_run='probes-final.xml: 17 passed', production_source_changed=False,
            original_test_source='test_independent-initial.txt'),
        limits=['bounded local CPU replay correctness', 'no large-scale replay performance or capacity claim',
            'SIGKILL/transaction crash checks do not establish hardware power-loss behavior',
            'first admitted completed source determines the representative',
            'checkpoint identities are retained source claims; weight authentication is outside R02',
            'no learning update, GPU validation or playing-strength claim'],
        evidence={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,sha256=sha(p))
                  for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'summary.json'}))
    print(json.dumps(dict(verdict='accept for bounded CPU R02', tests=checks,
                         export=historical['audited_export_sha256'], preservation=True), indent=2))
