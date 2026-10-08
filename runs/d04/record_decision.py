"""Resolve four D04 metadata fields in a new manifest; preserve accepted original."""
from dataclasses import replace
import json
from pathlib import Path
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity
from chess_rl.position_sampling import PHASE_DESCRIPTION, SAMPLING_DESCRIPTION

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/private/tmp/rl-chess-d04-prepared')
manifest = DatasetManifest.load(ROOT/'data/manifests/elite-2025-01-v1.json')
audit = json.loads((OUT/'overlap-audit.json').read_text())
verified = json.loads((ROOT/'runs/d04/output-verification.json').read_text())
assert audit['complete'] and verified['cross_split_overlaps_after']==0
assert verified['initial_10000_capacity_sufficient']
assert audit['sampling_algorithm']==SAMPLING_DESCRIPTION and audit['phase_boundaries']==PHASE_DESCRIPTION
assert audit['outputs']['protected-heldout.jsonl']==file_identity(OUT/'protected-heldout.jsonl')
result = replace(manifest,preparation=replace(manifest.preparation,
    sampling_algorithm=SAMPLING_DESCRIPTION,phase_boundaries=PHASE_DESCRIPTION,
    overlap_audit_sha256=file_identity(OUT/'overlap-audit.json')['sha256'],
    protected_heldout_registry_sha256=file_identity(OUT/'protected-heldout.jsonl')['sha256']))
result.require_annotation_ready()  # Metadata completeness only, not AI review acceptance.
(ROOT/'data/manifests/elite-2025-01-d04-v1.json').write_text(result.to_json())
(ROOT/'runs/d04/overlap-audit.json').write_bytes((OUT/'overlap-audit.json').read_bytes())
print(result.to_json())
