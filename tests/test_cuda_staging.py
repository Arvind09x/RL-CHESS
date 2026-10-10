"""Portable path configuration remains outside the immutable D07 data bytes."""
import json
from pathlib import Path

import pytest

from chess_rl.supervised import open_dataset
from scripts.stage_cuda_data import identity, stage
from .test_supervised import fixture_bundle


def test_staging_copies_exact_bytes_and_relocates(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    paths = fixture_bundle(source)
    config = source/"data-config.json"
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1",
        paths={k:str(v) for k,v in paths.items()})))
    destination = tmp_path/"staged"
    report = stage(config, destination)
    assert report["staged_d07_verified"] and report["train_count"] == 3
    for name, entry in report["files"].items():
        expected = {k:entry[k] for k in ("bytes", "sha256")}
        assert identity(destination/name) == identity(Path(entry["source"])) == expected
    moved = tmp_path/"moved"
    destination.rename(moved)
    portable = json.loads((moved/"data-config.json").read_text())
    assert all(not Path(v).is_absolute() for v in portable["paths"].values())
    assert len(open_dataset(**{k:moved/v for k,v in portable["paths"].items()}, split="train")) == 3
    with pytest.raises(FileExistsError): stage(config, moved)
