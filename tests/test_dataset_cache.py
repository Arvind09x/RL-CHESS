import json
from pathlib import Path

import numpy as np
import pytest

from chess_rl import dataset_cache as DC
from chess_rl.dataset_cache import CachedSupervisedDataset, build_split_cache
from chess_rl.supervised import SupervisedDataset, open_dataset, verify_bundle
from chess_rl.training.checkpoints import DataOrder
from .test_supervised import fixture_bundle


@pytest.fixture
def accepted(tmp_path):
    """Fixture bundle whose summary carries loader_acceptance, as acceptance leaves it."""
    paths = fixture_bundle(tmp_path)
    summary_path = Path(paths["bundle"]) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["loader_acceptance"] = {"train": 3, "validation": 2, "test": 2}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return paths


def build(paths, *splits):
    verified = verify_bundle(**paths)
    return [build_split_cache(verified, split, bundle=paths["bundle"], data_paths=paths) for split in splits]


def test_cache_requires_finalized_summary(tmp_path):
    paths = fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="loader_acceptance"):
        build(paths, "train")


def test_cached_split_equals_fresh_dataset_in_every_field(accepted):
    build(accepted, "train", "validation")
    for split in ("train", "validation"):
        fresh = open_dataset(**accepted, split=split).freeze()
        cached = CachedSupervisedDataset.open(split=split, **accepted)
        assert isinstance(cached, SupervisedDataset) and cached.split == split
        assert len(cached) == len(fresh) and cached.source_identity == fresh.source_identity
        assert cached._admission_fingerprint == fresh._admission_fingerprint
        assert cached.freeze() is cached
        for i in range(len(fresh)):
            a, b = fresh[i], cached[i]
            assert a.reference == b.reference == cached.reference(i) and b.state is None
            assert a.policy_class == b.policy_class and type(b.policy_class) is int
            assert np.array_equal(a.features, b.features) and b.features.dtype == np.float32
            assert np.array_equal(a.legal_mask, b.legal_mask) and b.legal_mask.dtype == np.bool_
            assert np.array_equal(a.wdl, b.wdl) and b.wdl.dtype == np.float32
        fb, cb = fresh.batch(range(len(fresh))), cached.batch(range(len(cached)))
        assert fb.references == cb.references and np.array_equal(fb.features, cb.features)
        assert np.array_equal(fb.legal_mask, cb.legal_mask) and np.array_equal(fb.wdl, cb.wdl)
        assert np.array_equal(fb.policy_classes, cb.policy_classes) and cb.policy_classes.dtype == np.int64
        assert [b.references for b in fresh.batches(2)] == [b.references for b in cached.batches(2)]
    fresh_train = open_dataset(**accepted, split="train").freeze()
    assert DataOrder.reference_hash(CachedSupervisedDataset.open(split="train", **accepted)) == DataOrder.reference_hash(fresh_train)


def test_cached_arrays_are_owned_copies(accepted):
    build(accepted, "train")
    cached = CachedSupervisedDataset.open(split="train", **accepted)
    batch = cached.batch([0, 1])
    batch.features[0, 0, 0] = 99.0
    batch.legal_mask[0, 0] = not batch.legal_mask[0, 0]
    sample = cached[0]
    sample.wdl[0] = 7.0
    again = cached.batch([0])
    assert again.features[0, 0, 0] != 99.0 and again.legal_mask[0, 0] == cached[0].legal_mask[0]
    assert cached[0].wdl[0] != 7.0
    with pytest.raises(ValueError, match="frozen training dataset"):
        cached.split = "test"


def test_open_refuses_tampered_truncated_or_stale_caches(accepted, monkeypatch):
    build(accepted, "validation")
    directory = DC.cache_dir(accepted["bundle"], "validation")
    CachedSupervisedDataset.open(split="validation", **accepted)
    # one flipped byte
    data = bytearray((directory / "wdl.npy").read_bytes())
    data[-1] ^= 1
    (directory / "wdl.npy").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="altered"):
        CachedSupervisedDataset.open(split="validation", **accepted)
    (directory / "wdl.npy").write_bytes(bytes(data[:-1]))
    with pytest.raises(ValueError, match="altered"):
        CachedSupervisedDataset.open(split="validation", **accepted)
    # version mismatch
    data[-1] ^= 1
    (directory / "wdl.npy").write_bytes(bytes(data))
    CachedSupervisedDataset.open(split="validation", **accepted)
    monkeypatch.setattr(DC, "CACHE_FORMAT_VERSION", "d07-sample-cache-v2")
    with pytest.raises(ValueError, match="cache_format_version"):
        CachedSupervisedDataset.open(split="validation", **accepted)
    monkeypatch.undo()
    # source changed after the build: a different summary is a different source identity
    summary_path = Path(accepted["bundle"]) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["finalized_at_utc"] = "2000-01-01T00:00:00+00:00"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="source identity"):
        CachedSupervisedDataset.open(split="validation", **accepted)
    with pytest.raises(FileNotFoundError):
        CachedSupervisedDataset.open(split="test", **accepted)


def test_build_refuses_to_overwrite_and_leaves_no_partial_directory(accepted):
    build(accepted, "train")
    with pytest.raises(FileExistsError):
        build(accepted, "train")
    leftovers = [p for p in (Path(accepted["bundle"]) / "cache").iterdir() if ".tmp-" in p.name]
    assert leftovers == []
