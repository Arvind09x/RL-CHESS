"""Small, data-free checks of proportional allocation and versioned continuation."""
from argparse import Namespace
from dataclasses import asdict
import copy
from fractions import Fraction

import pytest

from chess_rl.evaluation.matches import digest
from chess_rl.training import round_sampling as S
from chess_rl.training import round_training as R
from chess_rl.training.mixed_state import MixedState
from scripts.train_round2 import resolve_sampling


def initial(weights, sizes=None, updates=100):
    sizes = sizes or [5] * len(weights)
    sources = R.SamplerSources(supervised_count=11, supervised_source_id=digest("sf"),
        supervised_membership_sha256=digest("members"), replay_id=digest("union"),
        replay_membership=[digest(i) for i in range(sum(sizes))])
    legacy = MixedState.initial(sources, R.SETTINGS, seed=123)
    config = S.configuration(weights, [digest(["round", i]) for i in range(len(sizes))], sizes, updates)
    return S.RoundSamplingState.initial(legacy, config, seed=123)


@pytest.mark.parametrize("weights", [[.2, .8], [0, 1], [1, 0], [1, 3], [.013, .987], [1, 2, 7], [0, 1, 3]])
def test_every_prefix_bound_local_cycles_and_stockfish_order(weights):
    state = initial(weights)
    shadow = state.legacy
    units = state.config["weight_units"]
    count, local = [0] * len(weights), [[] for _ in weights]
    active = sum(bool(u) for u in units)
    for update in range(100):
        si, ri, state = state.next_selection()
        expected, _, shadow = shadow.next_selection()
        assert si == expected and state.legacy == shadow
        for n, index in enumerate(ri, update * 24 + 1):
            owner, row = divmod(index, 5)
            count[owner] += 1
            local[owner].append(row)
            for i, u in enumerate(units):
                assert abs(count[i] - Fraction(n * u, sum(units))) < max(1, active - 1)
                if not u:
                    assert count[i] == 0
    for rows in local:
        cycles = [rows[i:i + 5] for i in range(0, len(rows) - 4, 5)]
        assert all(sorted(c) == list(range(5)) for c in cycles)
        if len(cycles) > 5:
            assert len(set(map(tuple, cycles))) > 1
    assert [r.samples_seen for r in state.rounds] == count
    report = state.report()["by_round"]
    assert [r["unique_positions_seen"] for r in report] == [min(5, n) for n in count]
    assert [r["reused_draws"] for r in report] == [max(0, n - 5) for n in count]


@pytest.mark.parametrize("weights", [[.2, .8], [0, 1]])
def test_serialized_resume_across_allocation_and_local_reshuffle(weights):
    state = initial(weights, sizes=[7, 13], updates=8)
    _, _, state = state.next_selection()
    restored = S.sampling_state_from_dict(asdict(state))
    for _ in range(7):
        a, b, state = state.next_selection()
        x, y, restored = restored.next_selection()
        assert (a, b, state) == (x, y, restored)
    with pytest.raises(ValueError, match="budget exhausted"):
        restored.next_selection()
    bad = asdict(state)
    bad["version"] = "future-sampler-v999"
    with pytest.raises(ValueError):
        S.sampling_state_from_dict(bad)


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [1], [1, 2, 3], [float("nan"), 1],
                                    [float("inf"), 1], ["0.2", .8]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        S.configuration(weights, [digest(1), digest(2)], [2, 3], 8)


def args(**kw):
    return Namespace(**dict(dict(recipe="mix", round_data=[("r1", "p"), ("r2", "s")],
                                  sampling_mode=None, round_weights=None, updates=8), **kw))


def test_defaults_overrides_budget_and_saved_legacy_resolution():
    assert resolve_sampling(args(), None) == ([.2, .8], 8)
    assert resolve_sampling(args(recipe="pure"), None) == ([0., 1.], 8)
    assert resolve_sampling(args(round_weights=[1, 3]), None) == ([1, 3], 8)
    with pytest.raises(ValueError, match="explicit positive"):
        resolve_sampling(args(updates=None), None)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_sampling(args(sampling_mode="legacy", round_weights=[1, 3]), None)
    legacy = dict(provenance=dict(sampler={}, total_updates=8))
    assert resolve_sampling(args(recipe="pure", updates=None), legacy) == (None, 8)
    assert resolve_sampling(args(sampling_mode="legacy", updates=None), None) == (None, None)
    weighted = copy.deepcopy(legacy)
    weighted["provenance"]["sampler"]["round_sampling"] = initial([1, 3], updates=8).config
    assert resolve_sampling(args(updates=None), weighted) == ([1, 3], 8)
    for overrides in (dict(round_weights=[.2, .8]), dict(updates=9), dict(sampling_mode="legacy")):
        with pytest.raises(ValueError, match="saved run"):
            resolve_sampling(args(**overrides), weighted)


@pytest.mark.parametrize("extra", [[], ["--round-weights", "1"], ["--round-weights", "0", "0"],
                                  ["--round-weights", "nan", "1"], ["--round-weights", "-1", "2"],
                                  ["--round-weights", "inf", "1"],
                                  ["--sampling-mode", "legacy", "--round-weights", "1", "3"]])
def test_cli_rejects_invalid_sampling_before_opening_data(tmp_path, capsys, extra):
    from scripts.train_round2 import main
    base = ["train", "--recipe", "mix", "--data-config", "missing", "--round-data", "r1", "p",
            "--round-data", "r2", "s", "--start-checkpoint", "missing", "--start-sha256", "s",
            "--p-checkpoint", "missing", "--p-sha256", "p", "--output", str(tmp_path / "run")]
    assert main(base + extra) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_unknown_version_or_inconsistent_allocation_is_not_reinterpreted():
    state = initial([.2, .8], sizes=[7, 13])
    _, _, state = state.next_selection()
    value = asdict(state)
    # Swap valid total counts between the two rounds while keeping each cursor internally valid.
    for r, count in zip(value["rounds"], [19, 5]):
        size = len(r["order"])
        r.update(samples_seen=count, epoch=count // size, cursor=count % size)
    with pytest.raises(ValueError, match="allocation progress"):
        S.sampling_state_from_dict(value)
