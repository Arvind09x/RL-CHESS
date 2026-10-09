"""D08 selection, held-out replay and bounded accounting contract checks."""
import asyncio
from collections import Counter
from copy import deepcopy

import chess
import pytest

from chess_rl.canonical_games import assign_split, canonical_payload, game_id
from chess_rl.initial_labels import (QUOTAS, accounting, key, replay_production_root,
                                     select_references, selection_order)
from chess_rl.position_sampling import phase, rule_position, sample_record
from chess_rl.stockfish_worker import AnnotationWorker
from tests.test_stockfish_worker import fixture_game, manifest, response


def rows():
    return [sample_record(f"g:{i:064x}", split, (i % 8, "middlegame", f"position-{i} w - -"))
            for split in QUOTAS for i in range(1, 60)]


def test_selection_matches_full_sort_and_order_independence():
    candidates = rows()
    excluded = {key(candidates[0])}
    quotas = dict(train=8, validation=3, test=4)
    # Distinct canonical games belong to a single split in this fixture.
    for i, r in enumerate(candidates):
        r["game_id"] = f"g:{i+1:064x}"
        r["rule_position"] = f"position-{i} w - -"
    expected = [r for s in QUOTAS for r in sorted(
        (r for r in candidates if r["split"] == s and key(r) not in excluded),
        key=selection_order)[:quotas[s]]]
    actual, eligible = select_references(iter(candidates), excluded, quotas=quotas)
    assert actual == expected
    assert select_references(reversed(candidates), excluded, quotas=quotas)[0] == expected
    assert Counter(r["split"] for r in actual) == quotas
    assert eligible == dict(train=58, validation=59, test=59)


@pytest.mark.parametrize("quotas", [dict(train=0, validation=1, test=1), dict(train=True, validation=1, test=1), dict(train=1)])
def test_invalid_quotas(quotas):
    with pytest.raises(ValueError):
        select_references([], set(), quotas=quotas)


def test_capacity_shortfall_fails():
    with pytest.raises(ValueError, match="capacity"):
        select_references([], set())


def test_cross_split_overlap_fails():
    candidates = [sample_record(f"g:{i:064x}", s, (0, "opening", "same w - -")) for i, s in enumerate(QUOTAS)]
    with pytest.raises(ValueError, match="overlap"):
        select_references(candidates, set(), quotas={s: 1 for s in QUOTAS})


def test_more_than_eight_roots_per_game_fails():
    candidates = [sample_record("g:train", "train", (i, "opening", f"p{i} w - -")) for i in range(9)]
    candidates += [sample_record("g:"+s, s, (0, "opening", s+" w - -")) for s in ("validation", "test")]
    with pytest.raises(ValueError, match="eight"):
        select_references(candidates, set(), quotas=dict(train=9, validation=1, test=1))


def heldout_game(manifest, split):
    game, ref, state = fixture_game(manifest, ply=4)
    for n in range(1, 10000):
        fen = chess.STARTING_FEN.rsplit(" ", 1)[0] + f" {n}"
        identity = game_id(canonical_payload(fen, game["moves_uci"]))
        if assign_split(identity, manifest.split_seed) == split:
            game.update(start_fen=fen, game_id=identity, split=split)
            state = chess.Board(fen)
            for move in game["moves_uci"]:
                state.push_uci(move)
            ref = sample_record(identity, split, (4, phase(state, 4), rule_position(state)))
            return game, ref
    raise AssertionError("fixture could not find split")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_explicit_heldout_replay_keeps_assignment(manifest, split):
    game, ref = heldout_game(manifest, split)
    calls, attempts = [], []
    async def attempt(state, trace):
        calls.append(state)
        trace.update(completed=True, hash_reset_completed=True)
        return response(state)
    default = AnnotationWorker(manifest, {}, attempt, attempts.append)
    assert asyncio.run(default.annotate(ref, game)) is None
    assert not calls and len(attempts) == 2
    attempts.clear()
    worker = AnnotationWorker(manifest, {}, attempt, attempts.append, root_replayer=replay_production_root)
    label = asyncio.run(worker.annotate(ref, game))
    assert label["split"] == split and label["game_id"] == game["game_id"]
    assert label["history_complete"] and label["available_snapshots"] == 5
    assert len(calls[0].board.move_stack) == 4 and len(attempts) == 1
    bad = deepcopy(ref)
    bad["split"] = "train"
    with pytest.raises(ValueError):
        replay_production_root(bad, game, manifest)


def test_exhaustion_and_interruption_are_separate_from_success(manifest):
    game, ref, _ = fixture_game(manifest, ply=4)
    attempts = []
    async def fail(state, trace):
        raise ValueError("controlled missing WDL")
    worker = AnnotationWorker(manifest, {}, fail, attempts.append, root_replayer=replay_production_root)
    assert asyncio.run(worker.annotate(ref, game)) is None
    exclusion = dict(reference=ref, reason="attempts_exhausted", attempts=2)
    result = accounting([ref], [], attempts, [exclusion])
    assert result["failed_attempts"] == 2 and result["retries"] == 1
    assert result["excluded"] == 1 and not result["pending_references"]
    assert result["successful_by_split"] == dict(train=0, validation=0, test=0)
    assert result["shortfall_by_split"] == QUOTAS
    async def interrupt(state, trace):
        raise asyncio.CancelledError("controlled interruption")
    attempts.clear()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(AnnotationWorker(manifest, {}, interrupt, attempts.append,
                    root_replayer=replay_production_root).annotate(ref, game))
    result = accounting([ref], [], attempts, [])
    assert result["interrupted_attempts"] == 1 and result["pending_references"] == [ref]
    assert not result["failed_attempts"] and not result["excluded"]


def test_noncallable_replay_hook_rejected(manifest):
    with pytest.raises(ValueError, match="callable"):
        AnnotationWorker(manifest, {}, None, None, root_replayer=None)
