"""Bounded real T02 fixture inference/search audit; no training or campaign."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from chess_rl.adapter import Evaluation, encode_position, move_to_id
from chess_rl.search import real_evaluator as implementation
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.puct import Node
from chess_rl.search.runner import search
from tests.test_checkpoints import rng_snapshot, assert_rng_equal
from tests.test_evaluator import nonterminal_cases, terminal_cases, preserved
from tests.test_real_evaluator import BATCH_ATOL, BATCH_RTOL, ORACLE_ATOL, ORACLE_RTOL, softmax, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before_rng = rng_snapshot()
    evaluator = RealEvaluator(args.checkpoint)
    model = evaluator.checkpoint.model
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    nonterminal = nonterminal_cases()
    cases = terminal_cases()
    terminals = [case[0] for case in cases]
    states = [terminals[4], nonterminal[0], terminals[6], nonterminal[1], terminals[0]]
    states += nonterminal + terminals + [nonterminal[0]]
    before = [preserved(state) for state in states]
    calls, encoded = [], []
    original_encode = implementation.encode_position
    def record_encode(state):
        assert any(state is candidate for candidate in nonterminal)
        encoded.append(id(state))
        return original_encode(state)
    def record_forward(module, inputs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert all(not child.training for child in module.modules())
        calls.append(list(inputs[0].shape))
    hook = model.register_forward_pre_hook(record_forward)
    with patch.object(implementation, "encode_position", record_encode):
        assert evaluator([]) == []
        assert evaluator(terminals)
        assert not calls and not encoded
        batch = evaluator(states)
    assert calls == [[16, 64, 128]] and len(encoded) == 16
    hook.remove()
    batch_max = dict(priors=0., wdl=0., value=0.)
    oracle_max = dict(priors=0., wdl=0., value=0.)
    predictions = []
    for index, (state, result) in enumerate(zip(states, batch)):
        if not isinstance(result, Evaluation):
            reason, winner = next((reason, winner) for candidate, reason, winner in cases if candidate is state)
            assert result.ending_reason == reason and result.winner is winner
            np.testing.assert_array_equal(result.wdl, [0, 1, 0] if winner is None else [0, 0, 1])
            assert result.value == (0. if winner is None else -1.) and result.priors is None
            predictions.append(dict(index=index, terminal=reason, wdl=result.wdl.tolist(), value=result.value,
                                    legal_moves=len(list(state.board.legal_moves))))
            continue
        support(state, result)
        individual, = evaluator([state])
        for name in batch_max:
            actual, expected = getattr(result, name), getattr(individual, name)
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            batch_max[name] = max(batch_max[name], error)
            np.testing.assert_allclose(actual, expected, atol=BATCH_ATOL, rtol=BATCH_RTOL)
        features = torch.from_numpy(encode_position(state).features[None]).to(
            dtype=getattr(torch, evaluator.checkpoint.config.dtype))
        with torch.inference_mode(): policy, raw_wdl = model(features)
        ids = [move_to_id(state, move) for move in state.board.legal_moves]
        expected_prior = np.zeros(4352)
        expected_prior[ids] = softmax(policy[0].numpy()[ids])
        expected_wdl = softmax(raw_wdl[0].numpy())
        expected_value = expected_wdl[0] - expected_wdl[2]
        for name, expected in [("priors", expected_prior), ("wdl", expected_wdl), ("value", expected_value)]:
            # Isolate conversion from batch numerics by using the individual output.
            actual = getattr(individual, name)
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            oracle_max[name] = max(oracle_max[name], error)
            np.testing.assert_allclose(actual, expected, atol=ORACLE_ATOL, rtol=ORACLE_RTOL)
        predictions.append(dict(index=index, terminal=None, wdl=result.wdl.tolist(), value=result.value,
            legal_moves=len(ids), prior_sum=float(result.priors.sum(dtype=np.float64)),
            priors_sha256=hashlib.sha256(result.priors.tobytes()).hexdigest()))
    searches = []
    for index in [0, 1, 2, 4, 6, 7, 11, 12]:
        state = nonterminal[index]
        result = search(Node(state), 3, evaluator=evaluator)
        assert result.completed_simulations == 3 and result.root_visits.sum() == 3
        assert result.move in state.board.legal_moves
        assert result.training_policy.sum() == 1
        ids = {move_to_id(state, move) for move in state.board.legal_moves}
        assert set(np.flatnonzero(result.root_visits)) <= ids
        searches.append(dict(nonterminal_index=index, budget=3, completed=3, move=result.move.uci()))
    with patch.object(model, "forward", side_effect=AssertionError("Terminal inference")):
        for state, reason, _ in cases:
            result = search(Node(state), 3, evaluator=evaluator)
            assert result.completed_simulations == 0 and result.terminal.ending_reason == reason
    assert [preserved(state) for state in states] == before
    assert_rng_equal(before_rng, rng_snapshot())
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report = dict(checkpoint=dict(path=str(args.checkpoint), sha256=checkpoint_hash,
        bytes=args.checkpoint.stat().st_size, checkpoint_id=evaluator.checkpoint.checkpoint_id,
        config=asdict(evaluator.checkpoint.config), provenance=evaluator.checkpoint.provenance),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                     platform=platform.platform(), threads=1, interop_threads=1),
        tolerances=dict(batch=dict(atol=BATCH_ATOL, rtol=BATCH_RTOL),
                        independent_conversion=dict(atol=ORACLE_ATOL, rtol=ORACLE_RTOL)),
        batch_max_absolute_error=batch_max, independent_conversion_max_absolute_error=oracle_max,
        mixed_forward_shapes=calls, encoded_count=len(encoded), predictions=predictions,
        empty_and_all_terminal_inference_bypassed=True, exact_terminal_results=True,
        input_histories_and_rng_unchanged=True, bounded_search=searches,
        terminal_searches_bypassed=len(cases), scope="T04 correctness only; independent review pending")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ["batch_max_absolute_error",
        "independent_conversion_max_absolute_error", "mixed_forward_shapes", "scope"]}, indent=2))


if __name__ == "__main__": main()
