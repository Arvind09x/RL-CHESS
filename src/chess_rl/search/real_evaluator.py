"""T04 checkpoint-backed batched evaluation using shared adapter semantics.

Import this optional-Torch module explicitly; dummy search remains Torch-free.
The evaluator owns its T02-loaded model and never restores training RNG.
"""
from collections.abc import Sequence

import numpy as np
import torch

from chess_rl.adapter import (ChessState, Evaluation, TerminalResult,
                              encode_position, legal_mask, terminal_result)
from chess_rl.contracts import MOVE_COUNT, WDL_ORDER
from chess_rl.tensor_ops import finite_matrix, legal_policy_probs, validate_legal_support
from chess_rl.training.checkpoints import ModelConfig, load_inference


class RealEvaluator:
    """Callable search evaluator with ``evaluate(states)`` in original order.

    Checkpoints use T02's supported CPU float32/float64 combined M06 model.
    Public adapter results own float32 arrays, including for float64 inference.
    Nonterminal states share one forward call per invocation. No model work is
    performed for empty or entirely terminal input. A caller must serialize
    access to this owned model and avoid changing its parameters/configuration.
    """

    def __init__(self, checkpoint_path, *, expected_config: ModelConfig | None = None):
        self.checkpoint = load_inference(checkpoint_path, expected_config=expected_config)

    def __call__(self, states: Sequence[ChessState]) -> list[Evaluation | TerminalResult]:
        return self.evaluate(states)

    def evaluate(self, states: Sequence[ChessState]) -> list[Evaluation | TerminalResult]:
        # Complete adjudication before any encoding or model work. Rule queries
        # and feature history inspection are owned by the accepted adapter.
        results = []
        pending = []
        for index, state in enumerate(states):
            terminal = terminal_result(state)
            results.append(terminal)
            if terminal is None:
                pending.append((index, state))
        if not pending:
            return results

        masks = torch.from_numpy(np.stack([legal_mask(state) for _, state in pending]))
        validate_legal_support(masks)
        features = torch.from_numpy(np.stack([
            encode_position(state).features for _, state in pending
        ])).to(dtype=getattr(torch, self.checkpoint.config.dtype))

        with torch.inference_mode():
            model = self.checkpoint.model
            model.eval()
            policy_logits, wdl_logits = model(features)
            finite_matrix(policy_logits, "policy logits", MOVE_COUNT)
            finite_matrix(wdl_logits, "WDL logits", len(WDL_ORDER))
            if policy_logits.shape[0] != len(pending) or wdl_logits.shape[0] != len(pending):
                raise ValueError("model output batch must match nonterminal inputs")
            priors = legal_policy_probs(policy_logits, masks).to(torch.float32).numpy()
            wdl = torch.softmax(wdl_logits, dim=1).to(torch.float32).numpy()
            for row, (index, _) in enumerate(pending):
                results[index] = Evaluation(priors[row], wdl[row],
                                            float(wdl[row, 0] - wdl[row, 2]))
        return results
