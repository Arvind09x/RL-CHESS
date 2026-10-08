"""Uniform legal priors and synthetic WDL for CPU plumbing checks only."""
from collections.abc import Sequence

import numpy as np

from chess_rl.adapter import ChessState, Evaluation, TerminalResult, legal_mask, terminal_result
from chess_rl.contracts import MOVE_COUNT


def evaluate(states: Sequence[ChessState]) -> list[Evaluation | TerminalResult]:
    """Evaluate in input order without modifying boards or retained history.

    Terminal results pass through directly from the adapter. Nonterminal WDL
    is synthetic [win, draw, loss] = [0.25, 0.5, 0.25], so value is zero from
    either side's perspective. These outputs convey no playing strength.
    No feature encoding is needed. Each nonterminal result owns its arrays.
    """
    results: list[Evaluation | TerminalResult] = []
    for state in states:
        terminal = terminal_result(state)
        if terminal is not None:
            results.append(terminal)
            continue
        mask = legal_mask(state)
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError("Nonterminal state must have at least one legal move")
        priors = np.zeros(MOVE_COUNT, dtype=np.float32)
        priors[mask] = np.float32(1.0 / count)
        results.append(Evaluation(priors, np.array([0.25, 0.5, 0.25], dtype=np.float32), 0.0))
    return results
