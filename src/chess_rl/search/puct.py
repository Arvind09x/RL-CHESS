"""History-preserving PUCT selection/expansion; backup belongs to S03."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
from numbers import Real

import numpy as np

from chess_rl.adapter import (
    ChessState, Evaluation, TerminalResult, apply_move, legal_mask, terminal_result,
)
from chess_rl.contracts import MOVE_COUNT
from .evaluator import evaluate

LeafResult = Evaluation | TerminalResult
Evaluator = Callable[[Sequence[ChessState]], list[LeafResult]]


@dataclass(frozen=True)
class SearchConfig:
    """Selection choices only; no simulation runner or exploration settings."""
    c_puct: float = 1.5
    unvisited_value: float = 0.0

    def __post_init__(self):
        for name in ("c_puct", "unvisited_value"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not -1 <= self.unvisited_value <= 1:
            raise ValueError("unvisited_value must be in [-1,1]")


@dataclass(eq=False)
class Edge:
    """P is parent policy; N/W/Q are completed backups in parent-player view.

    S02 never updates N or W. S03 will update these mutable fields together.
    A child belongs exclusively to this edge, with no transposition cache.
    """
    move_id: int
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    def __post_init__(self):
        if type(self.move_id) is not int or not 0 <= self.move_id < MOVE_COUNT:
            raise ValueError("move_id must be an integer in 0..4351")
        if not isfinite(self.prior) or not 0 <= self.prior <= 1:
            raise ValueError("prior must be finite and in [0,1]")
        if type(self.visit_count) is not int or self.visit_count < 0:
            raise ValueError("visit_count must be a nonnegative integer")
        if not isfinite(self.value_sum) or abs(self.value_sum) > self.visit_count:
            raise ValueError("value_sum must be finite with abs(value_sum) <= visit_count")


@dataclass(eq=False)
class Node:
    """Owns a full-stack state copy and a side-to-move leaf evaluation.

    Treat state, edges and result as search-owned after construction. Only
    edge statistics are intended for S03 mutation. No independent node N/W
    is stored: parent visits are the sum of outgoing completed edge visits.
    """
    state: ChessState
    edges: dict[int, Edge] = field(default_factory=dict, init=False)
    result: LeafResult | None = field(default=None, init=False)

    def __post_init__(self):
        self.state = self.state.copy()

    @property
    def expanded(self) -> bool:
        return self.result is not None

    @property
    def parent_visits(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


def expand(node: Node, evaluator: Evaluator = evaluate) -> LeafResult:
    """Idempotent expansion. Publish edges/result only after successful checks."""
    if node.result is not None:
        return node.result
    outcome = terminal_result(node.state)
    if outcome is not None:
        node.result = outcome
        return outcome
    results = evaluator([node.state])
    if len(results) != 1 or not isinstance(results[0], Evaluation):
        raise ValueError("Nonterminal evaluate([state]) must return one Evaluation")
    result = results[0]
    mask = legal_mask(node.state)
    ids = np.flatnonzero(mask)
    if not len(ids):
        raise ValueError("Nonterminal node must have at least one legal move")
    if np.any(result.priors[~mask] != 0):
        raise ValueError("Evaluator priors must be zero on illegal moves")
    edges = {int(move_id): Edge(int(move_id), float(result.priors[move_id])) for move_id in ids}
    node.edges = edges  # Include every legal edge, even a zero-prior edge.
    node.result = result
    return result


def puct_score(edge: Edge, parent_visits: int, config: SearchConfig = SearchConfig()) -> float:
    """Q + c*P*sqrt(max(1, sum outgoing N))/(1+N); exact float ties."""
    if type(parent_visits) is not int or parent_visits < edge.visit_count:
        raise ValueError("parent_visits must be an integer >= edge.visit_count")
    q = edge.value_sum / edge.visit_count if edge.visit_count else config.unvisited_value
    return q + config.c_puct * edge.prior * sqrt(max(1, parent_visits)) / (1 + edge.visit_count)


def select_edge(node: Node, config: SearchConfig = SearchConfig()) -> Edge:
    """Largest score, then smallest shared move ID. No RNG or statistics writes."""
    if not isinstance(node.result, Evaluation) or not node.edges:
        raise ValueError("Selection requires an expanded nonterminal node")
    visits = node.parent_visits
    return max(node.edges.values(), key=lambda edge: (puct_score(edge, visits, config), -edge.move_id))


@dataclass(frozen=True)
class PathStep:
    parent: Node
    edge: Edge
    child: Node


@dataclass(frozen=True)
class TraversalResult:
    """Root-to-leaf references for S03; result.value is leaf side-to-move view.

    Returning this record means a leaf was resolved, not a simulation backed
    up. No visit has yet been completed. An expanded terminal root has path=().
    """
    path: tuple[PathStep, ...]
    leaf: Node
    result: LeafResult


def traverse(root: Node, config: SearchConfig = SearchConfig(), evaluator: Evaluator = evaluate) -> TraversalResult:
    """Follow legal edges to first unexpanded/terminal leaf, then resolve it.

    Call expand(root) explicitly first; that preparation consumes no traversal
    or visit. Repeated traversals may descend farther into the same branch
    because resolved leaves remain expanded. Neither success nor failure
    changes N/W. Each move is applied to a full-history copy via the adapter.
    """
    if not root.expanded:
        raise ValueError("Expand the root before traversal")
    node = root
    path = []
    while isinstance(node.result, Evaluation):
        edge = select_edge(node, config)
        if edge.child is None:
            state = node.state.copy()
            apply_move(state, edge.move_id)
            edge.child = Node(state)
        child = edge.child
        path.append(PathStep(node, edge, child))
        node = child
        if not node.expanded:
            break
    result = expand(node, evaluator)
    return TraversalResult(tuple(path), node, result)
