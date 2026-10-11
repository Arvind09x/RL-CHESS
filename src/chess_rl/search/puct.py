"""History-preserving PUCT traversal and parent-perspective value backup."""
from collections.abc import Callable, Sequence
import copy
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

    Traversal never updates N or W. Backup updates these fields together.
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


@dataclass(eq=False, init=False)
class Node:
    """Owns a full-stack state copy and a side-to-move leaf evaluation.

    Treat state, edges and result as search-owned after construction. Only
    edge statistics are intended for S03 mutation. No independent node N/W
    is stored: parent visits are the sum of outgoing completed edge visits.
    """
    _state: ChessState = field(repr=False)
    edges: dict[int, Edge] = field(default_factory=dict, init=False)
    result: LeafResult | None = field(default=None, init=False)

    def __init__(self, state):
        if not isinstance(state, ChessState):
            raise ValueError("search admission requires ChessState")
        # Do not dispatch a caller-overridden copy method at the trust boundary.
        self._state = ChessState(state.board, state.initial_fen, state.history_complete)
        self.edges, self.result = {}, None

    @property
    def state(self):
        """Independent snapshot; public board mutation cannot change the tree."""
        return _copy_search_state(self)

    @property
    def expanded(self) -> bool:
        return self.result is not None

    @property
    def parent_visits(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


def _copy_search_state(node: Node) -> ChessState:
    """Trusted copying accepts only a search owner, never a public ChessState.

    The private state is admitted once and advanced only by legal traversal.
    Neither evaluators nor public snapshots receive its board storage.
    """
    if type(node) is not Node:
        raise ValueError("trusted copy requires a search-owned node")
    state = node._state
    result = object.__new__(ChessState)
    result.board = state.board.copy(stack=True)
    result.board._stack = [copy.copy(item) for item in result.board._stack]
    result.initial_fen, result.history_complete = state.initial_fen, state.history_complete
    return result


def _child_node(parent, move_id):
    state = _copy_search_state(parent)
    apply_move(state, move_id)
    node = object.__new__(Node)
    node._state, node.edges, node.result = state, {}, None
    return node


def _fresh_root(node):
    """Start a fresh tree from an owned child without replaying legal history."""
    result = object.__new__(Node)
    result._state, result.edges, result.result = _copy_search_state(node), {}, None
    return result


def expand(node: Node, evaluator: Evaluator = evaluate) -> LeafResult:
    """Idempotent expansion. Publish edges/result only after successful checks."""
    if node.result is not None:
        return node.result
    outcome = terminal_result(node._state)
    if outcome is not None:
        node.result = outcome
        return outcome
    results = evaluator([node.state])
    if len(results) != 1 or not isinstance(results[0], Evaluation):
        raise ValueError("Nonterminal evaluate([state]) must return one Evaluation")
    result = results[0]
    mask = legal_mask(node._state)
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
    """Root-to-leaf references; result.value is leaf side-to-move view.

    Returning this record means a leaf was resolved, not a simulation backed
    up. No visit has yet been completed. An expanded terminal root has path=().
    """
    path: tuple[PathStep, ...]
    leaf: Node
    result: LeafResult
    _completion: "_Completion | None" = field(default=None, init=False, repr=False, compare=False)

    @property
    def completed(self) -> bool:
        """True only after successful backup of this nonempty traversal."""
        return self._completion is not None and self._completion.completed


@dataclass(eq=False)
class _Completion:
    # Issued only by traverse. Shallow copies share the same completion ticket.
    path: tuple[PathStep, ...]
    leaf: Node
    result: LeafResult
    completed: bool = False


def backup(resolved: TraversalResult) -> int:
    """Complete one resolved traversal, returning 1, or 0 for a terminal root.

    Negate the leaf-player value before every reverse-path parent-edge update.
    Each edge gains one N and that value in W. Only records issued by traverse
    are accepted; duplicates/invalid records raise ValueError without writes.
    Validate and stage every update first, then publish N/W and completion
    together, rolling back on a publication exception. Failed records remain
    retryable after the cause is fixed. This API requires sequential access to
    search-owned records; no evaluator, board mutation or node counter is used.
    """
    if not isinstance(resolved, TraversalResult) or resolved._completion is None:
        raise ValueError("Backup requires a resolved record issued by traverse")
    ticket = resolved._completion
    if ticket.completed:
        raise ValueError("Traversal has already completed backup")
    if (resolved.path is not ticket.path or resolved.leaf is not ticket.leaf
            or resolved.result is not ticket.result):
        raise ValueError("Traversal record no longer matches its completion ticket")
    if (not isinstance(resolved.leaf, Node)
            or not isinstance(resolved.result, (Evaluation, TerminalResult))
            or resolved.leaf.result is not resolved.result):
        raise ValueError("Backup requires the leaf's resolved result")
    value = resolved.result.value
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not isfinite(value) or not -1 <= value <= 1):
        raise ValueError("Leaf value must be finite and in [-1,1]")
    value = float(value)
    if not resolved.path:
        if not isinstance(resolved.result, TerminalResult) or resolved.leaf.edges:
            raise ValueError("Empty paths require a terminal root")
        return 0

    # Verify the entire path, including ownership and continuity, before writes.
    node = resolved.path[0].parent
    nodes, edges = {id(node)}, set()
    for step in resolved.path:
        if (not isinstance(step, PathStep) or not isinstance(step.parent, Node)
                or not isinstance(step.edge, Edge) or not isinstance(step.child, Node)
                or step.parent is not node or not isinstance(step.parent.result, Evaluation)
                or step.parent.edges.get(step.edge.move_id) is not step.edge
                or step.edge.child is not step.child or not step.child.expanded
                or step.parent._state.board.turn == step.child._state.board.turn
                or id(step.child) in nodes or id(step.edge) in edges):
            raise ValueError("Backup path must be connected, resolved and owned by its edges")
        nodes.add(id(step.child))
        edges.add(id(step.edge))
        node = step.child
    if node is not resolved.leaf:
        raise ValueError("Backup path must end at the resolved leaf")

    updates = []
    for step in reversed(resolved.path):
        edge = step.edge
        n, w = edge.visit_count, edge.value_sum
        if (type(n) is not int or n < 0 or isinstance(w, (bool, np.bool_))
                or not isinstance(w, Real) or not isfinite(w) or abs(w) > n):
            raise ValueError("Edge statistics must have nonnegative integer N and finite abs(W)<=N")
        value = -value
        new_n, new_w = n + 1, float(w) + value
        if not isfinite(new_w) or abs(new_w) > new_n:
            raise ValueError("Backed-up statistics must remain finite with abs(W)<=N")
        updates.append((edge, n, w, new_n, new_w))

    try:
        for edge, _, _, new_n, new_w in updates:
            edge.visit_count = new_n
            edge.value_sum = new_w
        ticket.completed = True
    except BaseException:
        for edge, n, w, _, _ in updates:
            object.__setattr__(edge, "visit_count", n)
            object.__setattr__(edge, "value_sum", w)
        object.__setattr__(ticket, "completed", False)
        raise
    return 1


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
            edge.child = _child_node(node, edge.move_id)
        child = edge.child
        path.append(PathStep(node, edge, child))
        node = child
        if not node.expanded:
            break
    result = expand(node, evaluator)
    resolved = TraversalResult(tuple(path), node, result)
    object.__setattr__(resolved, "_completion", _Completion(resolved.path, node, result))
    return resolved
