"""Sequential CPU search, visit targets and explicit training exploration."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import chess
import numpy as np

from chess_rl.adapter import Evaluation, TerminalResult, id_to_move, legal_mask
from chess_rl.contracts import MOVE_COUNT
from .evaluator import evaluate
from .puct import Evaluator, Node, SearchConfig, backup, expand, traverse


@dataclass(frozen=True)
class TrainingExploration:
    """Per-search settings, not a self-play schedule. None means evaluation.

    Temperature 0 chooses maximum visits; positive temperature samples
    N**(1/temperature). Optional root Dirichlet noise mixes (1-epsilon)*P
    + epsilon*Dir(alpha), over sorted legal IDs, for this search call only.
    """
    temperature: float = 1.0
    noise_fraction: float = 0.0
    dirichlet_alpha: float = 0.3

    def __post_init__(self):
        for name in ("temperature", "noise_fraction", "dirichlet_alpha"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0 <= self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be in [0,1]")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")


def _validate_exploration(exploration, rng):
    if exploration is not None and not isinstance(exploration, TrainingExploration):
        raise ValueError("exploration must be TrainingExploration or None")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    if (exploration is not None
            and (exploration.temperature > 0 or exploration.noise_fraction > 0)
            and rng is None):
        raise ValueError("Stochastic training exploration requires an explicit RNG")


def root_visits(root: Node) -> np.ndarray:
    """Independent int64 [4352] snapshot of completed outgoing visits."""
    if not root.expanded:
        raise ValueError("Root visits require an expanded root")
    visits = np.zeros(MOVE_COUNT, dtype=np.int64)
    if isinstance(root.result, TerminalResult):
        if root.edges:
            raise ValueError("Terminal root must have no edges")
        return visits
    mask = legal_mask(root.state)
    if not isinstance(root.result, Evaluation) or set(root.edges) != set(np.flatnonzero(mask)):
        raise ValueError("Root edges must match all legal move IDs")
    for move_id, edge in root.edges.items():
        n = edge.visit_count
        if (edge.move_id != move_id or type(n) is not int or n < 0
                or n > np.iinfo(np.int64).max):
            raise ValueError("Root visits must be nonnegative int64 counts on legal edges")
        visits[move_id] = n
    return visits


def _policy(visits):
    total = sum(int(n) for n in visits)
    if total <= 0:
        raise ValueError("Training policy requires positive completed root visits")
    return visits.astype(np.float64) / total


def training_policy(root: Node) -> np.ndarray | None:
    """Float64 [4352] N/sum(N); terminal roots have no training target.

    Zero nonterminal visits raise ValueError; priors never supply a target.
    """
    visits = root_visits(root)
    return None if isinstance(root.result, TerminalResult) else _policy(visits)


def _choose(visits, exploration, rng):
    ids = np.flatnonzero(visits)
    if not len(ids):
        raise ValueError("Move selection requires positive completed root visits")
    if exploration is None or exploration.temperature == 0:
        return int(ids[np.argmax(visits[ids])])  # Sorted IDs: exact ties smallest.
    # Subtract maximum log-count before division: no positive exponent,
    # even for a very small temperature. Unvisited moves keep zero support.
    logs = np.log(visits[ids].astype(np.float64))
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((logs - logs.max()) / exploration.temperature)
    probabilities = weights / weights.sum()
    return int(rng.choice(ids, p=probabilities))


def select_move(root: Node, *, exploration: TrainingExploration | None = None,
                rng: np.random.Generator | None = None) -> int | None:
    """Select a shared legal move ID from completed visits, or None if terminal.

    Evaluation (exploration=None) never uses RNG or noise. This operation
    only selects an action; root noise is applied by search during traversal.
    """
    _validate_exploration(exploration, rng)
    visits = root_visits(root)
    if isinstance(root.result, TerminalResult):
        return None
    move_id = _choose(visits, exploration, rng)
    id_to_move(root.state, move_id)  # Shared legality/round-trip boundary.
    return move_id


@dataclass(frozen=True)
class SearchResult:
    """Snapshots include all root visits; completed_simulations is this call.

    Zero-budget fresh roots and failures before any completion have no target
    or action. Terminal roots always have neither. Failure snapshots never
    select an action; their target, if present, uses surviving completed visits.
    """
    requested_simulations: int
    completed_simulations: int
    root_visits: np.ndarray
    training_policy: np.ndarray | None
    move_id: int | None
    move: chess.Move | None
    terminal: TerminalResult | None


class SearchFailure(RuntimeError):
    """Stopped search, with partial completed statistics and original cause.

    partial is a SearchResult; stage identifies preparation/noise/simulation/
    selection. The original exception is available as __cause__.
    """
    def __init__(self, stage: str, partial: SearchResult):
        self.stage = stage
        self.partial = partial
        super().__init__(f"Search failed during {stage} after "
                         f"{partial.completed_simulations}/{partial.requested_simulations} completed simulations")


def _snapshot(root, budget, completed, move_id=None):
    visits = root_visits(root) if root.expanded else np.zeros(MOVE_COUNT, dtype=np.int64)
    terminal = root.result if isinstance(root.result, TerminalResult) else None
    policy = _policy(visits) if terminal is None and np.any(visits) else None
    move = id_to_move(root.state, move_id) if move_id is not None else None
    return SearchResult(budget, completed, visits, policy, move_id, move, terminal)


def search(root: Node, simulation_budget: int, *, config: SearchConfig = SearchConfig(),
           evaluator: Evaluator = evaluate, exploration: TrainingExploration | None = None,
           rng: np.random.Generator | None = None) -> SearchResult:
    """Prepare root, complete exactly budget successful nonempty backups, select.

    Budget counts additional completions, including when reusing a root.
    Preparation, pending paths and failures count zero. Stop on the first
    ordinary exception and raise SearchFailure with surviving statistics;
    no automatic retry. S03 rolls back failed backups. Interrupts propagate.
    Noise modifies only root edge P temporarily, restored even on failure or
    interruption; evaluator priors and descendant P remain untouched.
    Requires sequential ownership of the accepted full-history tree APIs.
    """
    if not isinstance(root, Node):
        raise ValueError("root must be a search-owned Node")
    if type(simulation_budget) is not int or simulation_budget < 0:
        raise ValueError("simulation_budget must be a nonnegative integer")
    if not isinstance(config, SearchConfig):
        raise ValueError("config must be SearchConfig")
    _validate_exploration(exploration, rng)
    completed, stage, original_priors = 0, "preparation", []
    try:
        expand(root, evaluator)
        if isinstance(root.result, TerminalResult):
            return _snapshot(root, simulation_budget, completed)
        if simulation_budget and exploration is not None and exploration.noise_fraction > 0:
            stage = "noise"
            ids = sorted(root.edges)
            noise = rng.dirichlet(np.full(len(ids), exploration.dirichlet_alpha))
            if not np.isfinite(noise).all() or (noise < 0).any() or not np.isclose(noise.sum(), 1):
                raise ValueError("Root Dirichlet sample must be finite probabilities")
            original_priors = [(root.edges[i], root.edges[i].prior) for i in ids]
            epsilon = exploration.noise_fraction
            for (edge, prior), sample in zip(original_priors, noise, strict=True):
                edge.prior = (1 - epsilon) * prior + epsilon * float(sample)
        stage = "simulation"
        while completed < simulation_budget:
            resolved = traverse(root, config, evaluator)
            count = backup(resolved)
            if count != 1 or not resolved.completed or not resolved.path:
                raise RuntimeError("Nonterminal simulation did not complete a nonempty backup")
            completed += count
        stage = "selection"
        visits = root_visits(root)
        move_id = _choose(visits, exploration, rng) if np.any(visits) else None
        return _snapshot(root, simulation_budget, completed, move_id)
    except Exception as exc:
        raise SearchFailure(stage, _snapshot(root, simulation_budget, completed)) from exc
    finally:
        for edge, prior in original_priors:
            edge.prior = prior
