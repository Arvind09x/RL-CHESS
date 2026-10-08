"""Paired P/R and fixed-opponent protocols using accepted chess/search APIs."""
from dataclasses import asdict, dataclass
import hashlib
import json
import importlib.metadata
import platform
import random

import chess

from chess_rl.adapter import apply_move, id_to_move, make_state, move_to_id, terminal_result
from chess_rl.contracts import ENCODING_VERSION, ENDING_POLICY, MOVE_VOCAB_VERSION
from .players import (Decision, DecisionFailure, Player, SearchPlayer, integer,
                      snapshot, text_id)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def turn_seed(game_seed, ply, player_id):
    return int(digest(["per-turn-sha256-v1", game_seed, ply, player_id])[:16], 16)


def state_trace(state):
    return dict(initial_fen=state.initial_fen, moves_uci=[m.uci() for m in state.board.move_stack],
                history_complete=state.history_complete, final_fen=state.board.fen(en_passant="fen"),
                final_promoted=int(state.board.promoted))


@dataclass(frozen=True)
class Opening:
    opening_id: str
    initial_fen: str
    moves_uci: tuple[str, ...]
    history_complete: bool
    source: str

    def __post_init__(self):
        text_id(self.opening_id, "opening_id")
        text_id(self.source, "opening source")
        if type(self.history_complete) is not bool:
            raise ValueError("Opening history_complete must be an explicit boolean")
        state = make_state(self.initial_fen, self.moves_uci, history_complete=self.history_complete)
        object.__setattr__(self, "initial_fen", state.initial_fen)
        object.__setattr__(self, "moves_uci", tuple(m.uci() for m in state.board.move_stack))
        # A match opening may end at a rule outcome, but never continue beyond it.
        prefix = make_state(state.initial_fen, history_complete=state.history_complete)
        for move in self.moves_uci:
            if terminal_result(prefix) is not None:
                raise ValueError("Opening continues after actual-occurrence termination")
            apply_move(prefix, move_to_id(prefix, chess.Move.from_uci(move)))

    @classmethod
    def from_state(cls, opening_id, state, *, source):
        return cls(opening_id, state.initial_fen, tuple(m.uci() for m in state.board.move_stack),
                   state.history_complete, source)

    def state(self):
        return make_state(self.initial_fen, self.moves_uci, history_complete=self.history_complete)

    def to_dict(self):
        return dict(**asdict(self), replay_sha256=digest(dict(initial_fen=self.initial_fen,
                    moves_uci=self.moves_uci, history_complete=self.history_complete)))

    def rule_key(self):
        # Public python-chess FEN: legal EP, no counters. Detect transpositions.
        return " ".join(self.state().board.fen(en_passant="legal").split()[:4])


@dataclass(frozen=True)
class OpeningSets:
    set_id: str
    validation: tuple[Opening, ...]
    test: tuple[Opening, ...]

    def __post_init__(self):
        text_id(self.set_id, "opening set_id")
        for name in ("validation", "test"):
            values = tuple(getattr(self, name))
            if not values or any(not isinstance(o, Opening) for o in values):
                raise ValueError("Both opening splits must be nonempty Opening sequences")
            object.__setattr__(self, name, values)
        all_openings = self.validation + self.test
        if len({o.opening_id for o in all_openings}) != len(all_openings):
            raise ValueError("Opening IDs must be globally unique")
        if len({o.rule_key() for o in all_openings}) != len(all_openings):
            raise ValueError("Opening roots must be rule-position distinct across and within splits")

    def to_dict(self):
        content = dict(set_id=self.set_id, validation=[o.to_dict() for o in self.validation],
                       test=[o.to_dict() for o in self.test], separation="distinct-root-rule-position-v1")
        return dict(**content, sha256=digest(content))


@dataclass(frozen=True)
class MatchRecord:
    configuration: dict
    games: tuple[dict, ...]

    def to_dict(self):
        return snapshot(dict(schema="e01-match-v1", configuration=self.configuration,
                             games=self.games, summary=summarize(self.games,
                                 self.configuration["left_player"]["player_id"],
                                 seed=self.configuration["seed"])))


def _run_game(opening, white, black, *, metadata, pair_id, leg, game_seed, ply_cap):
    state = opening.state()
    attempts = []
    record = dict(schema="e01-game-v1", pair_id=pair_id, leg=leg, game_seed=game_seed,
                  opening=opening.to_dict(), players=metadata, ply_cap=ply_cap,
                  ending_policy=ENDING_POLICY, encoding_version=ENCODING_VERSION,
                  move_vocabulary=MOVE_VOCAB_VERSION, attempts=attempts)
    attempt = None

    def finish(status, reason, terminal=None, error=None):
        winner = None if terminal is None or terminal.winner is None else (
            "white" if terminal.winner else "black")
        outcome = None if terminal is None else (
            "1/2-1/2" if winner is None else "1-0" if winner == "white" else "0-1")
        return snapshot(dict(**record, **state_trace(state), status=status, ending_reason=reason,
                             winner=winner, outcome=outcome, error=error))

    try:
        while True:
            terminal = terminal_result(state)
            if terminal is not None:
                return finish("completed", terminal.ending_reason, terminal)
            ply = len(state.board.move_stack)
            if ply >= ply_cap:
                return finish("truncated", "ply_limit")
            color = "white" if state.board.turn else "black"
            player = white if state.board.turn else black
            player_id = metadata[color]["player_id"]
            seed = turn_seed(game_seed, ply, player_id)
            attempt = dict(ply=ply, color=color, player_id=player_id, seed=seed,
                           status="started", stage="choose", move_id=None, move_uci=None,
                           applied=False, details={})
            attempts.append(attempt)
            # Even a broken injectable player cannot mutate authoritative history.
            supplied = state.copy()
            before = state_trace(supplied)
            decision = player.choose(supplied, seed=seed)
            if state_trace(supplied) != before:
                raise ValueError("Player mutated its supplied state")
            if not isinstance(decision, Decision):
                raise ValueError("Player must return Decision")
            attempt["details"] = snapshot(decision.details)
            attempt["move_id"] = decision.move_id
            attempt["move_uci"] = id_to_move(state, decision.move_id).uci()
            attempt["stage"] = "apply"
            candidate = state.copy()
            apply_move(candidate, decision.move_id)
            state = candidate
            attempt.update(status="completed", stage="done", applied=True)
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        cause = exc.cause if isinstance(exc, DecisionFailure) else exc
        status = "interrupted" if isinstance(cause, (KeyboardInterrupt, SystemExit)) else "failed"
        error = dict(type=type(cause).__name__, message=str(cause))
        if attempt is not None:
            attempt.update(status=status, error=error)
            if isinstance(exc, DecisionFailure):
                attempt["details"] = exc.details
        return finish(status, "execution_interrupted" if status == "interrupted" else "execution_failed",
                      error=error)


def run_match(left: Player, right: Player, openings: OpeningSets, *, split: str,
              match_id: str, comparison: str, seed: int, ply_cap: int,
              protocol_only: bool = False) -> MatchRecord:
    """Run exactly two games/opening, assigning left White then left Black.

    Identical actual opening/history is replayed in each leg; the board is not
    mirrored. Both legs are attempted independently, even if one fails or
    catches a soft interruption. No retries/replacement openings/forfeits.
    protocol_only explicitly permits absent or fixture checkpoints. A real
    P/R comparison requires hash-verified checkpoint bytes of the right kind;
    the injected evaluator's association with those bytes is caller attested
    until T04 exists. Records have no training targets or strength claims.
    """
    text_id(match_id, "match_id")
    integer(seed, "seed")
    integer(ply_cap, "ply_cap")
    if type(protocol_only) is not bool:
        raise ValueError("protocol_only must be bool")
    if not isinstance(openings, OpeningSets) or split not in ("validation", "test"):
        raise ValueError("OpeningSets and explicit validation/test split required")
    if comparison not in ("P-versus-R", "fixed-opponent"):
        raise ValueError("Unknown comparison")
    metas = [snapshot(p.metadata()) for p in (left, right)]
    for p, meta in zip((left, right), metas, strict=True):
        for name in ("player_id", "kind", "version"):
            text_id(meta.get(name), name)
        if not isinstance(meta.get("settings"), dict) or not callable(p.choose):
            raise ValueError("Player settings and choose method required")
        if isinstance(p, SearchPlayer):
            if p.checkpoint is not None:
                p.checkpoint.verify()
            if not protocol_only and (p.checkpoint is None or p.checkpoint.kind == "fixture" or p.role == "fixture"):
                raise ValueError("Real evaluation requires explicit trained checkpoint identity; use protocol_only for fixtures")
    if metas[0]["player_id"] == metas[1]["player_id"]:
        raise ValueError("Player identities must differ")
    if comparison == "P-versus-R":
        if not isinstance(left, SearchPlayer) or not isinstance(right, SearchPlayer):
            raise ValueError("P/R must both use SearchPlayer")
        if (left.role, right.role) != ("P", "R") or left.settings != right.settings:
            raise ValueError("P/R require explicit P and R roles and identical search settings")
    elif not isinstance(left, SearchPlayer) or metas[1]["kind"] not in (
            "random-legal", "material-greedy", "stockfish", "controlled"):
        raise ValueError("Fixed matches require a search player and an explicit fixed opponent")
    selected = getattr(openings, split)
    configuration = snapshot(dict(match_id=match_id, comparison=comparison, split=split,
        protocol_only=protocol_only,
        runtime=dict(python=platform.python_version(), python_chess=importlib.metadata.version("python-chess"),
                     chess=chess.__version__, numpy=importlib.metadata.version("numpy")),
        seed=seed, seed_algorithm="per-turn-sha256-v1",
        ply_cap=ply_cap, left_player=metas[0], right_player=metas[1], opening_sets=openings.to_dict(),
        planned_pairs=len(selected), planned_games=2 * len(selected),
        pairing="same-actual-opening-left-white-then-black", exploration_noise=False,
        scoring="completed-games-only; complete-pairs-only for paired uncertainty",
        interruption_policy="record soft interrupt and attempt paired leg; hard kill is outside snapshot API"))
    games = []
    for opening in selected:
        pair_id = f"{match_id}:{split}:{opening.opening_id}"
        for leg in (0, 1):
            game_seed = int(digest(["e01-game-seed-v1", seed, pair_id, leg])[:16], 16)
            white, black = (left, right) if leg == 0 else (right, left)
            metadata = dict(white=metas[leg], black=metas[1 - leg])
            games.append(_run_game(opening, white, black, metadata=metadata,
                                   pair_id=pair_id, leg=leg, game_seed=game_seed, ply_cap=ply_cap))
    return MatchRecord(configuration, tuple(games))


def summarize(games, left_id, *, seed=0, bootstrap_samples=2000):
    """Descriptive pair bootstrap; no Elo or significance/strength assertion.

    Resample opening pairs, not correlated legs. Omit incomplete pairs from
    this estimate and explicitly disclose attrition; completed single legs
    still enter the separate completed-game score denominator.
    """
    integer(seed, "seed")
    integer(bootstrap_samples, "bootstrap_samples", 1)
    counts = {k: 0 for k in ("completed", "truncated", "failed", "interrupted")}
    wdl = dict(win=0, draw=0, loss=0)
    pairs = {}
    scores = {}
    for game in games:
        counts[game["status"]] += 1
        pairs.setdefault(game["pair_id"], []).append(game)
        if game["status"] != "completed":
            if game["outcome"] is not None or game["winner"] is not None:
                raise ValueError("Unfinished game cannot have an outcome")
            continue
        expected_outcome = {None: "1/2-1/2", "white": "1-0", "black": "0-1"}
        if game["winner"] not in expected_outcome or game["outcome"] != expected_outcome[game["winner"]]:
            raise ValueError("Completed game requires a consistent rule outcome")
        color = next(c for c in ("white", "black") if game["players"][c]["player_id"] == left_id)
        label = "draw" if game["winner"] is None else "win" if game["winner"] == color else "loss"
        wdl[label] += 1
        scores[(game["pair_id"], game["leg"])] = {"win": 1.0, "draw": 0.5, "loss": 0.0}[label]
    pair_scores = []
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {g["leg"] for g in pair} != {0, 1}:
            raise ValueError("Every pair must retain exactly legs 0 and 1")
        if all(g["status"] == "completed" for g in pair):
            pair_scores.append((scores[(pair_id, 0)] + scores[(pair_id, 1)]) / 2)
    n = counts["completed"]
    uncertainty = None
    if len(pair_scores) >= 2:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
                       for _ in range(bootstrap_samples))
        uncertainty = dict(method="percentile-paired-bootstrap-v1", samples=bootstrap_samples,
            seed=seed, confidence=0.95, interval=[draws[int(0.025 * (bootstrap_samples - 1))],
                                                draws[int(0.975 * (bootstrap_samples - 1))]],
            limitation="descriptive conditional on completed pairs; small samples/attrition can bias estimates")
    return dict(status_counts=counts, left_player_id=left_id, completed_game_wdl=wdl,
        completed_game_denominator=n, completed_game_score=None if not n else (wdl["win"] + 0.5 * wdl["draw"]) / n,
        pairs=len(pairs), complete_pairs=len(pair_scores), incomplete_pairs=len(pairs) - len(pair_scores),
        complete_pair_score=None if not pair_scores else sum(pair_scores) / len(pair_scores),
        paired_uncertainty=uncertainty)
