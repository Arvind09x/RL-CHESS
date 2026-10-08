"""Audit saved match traces without checkpoints, evaluators or an engine."""
import math

from chess_rl.adapter import apply_move, id_to_move, legal_mask, make_state, terminal_result
from chess_rl.contracts import ENCODING_VERSION, ENDING_POLICY, MOVE_VOCAB_VERSION
from .matches import Opening, OpeningSets, digest, snapshot, state_trace, summarize, turn_seed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_match(record):
    """Verify pairing, provenance, routing, legal history, budgets and accounting.

    This reproduces states/outcomes, not neural predictions or engine timing.
    Failed player calls and search partials remain recorded assertions; no
    replay can prove the original external process/evaluator computation.
    """
    require(record["schema"] == "e01-match-v1", "Match schema")
    config = record["configuration"]
    require(config["exploration_noise"] is False, "Exploration noise")
    raw = config["opening_sets"]
    def opening(value):
        return Opening(**{k: value[k] for k in (
            "opening_id", "initial_fen", "moves_uci", "history_complete", "source")})
    sets = OpeningSets(raw["set_id"], tuple(map(opening, raw["validation"])), tuple(map(opening, raw["test"])))
    require(snapshot(sets.to_dict()) == raw, "Opening set provenance/hash")
    require(config["split"] in ("validation", "test"), "Opening split")
    selected = getattr(sets, config["split"])
    games = record["games"]
    require(len(games) == config["planned_games"] == 2 * len(selected), "Planned game count")
    require(config["planned_pairs"] == len(selected), "Planned pair count")
    left, right = config["left_player"], config["right_player"]
    require(left["player_id"] != right["player_id"], "Distinct players")
    if config["comparison"] == "P-versus-R":
        require((left["role"], right["role"]) == ("P", "R") and left["kind"] == right["kind"] == "search",
                "P/R routing")
        require(left["settings"] == right["settings"], "P/R identical search settings")
    else:
        require(config["comparison"] == "fixed-opponent", "Comparison")
    applied_total, search_completions, engine_nodes = 0, 0, 0
    for index, game in enumerate(games):
        o, leg = selected[index // 2], index % 2
        pair_id = f"{config['match_id']}:{config['split']}:{o.opening_id}"
        game_seed = int(digest(["e01-game-seed-v1", config["seed"], pair_id, leg])[:16], 16)
        require(game["pair_id"] == pair_id and game["leg"] == leg and game["game_seed"] == game_seed,
                "Pair/leg/seed identity")
        require(game["opening"] == snapshot(o.to_dict()), "Opening provenance")
        require(game["players"] == dict(white=(left, right)[leg], black=(right, left)[leg]), "Color reversal")
        require(game["ply_cap"] == config["ply_cap"], "Game cap")
        require((game["ending_policy"], game["encoding_version"], game["move_vocabulary"]) ==
                (ENDING_POLICY, ENCODING_VERSION, MOVE_VOCAB_VERSION), "Adapter contracts")
        state = o.state()
        for number, attempt in enumerate(game["attempts"]):
            require(terminal_result(state) is None, "Attempt after terminal")
            ply = len(state.board.move_stack)
            require(ply < game["ply_cap"] and attempt["ply"] == ply, "Attempt cap/ply")
            color = "white" if state.board.turn else "black"
            meta = game["players"][color]
            require(attempt["color"] == color and attempt["player_id"] == meta["player_id"], "Player routing")
            require(attempt["seed"] == turn_seed(game_seed, ply, meta["player_id"]), "Turn seed")
            require(attempt["status"] in ("completed", "failed", "interrupted"), "Attempt status")
            if attempt["move_uci"] is not None:
                require(id_to_move(state, attempt["move_id"]).uci() == attempt["move_uci"], "Legal selected ID")
            details = attempt["details"]
            if meta["kind"] == "search" and "requested_simulations" not in details:
                require(attempt["status"] in ("failed", "interrupted") and attempt["stage"] == "choose",
                        "Missing completed search trace")
            if meta["kind"] == "search" and "requested_simulations" in details:
                settings = meta["settings"]
                require(settings["exploration_noise"] is False and settings["exploration"] is None,
                        "Search noise disabled")
                budget = settings["simulation_budget"]
                require(details["requested_simulations"] == budget, "Requested search budget")
                edges = details["visited_edges"]
                require(len({e[0] for e in edges}) == len(edges), "Duplicate root edge")
                mask = legal_mask(state)
                for i, n, w in edges:
                    require(type(i) is int and 0 <= i < len(mask) and mask[i] and type(n) is int and n > 0
                            and type(w) in (int, float) and math.isfinite(w) and abs(w) <= n, "Root edge statistics")
                completed = details["completed_simulations"]
                require(type(completed) is int and completed == sum(e[1] for e in edges)
                        and 0 <= completed <= budget, "Search completions")
                search_completions += completed
                if details["search_status"] == "completed":
                    require(completed == budget and edges, "Completed search budget")
                    best = max(edges, key=lambda e: (e[1], -e[0]))[0]
                    require(details["selected_id"] == attempt["move_id"] == best, "Evaluation visit selection")
                else:
                    require(details["selected_id"] is None, "Failed search selected action")
            if meta["kind"] == "stockfish" and "returned_info" in details:
                engine_nodes += details["returned_info"].get("nodes", 0)
                require(details["requested_nodes"] == meta["settings"]["nodes"], "Engine node budget")
                require(details["hash_reset_completed"] is True, "Engine hash reset")
                require(details["engine_id"]["name"] == meta["engine"]["uci_name"], "Engine identity")
                effective = details["effective_options"]
                requested = dict(meta["settings"]["options"])
                requested["Skill Level"] = requested.pop("SkillLevel")
                require(all(effective.get(k) == v for k, v in requested.items()), "Engine settings")
                if attempt["applied"]:
                    require(details["returned_bestmove"] == attempt["move_uci"]
                            and type(details["returned_info"].get("nodes")) is int
                            and details["returned_info"]["nodes"] > 0, "Engine move/node accounting")
            if attempt["applied"]:
                require(attempt["status"] == "completed" and attempt["stage"] == "done", "Applied status")
                require(attempt["move_uci"] is not None, "Applied move missing")
                apply_move(state, attempt["move_id"])
                applied_total += 1
            else:
                require(number == len(game["attempts"]) - 1 and attempt["status"] == game["status"]
                        and game["status"] in ("failed", "interrupted"), "Unapplied attempt must end game")
        require(all(game[k] == v for k, v in state_trace(state).items()), "Final history/FEN/promoted/provenance")
        terminal = terminal_result(state)
        status = game["status"]
        if status == "completed":
            require(terminal is not None, "Completed game is nonterminal")
            winner = None if terminal.winner is None else "white" if terminal.winner else "black"
            outcome = "1/2-1/2" if winner is None else "1-0" if winner == "white" else "0-1"
            require(game["winner"] == winner and game["outcome"] == outcome
                    and game["ending_reason"] == terminal.ending_reason and game["error"] is None, "Rule outcome")
        else:
            require(game["winner"] is None and game["outcome"] is None, "Unfinished outcome")
            if status == "truncated":
                require(terminal is None and len(state.board.move_stack) >= game["ply_cap"]
                        and game["ending_reason"] == "ply_limit" and game["error"] is None, "Truncation boundary")
            else:
                require(status in ("failed", "interrupted") and game["error"] is not None
                        and game["ending_reason"] == ("execution_failed" if status == "failed" else "execution_interrupted"),
                        "Failure/interruption boundary")
    require(record["summary"] == summarize(games, left["player_id"], seed=config["seed"]), "Score accounting")
    return dict(schema="e01-replay-audit-v1", match_id=config["match_id"], games=len(games),
                pairs=len(selected), generated_plies=applied_total, completed_search_simulations=search_completions,
                measured_engine_nodes=engine_nodes, summary=record["summary"], passed=True)
