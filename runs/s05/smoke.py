"""Small fixed CPU suite: actual dummy games, boundaries and injected faults."""
from collections import Counter
import json
from pathlib import Path
import sys

import chess

from chess_rl.adapter import make_state, terminal_result
from chess_rl.search.evaluator import evaluate
from chess_rl.search.game import run_game
from chess_rl.search.puct import Node
from chess_rl.search.runner import search
from tests.test_game_runner import (BLACK_MATE, LONG_OPENING, WHITE_MATE,
                                    fault_evaluator, line_evaluator)

CYCLE = "g1f3 g8f6 f3g1 f6g8".split()


def main():
    out = Path(sys.argv[1])
    out.mkdir()
    specs = [
        ("start", "dummy-game", make_state(), 16, 24, None),
        ("opening", "dummy-game", make_state(moves_uci=LONG_OPENING), 16, 22, None),
        ("white-mate", "dummy-game", make_state(WHITE_MATE), 64, 1, None),
        ("black-mate", "dummy-game", make_state(BLACK_MATE), 64, 1, None),
        ("reached-halfmove-on-last-ply", "dummy-game", make_state(
            "4k3/8/8/8/8/8/8/R3K3 w - - 99 1"), 16, 1, None),
        ("terminal-checkmate", "terminal-boundary", make_state(
            moves_uci="f2f3 e7e5 g2g4 d8h4".split()), 4, 4, None),
        ("terminal-stalemate", "terminal-boundary", make_state(
            "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"), 4, 0, None),
        ("terminal-repetition", "terminal-boundary", make_state(moves_uci=CYCLE * 2), 4, 8, None),
        ("terminal-halfmove", "terminal-boundary", make_state(
            "4k3/8/8/8/8/8/8/R3K3 w - - 100 1"), 4, 0, None),
        ("last-ply-repetition", "controlled-boundary", make_state(
            moves_uci=(CYCLE * 2)[:-1]), 4, 8, "f6g8"),
        ("last-ply-stalemate", "controlled-boundary", make_state(WHITE_MATE), 4, 1, "g6f7"),
        ("intended-repetition", "intended-claim", make_state(
            moves_uci=(CYCLE * 2)[:-1]), 1, 8, None),
        ("intended-halfmove", "intended-claim", make_state(
            "4k3/8/8/8/8/8/8/R3K3 w - - 99 1"), 1, 1, None),
        ("search-failure", "injected-fault", make_state(), 4, 10, RuntimeError),
        ("search-interruption", "injected-fault", make_state(), 4, 10, KeyboardInterrupt),
    ]
    manifest = []
    for name, category, state, budget, cap, control in specs:
        manifest.append(dict(game_id=name, category=category, initial_fen=state.initial_fen,
                             opening_prefix=[m.uci() for m in state.board.move_stack],
                             history_complete=state.history_complete,
                             simulation_budget=budget, total_retained_ply_cap=cap,
                             control=control.__name__ if isinstance(control, type) else control))
    (out / 'configuration.json').write_text(json.dumps(dict(
        specs=manifest, selection='evaluation', exploration_noise=False, rng=None,
        c_puct=1.5, unvisited_value=0, fault_evaluator_call=9,
        limitations=['Fixed bounded plumbing suite; no playing-strength inference',
                     'Setup-FEN tactical games have incomplete prior history',
                     'No stochastic games, full self-play schedule, learning or resume protocol']), indent=2) + '\n')
    records, known = [], []
    for name, category, state, budget, cap, control in specs:
        if isinstance(control, type):
            evaluator = fault_evaluator(control, 9)
            evaluator_id = f'injected-{control.__name__}-call-9'
        elif isinstance(control, str):
            evaluator = line_evaluator(control)
            evaluator_id = 'controlled-onehot-' + control
        else:
            evaluator, evaluator_id = evaluate, 's01-uniform-dummy-v1'
        record = run_game(state, budget, ply_cap=cap, evaluator=evaluator,
                          evaluator_id=evaluator_id)
        records.append(dict(game_id=name, category=category, record=record.to_dict()))
        if name in {'white-mate', 'black-mate'}:
            root = Node(state)
            result = search(root, budget)
            edge = root.edges[result.move_id]
            terminal = terminal_result(edge.child.state)
            assert terminal is not None and terminal.winner == state.board.turn
            assert edge.value_sum == edge.visit_count and edge.child.result.value == -1
            assert record.status == 'completed' and record.ending_reason == 'CHECKMATE'
            known.append(dict(game_id=name, simulation_budget=budget,
                              completed_simulations=result.completed_simulations,
                              selected_uci=result.move.uci(), selected_id=result.move_id,
                              parent_n=edge.visit_count, parent_w=edge.value_sum,
                              parent_q=edge.value_sum / edge.visit_count,
                              child_value=edge.child.result.value,
                              winner=record.winner, evaluator=evaluator_id))
    (out / 'games.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    (out / 'known-positions.json').write_text(json.dumps(known, indent=2) + '\n')
    dummy = [r['record'] for r in records if r['category'] == 'dummy-game']
    summary = dict(total_records=len(records), status_counts=dict(Counter(
        r['record']['status'] for r in records)), outcome_counts=dict(Counter(
        r['record']['outcome'] for r in records if r['record']['status'] == 'completed')),
        dummy_games=len(dummy), dummy_status_counts=dict(Counter(r['status'] for r in dummy)),
        dummy_outcome_counts=dict(Counter(r['outcome'] for r in dummy if r['status'] == 'completed')),
        actual_generated_plies=sum(len(r['record']['moves_uci']) - len(r['record']['opening_prefix'])
                                   for r in records),
        completed_search_simulations=sum(a['completed_simulations'] for r in records
                                         for a in r['record']['attempts']),
        deliberate_stochastic_configuration=None,
        limitation='Dummy/controlled games establish search plumbing only; no playing-strength claim')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
