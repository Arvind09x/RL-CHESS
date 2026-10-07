"""Replay the review fixtures and emit portable deterministic fingerprints."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from chess_rl.adapter import encode_position, legal_mask, make_state, move_to_id, terminal_result
from chess_rl.contracts import AdapterConfig


def summarize(record):
    state = make_state(record["start_fen"], record["moves_uci"],
                       history_complete=record["history_complete"])
    encoded = encode_position(state)
    mask = legal_mask(state)
    terminal = terminal_result(state)
    versions = asdict(AdapterConfig())
    versions["wdl_order"] = list(versions["wdl_order"])
    return {
        "name": record["name"],
        "versions": versions,
        "root_fen": state.board.fen(en_passant="fen"),
        "history_complete": encoded.history_complete,
        "retained_plies": len(state.board.move_stack),
        "available_snapshots": encoded.available_snapshots,
        "feature_shape": list(encoded.features.shape),
        "feature_dtype": str(encoded.features.dtype),
        "features_sha256": hashlib.sha256(encoded.features.astype("<f4").tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "legal_moves": {move.uci(): move_to_id(state, move) for move in sorted(state.board.legal_moves, key=lambda move: move.uci())},
        "terminal": None if terminal is None else {
            "winner": terminal.winner, "ending_reason": terminal.ending_reason,
            "wdl": terminal.wdl.tolist(), "value": terminal.value,
            "history_complete": terminal.history_complete, "priors": terminal.priors,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/adapter_records.json")
    args = parser.parse_args()
    records = json.loads(args.fixtures.read_text())
    summaries = [summarize(record) for record in records]
    for record, summary in zip(records, summaries):
        if summary != record["expected"]:
            raise ValueError(f"Replay differs from frozen acceptance fixture: {record['name']}")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
