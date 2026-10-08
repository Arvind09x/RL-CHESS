"""D05 must not pull model/search/Torch into CPU data preparation."""
import importlib.util
import sys

import chess_rl.stockfish_worker

assert importlib.util.find_spec("torch") is None
assert "torch" not in sys.modules
assert not any(name.startswith("chess_rl.model") or name.startswith("chess_rl.search") for name in sys.modules)
print("D05 Torch/model/search-free CPU import boundary passed")
