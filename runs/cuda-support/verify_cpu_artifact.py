"""Read the unchanged accepted T03 checkpoint/data through the new shared loader."""
from dataclasses import asdict
import json
from pathlib import Path
import torch

from scripts.run_t03 import arrays, tensor_hashes
from scripts.stage_cuda_data import identity
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import ModelConfig, load_inference
from chess_rl.training.tiny_overfit import measure

ROOT = Path(__file__).resolve().parents[2]
STAGED = ROOT.parent/"external-artifacts/cuda-support-portable"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
expected = json.loads((ROOT/"runs/t03/expected-reload.json").read_text())
plan = json.loads((ROOT/"runs/t03/frozen-plan.json").read_text())
path = STAGED/"checkpoint.pt"
assert identity(path) == expected["checkpoint_identity"]
loaded = load_inference(path, expected_config=ModelConfig(**plan["model"]))
assert tensor_hashes(loaded.model.state_dict()) == expected["model"]
assert asdict(loaded.config) == expected["config"] and loaded.provenance == expected["provenance"]
config = json.loads((STAGED/"data-config.json").read_text())
data = open_dataset(**{k:STAGED/v for k,v in config["paths"].items()}, split="train")
batch = next(data.batches(plan["batch_size"]))
assert arrays(batch) == plan["batch_array_sha256"]
assert list(map(list,batch.references)) == plan["references"]
actual = measure(loaded.model, batch)
assert actual == expected["metrics"]
report = dict(checkpoint=identity(path), model_tensors_exact=True, metrics_exact=True,
    metrics=actual, checkpoint_id=loaded.checkpoint_id, updates_performed=0,
    portable_data_verified=True)
(ROOT/"runs/cuda-support/accepted-cpu-artifact.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
