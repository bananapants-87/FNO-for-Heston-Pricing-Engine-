"""Run simple inference benchmarks for the Heston FNO model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from heston_fno.benchmarking.speedup import time_fno_inference
from heston_fno.data.dataset import HestonDataset
from heston_fno.models.fno import build_model_from_config


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Heston FNO inference speed"
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "fno.yaml"),
    )

    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "final"),
    )

    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "heston_fno.pth"),
    )

    args = parser.parse_args()

    # Load configuration
   
    config = load_config(args.config)

    batch_size = int(
        config.get("benchmarking", {}).get(
            "batch_size",
            config.get("training", {}).get("batch_size"),
        )
    )

    # Load dataset

    dataset = HestonDataset(
        inputs_path=Path(args.data_dir) / "inputs.pt",
        targets_path=Path(args.data_dir) / "targets.pt",
        s_grid_path=Path(args.data_dir) / "s_grid.pt",
        v_grid_path=Path(args.data_dir) / "v_grid.pt",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Take one batch for the benchmark
    batch_x, _ = next(iter(loader))

    print(f"Dataset size: {len(dataset)}")
    print(f"Benchmark batch size: {batch_size}")
    print(f"Input shape: {tuple(batch_x.shape)}")

    # Select device

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Build model
   
    model = build_model_from_config(config).to(device)

    print("Model created")

    # Load trained checkpoint
    checkpoint = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
)
    state_dict = checkpoint["model_state_dict"]

    # Remove PyTorch metadata if it was serialized as a state_dict key
    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)

    model.eval()

    print("Checkpoint loaded")
    print("Model ready for inference")
 
    # Benchmark inference
   
    avg_time = time_fno_inference(
        model,
        batch_x,
        device=device,
    )

    print()
    print("=" * 60)
    print("FNO INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Average inference time: {avg_time:.6f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()