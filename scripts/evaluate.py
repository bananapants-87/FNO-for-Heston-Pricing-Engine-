"""Evaluate trained models on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchgen import model
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.preprocessing import prepare_datasets
from heston_fno.evaluation.pricing import pricing_error
from heston_fno.models.fno import build_model_from_config


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Heston FNO model")
    parser.add_argument("--config", default=str(ROOT / "configs" / "fno.yaml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "final"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "heston_fno.pth"))
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config.get("training", {})
    _, _, test_set, _, target_scaler = prepare_datasets(
        args.data_dir,
        split=training_cfg.get("split"),
        batch_size=int(training_cfg.get("stats_batch_size", training_cfg.get("batch_size"))),
        seed=training_cfg.get("seed"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)

    rmse, relative_error = pricing_error(model, test_set, device=device, target_scaler=target_scaler)
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test relative error: {relative_error:.6f}")


if __name__ == "__main__":
    main()
