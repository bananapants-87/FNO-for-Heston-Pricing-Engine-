"""Train the Heston FNO surrogate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.preprocessing import prepare_datasets
from heston_fno.models.fno import build_model_from_config
from heston_fno.training.train import train_model


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Heston FNO surrogate")
    parser.add_argument("--config", default=str(ROOT / "configs" / "fno.yaml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "final"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "heston_fno.pth"))
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config.get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, _, input_scaler, target_scaler = prepare_datasets(
        args.data_dir,
        split=training_cfg.get("split", (0.8, 0.1, 0.1)),
        batch_size=int(training_cfg.get("stats_batch_size", training_cfg.get("batch_size", 16))),
        seed=training_cfg.get("seed", 42),
    )

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_config(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg.get("lr", 1e-3)))

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(training_cfg.get("epochs", 50)),
        checkpoint_path=args.checkpoint,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint.update(
        {
            "config": config,
            "history": history,
            "input_scaler": input_scaler.state_dict(),
            "target_scaler": target_scaler.state_dict(),
        }
    )
    torch.save(checkpoint, args.checkpoint)


if __name__ == "__main__":
    main()
