"""Train the Heston FNO surrogate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset


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
    print("1. Config loaded")
    training_cfg = config.get("training", {})
    print("Training config loaded")
    print("Device initialized")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, _, input_scaler, target_scaler = prepare_datasets(
        args.data_dir,
        split=training_cfg.get("split"),
        batch_size=int(training_cfg.get("stats_batch_size", training_cfg.get("batch_size"))),
        seed=training_cfg.get("seed"),
    )

    max_train_samples = training_cfg.get("max_train_samples", None)

    if max_train_samples is not None:
        train_set = Subset(train_set, range(int(max_train_samples)))

    print("2. Dataset prepared")

    batch_size = int(training_cfg.get("batch_size"))
    num_workers = int(training_cfg.get("num_workers"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("3. DataLoaders created")

    model = build_model_from_config(config).to(device)
    print("4. Model created")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg.get("lr", 1e-3)))

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(training_cfg.get("epochs")),
        checkpoint_path=args.checkpoint,
    )
    print("5. Training finished")

    checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False,
    )
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
