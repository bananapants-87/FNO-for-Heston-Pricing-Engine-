"""Quick integration test for evaluation helper modules."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.dataset import HestonDataset
from heston_fno.models.fno import build_model_from_config
from heston_fno.evaluation.errors import summarize_errors
from heston_fno.evaluation.greeks import compute_delta, compute_vega
from heston_fno.data.normalization import StandardScaler


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    print("=" * 70)
    print("TESTING EVALUATION HELPERS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load config
    # ---------------------------------------------------------
    config = load_config(ROOT / "configs" / "fno.yaml")
    print("1. Config loaded")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    data_dir = ROOT / "data" / "final"

    dataset = HestonDataset(
        inputs_path=data_dir / "inputs.pt",
        targets_path=data_dir / "targets.pt",
        s_grid_path=data_dir / "s_grid.pt",
        v_grid_path=data_dir / "v_grid.pt",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    x, y = next(iter(loader))

    print("2. Dataset loaded")
    print(f"   Input shape : {tuple(x.shape)}")
    print(f"   Target shape: {tuple(y.shape)}")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"3. Device: {device}")

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    model = build_model_from_config(config).to(device)

    print("4. Model created")

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    checkpoint = torch.load(
        ROOT / "checkpoints" / "heston_fno.pth",
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)

    print("5. Checkpoint loaded")

    # ---------------------------------------------------------
    # Test errors.py
    # ---------------------------------------------------------
    model.eval()

    with torch.no_grad():
        prediction = model(x.to(device))

    prediction = prediction.cpu()

    summary = summarize_errors(
        prediction,
        y,
    )

    print()
    print("6. errors.py")
    print(f"   MAE          : {summary.to_dict()['mean_absolute_error']:.6f}")
    print(f"   MRE          : {summary.to_dict()['mean_relative_error']:.6f}")
    print(f"   RMSE         : {summary.to_dict()['rmse']:.6f}")
    print(f"   Max AE       : {summary.to_dict()['max_absolute_error']:.6f}")

    print("   errors.py working ✅")

    # ---------------------------------------------------------
    # Create simple scalers
    # ---------------------------------------------------------
    input_scaler = StandardScaler().fit(x)
    target_scaler = StandardScaler().fit(y)

    # ---------------------------------------------------------
    # Test greeks.py
    # ---------------------------------------------------------
    delta = compute_delta(
        model=model,
        x=x,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        device=device,
    )

    print()
    print("7. compute_delta()")
    print(f"   Delta shape: {tuple(delta.shape)}")
    print(f"   Delta mean : {delta.mean().item():.6f}")
    print("   compute_delta() working ✅")

    vega = compute_vega(
        model=model,
        x=x,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        device=device,
    )

    print()
    print("8. compute_vega()")
    print(f"   Vega shape : {tuple(vega.shape)}")
    print(f"   Vega mean  : {vega.mean().item():.6f}")
    print("   compute_vega() working ✅")

    print()
    print("=" * 70)
    print("EVALUATION HELPER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()