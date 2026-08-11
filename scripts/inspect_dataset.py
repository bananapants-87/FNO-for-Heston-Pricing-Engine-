"""Inspect the prepared dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.dataset import HestonDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the prepared Heston dataset")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "final"))
    args = parser.parse_args()

    dataset = HestonDataset(
        inputs_path=Path(args.data_dir) / "inputs.pt",
        targets_path=Path(args.data_dir) / "targets.pt",
        s_grid_path=Path(args.data_dir) / "s_grid.pt",
        v_grid_path=Path(args.data_dir) / "v_grid.pt",
    )

    x, y = dataset[0]
    print(f"Samples: {len(dataset)}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Target shape: {tuple(y.shape)}")
    print(f"S grid: {tuple(dataset.s_grid.shape)}")
    print(f"v grid: {tuple(dataset.v_grid.shape)}")


if __name__ == "__main__":
    main()
