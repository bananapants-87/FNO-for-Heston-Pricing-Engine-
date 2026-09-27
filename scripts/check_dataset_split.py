"""Prepare and split the training dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.preprocessing import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Heston dataset and report split sizes")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "final"))
    args = parser.parse_args()

    train_set, val_set, test_set, _, _ = prepare_datasets(args.data_dir)
    print(f"Train samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")


if __name__ == "__main__":
    main()
