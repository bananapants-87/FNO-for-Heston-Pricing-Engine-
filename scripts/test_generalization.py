import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heston_fno.data.dataset import HestonDataset
from heston_fno.evaluation.generalization import compare_carr_madan
from heston_fno.models.fno import HestonFNO


print("=" * 70)
print("TESTING generalization.py")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
dataset = HestonDataset(
    "data/final/inputs.pt",
    "data/final/targets.pt",
    "data/final/s_grid.pt",
    "data/final/v_grid.pt",
)

print("Dataset loaded ✅")
print("Samples:", len(dataset))


# ---------------------------------------------------------
# 2. Get one sample
# ---------------------------------------------------------
x, y = dataset[0]

print("Input shape :", tuple(x.shape))
print("Target shape:", tuple(y.shape))

s_grid = dataset.s_grid
v_grid = dataset.v_grid
params = dataset.params[0]

print("S grid shape :", tuple(s_grid.shape))
print("v grid shape:", tuple(v_grid.shape))
print("Params shape :", tuple(params.shape))
print("Params       :", params.tolist())


# ---------------------------------------------------------
# 3. Create FNO model
# ---------------------------------------------------------
model = HestonFNO(
    n_modes=(8, 8),
    hidden_channels=32,
    in_channels=7,
    out_channels=1,
)

print("Model created ✅")


# ---------------------------------------------------------
# 4. Load checkpoint
# ---------------------------------------------------------
checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location="cpu",
    weights_only=False,
)

state_dict = checkpoint["model_state_dict"]
state_dict.pop("_metadata", None)

model.load_state_dict(state_dict)

print("Checkpoint loaded ✅")


# ---------------------------------------------------------
# 5. Temporary reference function
# ---------------------------------------------------------
def reference_fn(params, s_grid, v_grid):
    """
    Temporary reference surface.

    We use the known target for this sample so that
    generalization.py can be tested without implementing
    Carr-Madan yet.
    """
    return y.squeeze(0)


# ---------------------------------------------------------
# 6. Run comparison
# ---------------------------------------------------------
result = compare_carr_madan(
    model=model,
    params=params,
    s_grid=s_grid,
    v_grid=v_grid,
    reference_fn=reference_fn,
    device="cpu",
)


# ---------------------------------------------------------
# 7. Display results
# ---------------------------------------------------------
print("\nResults")
print("-" * 70)

print(
    "Prediction shape :",
    tuple(result["prediction"].shape),
)

print(
    "Reference shape  :",
    tuple(result["reference"].shape),
)

print(
    "Absolute error   :",
    result["absolute_error"].item(),
)

print(
    "Relative error   :",
    result["relative_error"].item(),
)

print("\ngeneralization.py working ✅")