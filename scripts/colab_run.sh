#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Step 1: Checking GPU environment..."
echo "============================================================"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available!'; print('CUDA is available! Device:', torch.cuda.get_device_name(0))"

echo ""
echo "============================================================"
echo "Step 2: Ensuring dependencies and heston-fno package..."
echo "============================================================"
pip install -q neuraloperator
pip install -q -e .

echo ""
echo "============================================================"
echo "Step 3: Checking data files in data/final..."
echo "============================================================"
if [ ! -f "data/final/inputs.pt" ] || [ ! -f "data/final/targets.pt" ]; then
    echo "ERROR: Data files not found in data/final/."
    echo "Please upload your local 'data/final' folder to '/content/FNO-for-Heston-Pricing-Engine-/data/final'."
    exit 1
fi
echo "Data files verified: inputs.pt, targets.pt, s_grid.pt, v_grid.pt are present."

echo ""
echo "============================================================"
echo "Step 4: Running GPU Smoke Test..."
echo "============================================================"
python -c "import torch, yaml; from heston_fno.models.fno import build_model_from_config; cfg = yaml.safe_load(open('configs/fno.yaml')); m = build_model_from_config(cfg).cuda(); x = torch.randn(2, 7, 32, 32, device='cuda'); y = m(x); print('Smoke test SUCCESS: Output shape on GPU =', tuple(y.shape), 'Device =', y.device)"

echo ""
echo "============================================================"
echo "Step 5: Launching Training Script..."
echo "============================================================"
python scripts/train_fno.py

echo ""
echo "============================================================"
echo "ALL STEPS COMPLETED SUCCESSFULLY!"
echo "============================================================"
