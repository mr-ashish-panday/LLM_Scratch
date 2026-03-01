#!/bin/bash
# ============================================================
# ESH Training — BASELINE (no router, 50/50 forced split)
# Run on a SECOND account while width_only runs on the first
# Usage: bash go_baseline.sh
# ============================================================

set -e

echo "=========================================="
echo "  ESH BASELINE Training"
echo "=========================================="

# ---- System Info ----
echo ""
echo "System diagnostics..."
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo "  VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")' 2>/dev/null)"

# ---- Install deps ----
echo ""
echo "Installing packages..."
pip install transformers datasets bitsandbytes -q

# Try mamba-ssm (won't hurt to try)
pip install causal-conv1d>=1.2.0 -q 2>/dev/null || true
pip install mamba-ssm -q 2>/dev/null || true

# ---- Auto-config batch size ----
BATCH=$(python3 -c "
import torch
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
if vram >= 80: print(128)
elif vram >= 40: print(64)
elif vram >= 24: print(32)
elif vram >= 16: print(16)
else: print(8)
")

echo "  batch_size=$BATCH"

# ---- Train BASELINE ----
echo ""
echo "Starting BASELINE training (30K steps)..."
echo "=========================================="

python -u run_ablation.py \
    --mode baseline \
    --max_steps 30000 \
    --batch_size $BATCH \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    --save_every 1000 \
    2>&1 | tee baseline_train.log

echo "Training complete! Results saved to results/baseline/"
