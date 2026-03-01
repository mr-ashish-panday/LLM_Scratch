#!/bin/bash
# ============================================================
# ESH Training — BASELINE (30K steps)
# Usage: bash go_baseline.sh
# ============================================================

echo "=========================================="
echo "  ESH BASELINE Training"
echo "=========================================="

echo "Installing packages..."
pip install transformers datasets bitsandbytes -q

echo ""
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo "VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")' 2>/dev/null)"

BATCH=$(python3 -c "
import torch
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
if vram >= 80: print(128)
elif vram >= 40: print(64)
elif vram >= 24: print(32)
else: print(16)
")

echo "batch_size=$BATCH"
echo "max_steps=30000"
echo "=========================================="

python -u run_ablation.py \
    --mode baseline \
    --max_steps 30000 \
    --batch_size $BATCH \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    --save_every 5000 \
    2>&1 | tee baseline_train.log
