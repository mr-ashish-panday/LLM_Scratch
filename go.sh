#!/bin/bash
# ESH Training — WIDTH_ONLY (30K steps)
echo "=========================================="
echo "  ESH WIDTH_ONLY Training"
echo "=========================================="
echo "Installing packages..."
pip install transformers datasets bitsandbytes -q

echo ""
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" 2>/dev/null || echo "GPU detection failed"

BATCH=$(python3 -c "
import torch
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
if vram >= 80: print(128)
elif vram >= 40: print(64)
elif vram >= 24: print(32)
else: print(16)
" 2>/dev/null || echo 16)

echo "batch_size=$BATCH | max_steps=30000"
echo "=========================================="

python -u run_ablation.py \
    --mode width_only \
    --max_steps 30000 \
    --batch_size $BATCH \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    --save_every 5000 \
    2>&1 | tee width_only_train.log
