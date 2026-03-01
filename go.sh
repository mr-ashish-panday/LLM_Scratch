#!/bin/bash
# ============================================================
# ESH Training — Full Setup + Training Script for Lightning AI
# Usage: bash go.sh
# ============================================================

echo "=========================================="
echo "  ESH Training — Lightning AI Setup"
echo "=========================================="

# Step 1: Core dependencies
echo "[1/3] Installing core packages..."
pip install transformers datasets bitsandbytes -q

# Step 2: Install CUDA Mamba kernels (THE SPEED FIX)
echo "[2/3] Installing Mamba CUDA kernels..."
echo "  This is the key speedup: replaces Python for-loop with CUDA parallel scan"
pip install causal-conv1d -q 2>/dev/null
pip install mamba-ssm -q 2>/dev/null

# Verify installation
python3 -c "from mamba_ssm import Mamba; print('SUCCESS: mamba-ssm CUDA kernels installed! 10-20x speedup enabled.')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: mamba-ssm failed to install. Falling back to pure-PyTorch (slower)."
    echo "  Trying from source..."
    pip install mamba-ssm --no-build-isolation 2>/dev/null
    python3 -c "from mamba_ssm import Mamba; print('SUCCESS: mamba-ssm installed from source!')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  FAILED: Will use MambaMinimal (pure PyTorch). Training will be slower."
    fi
fi

# Step 3: Start training
echo "[3/3] Starting width_only training..."
echo "=========================================="
python -u run_ablation.py \
    --mode width_only \
    --max_steps 25000 \
    --batch_size 128 \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    2>&1 | tee width_only_train.log
