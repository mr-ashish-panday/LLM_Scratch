#!/bin/bash
# ============================================================
# ESH Training — MAMBA CUDA KERNEL INSTALLER + TRAINER
# This script MUST get mamba-ssm working or show exactly why not
# Usage: bash go.sh
# ============================================================

set -e  # Stop on any error

echo "=========================================="
echo "  ESH Training — Full Setup"
echo "=========================================="

# ---- STEP 0: System Info ----
echo ""
echo "[0/4] System diagnostics..."
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
which nvcc && echo "  nvcc: $(nvcc --version | tail -1)" || echo "  nvcc: NOT FOUND"

# ---- STEP 1: Core deps ----
echo ""
echo "[1/4] Installing core packages..."
pip install transformers datasets bitsandbytes -q

# ---- STEP 2: MAMBA CUDA KERNELS (the critical step) ----
echo ""
echo "[2/4] Installing Mamba CUDA kernels..."
echo "  This is the 10-20x speedup. If this fails, training will be very slow."
echo ""

# First install causal-conv1d (required by mamba-ssm)
echo "  → Installing causal-conv1d..."
pip install causal-conv1d>=1.2.0 2>&1 | tail -5
CONV_OK=$?

# Then install mamba-ssm
echo "  → Installing mamba-ssm..."
pip install mamba-ssm 2>&1 | tail -10
MAMBA_OK=$?

# ---- STEP 3: VERIFY ----
echo ""
echo "[3/4] Verifying mamba-ssm installation..."
python3 -c "
import torch
print(f'  PyTorch CUDA: {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')

try:
    from mamba_ssm import Mamba
    # Actually test it works with a forward pass
    model = Mamba(d_model=768, d_state=16, d_conv=4, expand=2).cuda().half()
    x = torch.randn(2, 64, 768, device='cuda', dtype=torch.float16)
    y = model(x)
    print(f'  Mamba output shape: {y.shape}')
    print('  ✅ SUCCESS: mamba-ssm CUDA kernels WORKING! 10-20x speedup active!')
    FAST = True
except ImportError as e:
    print(f'  ❌ IMPORT FAILED: {e}')
    FAST = False
except Exception as e:
    print(f'  ❌ RUNTIME FAILED: {e}')
    FAST = False

if not FAST:
    print('')
    print('  Falling back to MambaMinimal (pure PyTorch).')
    print('  Training will work but will be 10-20x slower.')
    print('  Consider reducing --max_steps to 3000-5000.')
"

# ---- STEP 4: TRAIN ----
echo ""
echo "[4/4] Starting WIDTH_ONLY training..."
echo "=========================================="

# Check if mamba-ssm is available to set appropriate max_steps
MAMBA_INSTALLED=$(python3 -c "
try:
    from mamba_ssm import Mamba; print('yes')
except:
    print('no')
" 2>/dev/null)

if [ "$MAMBA_INSTALLED" = "yes" ]; then
    echo "  🚀 FAST MODE: mamba-ssm active, targeting 25000 steps"
    MAX_STEPS=25000
    BATCH=128
else
    echo "  🐢 SLOW MODE: MambaMinimal fallback, targeting 3000 steps"
    MAX_STEPS=3000
    BATCH=128
fi

python -u run_ablation.py \
    --mode width_only \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    2>&1 | tee width_only_train.log
