#!/bin/bash
# ============================================================
# ESH Training — Auto-configures for any GPU
# Usage: bash go.sh
# ============================================================

set -e

echo "=========================================="
echo "  ESH Training — Full Setup"
echo "=========================================="

# ---- STEP 0: System Info ----
echo ""
echo "[0/4] System diagnostics..."
echo "  Python: $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "  CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")' 2>/dev/null || echo 'unknown')"
echo "  VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")' 2>/dev/null || echo 'unknown')"
which nvcc > /dev/null 2>&1 && echo "  nvcc: $(nvcc --version 2>&1 | grep release)" || echo "  nvcc: NOT FOUND"

# ---- STEP 1: Core deps ----
echo ""
echo "[1/4] Installing core packages..."
pip install transformers datasets bitsandbytes -q

# ---- STEP 2: MAMBA CUDA KERNELS ----
echo ""
echo "[2/4] Installing Mamba CUDA kernels..."
pip install causal-conv1d>=1.2.0 2>&1 | tail -3
pip install mamba-ssm 2>&1 | tail -5

# ---- STEP 3: VERIFY + AUTO-CONFIG ----
echo ""
echo "[3/4] Verifying and auto-configuring..."

# Python script to verify mamba and auto-set batch size
python3 << 'PYEOF'
import torch
import json

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"  GPU: {gpu_name}")
print(f"  VRAM: {vram_gb:.1f} GB")

# Check mamba-ssm
try:
    from mamba_ssm import Mamba
    model = Mamba(d_model=768, d_state=16, d_conv=4, expand=2).cuda().half()
    x = torch.randn(2, 64, 768, device='cuda', dtype=torch.float16)
    y = model(x)
    mamba_ok = True
    print(f"  mamba-ssm: WORKING (output shape: {y.shape})")
    print("  ✅ FAST MODE: 10-20x speedup active!")
    del model, x, y
    torch.cuda.empty_cache()
except Exception as e:
    mamba_ok = False
    print(f"  mamba-ssm: FAILED ({e})")
    print("  🐢 SLOW MODE: Using MambaMinimal fallback")

# Auto-configure batch size based on VRAM
# Conservative estimates to prevent OOM:
# Model (320M) + optimizer (8-bit) + gradients + activations
# ~4GB base + ~0.3GB per batch item
if vram_gb >= 80:
    batch = 128
elif vram_gb >= 40:
    batch = 64
elif vram_gb >= 24:
    batch = 32
elif vram_gb >= 16:
    batch = 16
else:
    batch = 8

# Set max steps based on mamba availability
max_steps = 25000 if mamba_ok else 30000

# Extra safety: if T4 or small GPU without mamba, reduce batch more
if vram_gb < 20 and not mamba_ok:
    batch = 16  # MambaMinimal uses more memory than mamba-ssm

config = {
    "batch_size": batch,
    "max_steps": max_steps,
    "mamba_ok": mamba_ok,
    "gpu": gpu_name,
    "vram_gb": round(vram_gb, 1)
}

with open("/tmp/train_config.json", "w") as f:
    json.dump(config, f)

print(f"\n  Auto-config:")
print(f"    batch_size: {batch}")
print(f"    max_steps: {max_steps}")
print(f"    mode: {'FAST (CUDA kernels)' if mamba_ok else 'SLOW (pure PyTorch)'}")
if not mamba_ok:
    est_hours = max_steps * (batch * 512 / 2500) / 3600
    print(f"    estimated time: ~{est_hours:.0f} hours")
PYEOF

# Read auto-config
BATCH=$(python3 -c "import json; print(json.load(open('/tmp/train_config.json'))['batch_size'])")
MAX_STEPS=$(python3 -c "import json; print(json.load(open('/tmp/train_config.json'))['max_steps'])")
MAMBA_OK=$(python3 -c "import json; print(json.load(open('/tmp/train_config.json'))['mamba_ok'])")

# ---- STEP 4: TRAIN ----
echo ""
echo "[4/4] Starting WIDTH_ONLY training..."
echo "  batch_size=$BATCH | max_steps=$MAX_STEPS"
echo "=========================================="

python -u run_ablation.py \
    --mode width_only \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH \
    --grad_accum 1 \
    --lambda-cost 0.005 \
    --penalty-warmup 1000 \
    --save_every 1000 \
    2>&1 | tee width_only_train.log

echo ""
echo "Training complete! Results saved to results/width_only/"
