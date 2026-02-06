# ESH: Entropy-Steered Hybridization

A dynamic hybrid LLM architecture that routes tokens between SSM (Mamba) and Attention paths based on learned complexity signals.

## Key Features

- **Soft Entropy Routing**: Differentiable α-blending between SSM and Attention paths
- **Gated Attention**: Sigmoid-gated attention with FlashAttention-2 support
- **Top-1 MoE**: Memory-efficient mixture of experts with load balancing
- **LayerScale**: Gradient balancing for stable training
- **Gradient Checkpointing**: Fits ~500M parameters in 12GB VRAM

## Architecture

```
Input → Embedding → [ESHBlock × N] → LM Head → Output

ESHBlock:
  x → Norm → Router(x) → α
  x → [α·Attention(x) + (1-α)·SSM(x)] → LayerScale → +Residual
  x → Norm → MoE(x) → LayerScale → +Residual → output
```

## Installation

```bash
pip install torch transformers datasets
pip install mamba-ssm  # Optional, for Mamba-2 support
```

## Quick Start

```python
from esh import ESHModel, ESHConfig
from esh.model import esh_medium

# Create model (~350M params, fits in 12GB VRAM)
config = esh_medium()
model = ESHModel(config)
model.print_model_size()

# Training
from esh.training import Trainer, TrainingConfig

trainer = Trainer(
    model=model,
    train_dataloader=your_dataloader,
    config=TrainingConfig(use_amp=True),
)
trainer.train()
```

## Model Sizes

| Config | Params | VRAM (fp16) | Target Dataset |
|--------|--------|-------------|----------------|
| `esh_small()` | ~125M | ~4GB | Debugging |
| `esh_medium()` | ~350M | ~8GB | Main target |
| `esh_large()` | ~500M | ~12GB | Maximum capacity |

## Training on 12GB VRAM

Key settings for memory efficiency:
- `use_checkpoint=True` (gradient checkpointing)
- `use_flash=True` (FlashAttention-2)
- `amp_dtype="bfloat16"` (mixed precision)
- `gradient_accumulation_steps=8` (reduce batch size)

## Files

```
esh/
├── __init__.py       # Package exports
├── layers.py         # Core layers (Router, Attention, SSM, MoE)
├── model.py          # ESHModel and configs
└── training.py       # Training utilities
train.py              # Example training script
```

## Citation

```
@article{esh2026,
  title={Entropy-Steered Hybridization: Dynamic Compute Allocation for Efficient Language Models},
  author={...},
  year={2026}
}
```
