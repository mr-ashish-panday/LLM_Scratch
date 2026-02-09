"""
ESH Random Router Baseline
===========================
Ablation study: Replace learned Entropy Router with random routing.
This proves the Entropy Router is the key innovation.

If Random Router shows:
- Worse perplexity than ESH
- No α differentiation between Math vs Story prompts

Then the Entropy Router is the "secret sauce."
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import time
import math

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from esh.model import ESHConfig
from esh.training import Trainer, TrainingConfig
from data_loader import create_mixed_dataloader, create_mixed_eval_dataloader


# =============================================================================
# Random Router (Ablation)
# =============================================================================

class RandomRouter(nn.Module):
    """
    Random routing baseline - outputs random α values.
    No learning, no complexity detection.
    
    This proves that the learned Entropy Router is essential.
    """
    
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        # Fixed random baseline α (no learning)
        self.register_buffer('base_alpha', torch.tensor(0.5))
        # Track last alpha for stats
        self._last_alpha = None
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, L, D = x.shape
        # Random α per token, uniform [0.3, 0.7] to avoid extremes
        alpha = 0.3 + 0.4 * torch.rand(B, L, 1, device=x.device)
        self._last_alpha = alpha
        return alpha
    
    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """Return routing statistics (required by ESHBlock)."""
        if self._last_alpha is not None:
            avg_alpha = self._last_alpha.mean().item()
        else:
            avg_alpha = 0.5
        return {
            "alpha_mean": avg_alpha,  # Required by training.py
            "attention_ratio": avg_alpha,
            "temperature": 1.0,
            "entropy": 0.0,  # No entropy for random router
        }


# =============================================================================
# Patched ESH Model with Random Router
# =============================================================================

def create_random_router_model():
    """Create ESH model but replace all routers with RandomRouter."""
    from esh import ESHModel
    from esh.model import ESHConfig
    
    # Same config as Phase 2
    config = ESHConfig(
        d_model=768,
        n_layers=8,
        n_heads=12,
        n_experts=8,
        expert_dim=3072,
        max_seq_len=2048,
        use_checkpoint=True,
        dropout=0.0,
        layer_scale_init=1e-5,
    )
    
    model = ESHModel(config)
    
    # Replace all routers with RandomRouter
    for i, block in enumerate(model.blocks):
        block.router = RandomRouter(config.d_model)
        print(f"  Layer {i}: Replaced SoftEntropyRouter with RandomRouter")
    
    return model, config


def main():
    print("=" * 70)
    print("ESH Random Router Baseline (Ablation Study)")
    print("=" * 70)
    
    # =========================================================================
    # Configuration (Same as Phase 2)
    # =========================================================================
    
    MAX_STEPS = 25_000
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 2
    GRAD_ACCUM = 16
    
    TINY_RATIO = 0.70
    WIKI_RATIO = 0.20
    GSM_RATIO = 0.10
    
    SAVE_INTERVAL = 5000
    EVAL_INTERVAL = 2000
    
    OUTPUT_DIR = "./esh_random_baseline_checkpoints"
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create model with Random Router
    print("\nCreating model with Random Router...")
    model, config = create_random_router_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.2f}M")
    print(f"(Note: Random Router has no learnable weights)")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loaders (SAME as Phase 2)
    print("\nCreating mixed complexity data loaders...")
    print(f"  TinyStories: {TINY_RATIO*100:.0f}%")
    print(f"  WikiText-103: {WIKI_RATIO*100:.0f}%")
    print(f"  GSM8K: {GSM_RATIO*100:.0f}%")
    
    train_loader = create_mixed_dataloader(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
        tiny_ratio=TINY_RATIO,
        wiki_ratio=WIKI_RATIO,
        gsm_ratio=GSM_RATIO,
    )
    
    eval_loader = create_mixed_eval_dataloader(
        tokenizer=tokenizer,
        batch_size=4,
        max_length=MAX_SEQ_LEN,
        samples_per_source=100,
    )
    
    # Training config (SAME as Phase 2)
    training_config = TrainingConfig(
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=2500,
        burn_in_steps=1000,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        output_dir=OUTPUT_DIR,
        use_amp=True,
        amp_dtype="bfloat16",
        use_8bit_optimizer=True,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {MAX_STEPS:,}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Router: RANDOM (ablation)")
    
    # =========================================================================
    # Training
    # =========================================================================
    
    model.to(device)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        config=training_config,
        eval_dataloader=eval_loader,
    )
    
    print("\n" + "=" * 70)
    print("Starting Random Router Baseline Training")
    print("=" * 70)
    print("\nExpected: Higher loss/PPL than ESH, no α differentiation")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pt")
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")
    print("\nNext: Run generate.py with this checkpoint to compare α values")


if __name__ == "__main__":
    main()
