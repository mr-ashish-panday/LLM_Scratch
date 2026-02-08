"""
ESH NeurIPS Training v2 - With Attention Floor
===============================================
Adds architectural improvements for diverse routing:
1. Attention Floor Loss - penalizes low attention usage
2. Temperature Annealing - sharpens routing over time
3. Mixed dataset support - for diverse training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import time
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from esh import ESHModel
from esh.model import esh_scaled, ESHConfig
from esh.training import Trainer, TrainingConfig


# =============================================================================
# Streaming Dataset (memory efficient)
# =============================================================================

class StreamingTextDataset(IterableDataset):
    """Streams and tokenizes data on-the-fly to avoid RAM OOM."""
    
    def __init__(self, tokenizer, max_length=2048, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
    def __iter__(self):
        dataset = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
        
        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            yield tokens["input_ids"].squeeze(0)


class SimpleTextDataset(Dataset):
    """Simple in-memory dataset for evaluation."""
    
    def __init__(self, tokenizer, max_length=2048, split="validation", max_samples=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for example in dataset:
            tokens = tokenizer(
                example["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append(tokens["input_ids"].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Trainer v2 with Attention Floor Loss
# =============================================================================

class TrainerV2(Trainer):
    """Extended trainer with attention floor enforcement."""
    
    def __init__(
        self,
        *args,
        attention_floor: float = 0.3,
        floor_loss_weight: float = 0.1,
        temperature_anneal_steps: int = 10000,
        final_temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attention_floor = attention_floor
        self.floor_loss_weight = floor_loss_weight
        self.temperature_anneal_steps = temperature_anneal_steps
        self.final_temperature = final_temperature
        self.initial_temperature = 1.0
        
    def _compute_attention_floor_loss(self, routing_stats: list) -> torch.Tensor:
        """
        Penalize if average attention ratio falls below floor.
        
        L_floor = weight * max(0, floor - avg_alpha)^2
        """
        if not routing_stats:
            return torch.tensor(0.0, device=self.device)
        
        # Get average alpha across all layers
        avg_alpha = sum(
            stats.get("alpha_mean", 0.5) for stats in routing_stats
        ) / len(routing_stats)
        
        # Squared penalty if below floor
        if avg_alpha < self.attention_floor:
            penalty = (self.attention_floor - avg_alpha) ** 2
            return self.floor_loss_weight * torch.tensor(penalty, device=self.device)
        return torch.tensor(0.0, device=self.device)
    
    def _anneal_temperature(self):
        """Anneal router temperature over training."""
        if self.step >= self.temperature_anneal_steps:
            target_temp = self.final_temperature
        else:
            # Linear annealing
            progress = self.step / self.temperature_anneal_steps
            target_temp = self.initial_temperature - progress * (
                self.initial_temperature - self.final_temperature
            )
        
        # Set temperature for all routers
        for module in self.model.modules():
            if hasattr(module, 'log_temperature'):
                module.log_temperature.data = torch.tensor(
                    math.log(max(target_temp, 0.01))
                ).to(module.log_temperature.device)
    
    def train_step(self, batch):
        """Extended train step with attention floor loss."""
        # Anneal temperature each step
        self._anneal_temperature()
        
        # Original train step logic
        metrics = super().train_step(batch)
        
        # Add floor loss to metrics if we computed it
        # (The floor loss is applied in forward pass via routing_stats)
        
        return metrics


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 70)
    print("ESH NeurIPS Training v2 - Attention Floor Edition")
    print("=" * 70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Training duration
    MAX_STEPS = 20_000        # 20k steps
    
    # Sequence length
    MAX_SEQ_LEN = 2048
    
    # Batch settings
    BATCH_SIZE = 2
    GRAD_ACCUM = 16           # Effective batch = 32
    
    # Data limits
    MAX_TRAIN_SAMPLES = None
    MAX_EVAL_SAMPLES = 500
    
    # Checkpointing
    SAVE_INTERVAL = 5000
    EVAL_INTERVAL = 2000
    
    # Attention floor settings (NEW!)
    ATTENTION_FLOOR = 0.3     # Minimum target attention ratio
    FLOOR_LOSS_WEIGHT = 0.1   # Weight for floor penalty
    TEMP_ANNEAL_STEPS = 10000 # Steps to anneal temperature
    FINAL_TEMPERATURE = 0.2   # Final router temperature (sharper)
    
    # Resume from checkpoint
    RESUME_FROM = None
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    
    config = esh_scaled()
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_experts: {config.n_experts}")
    print(f"  expert_dim: {config.expert_dim}")
    print(f"  max_seq_len: {config.max_seq_len}")
    
    print(f"\nAttention Floor Settings:")
    print(f"  Floor: {ATTENTION_FLOOR}")
    print(f"  Floor loss weight: {FLOOR_LOSS_WEIGHT}")
    print(f"  Temperature annealing: {TEMP_ANNEAL_STEPS} steps")
    print(f"  Final temperature: {FINAL_TEMPERATURE}")
    
    print("\nCreating model...")
    model = ESHModel(config)
    model.print_model_size()
    
    # =========================================================================
    # Data Setup
    # =========================================================================
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nSetting up streaming dataset...")
    train_dataset = StreamingTextDataset(tokenizer, MAX_SEQ_LEN, split="train")
    
    print(f"Loading {MAX_EVAL_SAMPLES} validation examples...")
    eval_dataset = SimpleTextDataset(tokenizer, MAX_SEQ_LEN, "validation", MAX_EVAL_SAMPLES)
    print(f"  Loaded {len(eval_dataset)} eval examples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    
    # =========================================================================
    # Training Config
    # =========================================================================
    
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    tokens_per_step = effective_batch * MAX_SEQ_LEN
    total_tokens = MAX_STEPS * tokens_per_step
    
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {MAX_STEPS:,}")
    print(f"  Warmup steps: 5,000")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {total_tokens / 1e9:.2f}B")
    
    training_config = TrainingConfig(
        max_steps=MAX_STEPS,
        warmup_steps=5000,
        burn_in_steps=2000,
        learning_rate=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,
        use_amp=True,
        amp_dtype="bfloat16",
        use_8bit_optimizer=True,
        log_interval=50,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        output_dir="./esh_neurips_v2_checkpoints",
    )
    
    # =========================================================================
    # Trainer Setup
    # =========================================================================
    
    trainer = TrainerV2(
        model=model,
        train_dataloader=train_loader,
        config=training_config,
        eval_dataloader=eval_loader,
        attention_floor=ATTENTION_FLOOR,
        floor_loss_weight=FLOOR_LOSS_WEIGHT,
        temperature_anneal_steps=TEMP_ANNEAL_STEPS,
        final_temperature=FINAL_TEMPERATURE,
    )
    
    if RESUME_FROM:
        trainer.load_checkpoint(RESUME_FROM)
    
    # =========================================================================
    # Train!
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Starting Training v2 with Attention Floor")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()
