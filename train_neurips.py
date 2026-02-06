"""
ESH NeurIPS Training Script
===========================
100,000-step training schedule optimized for Virtual Scaling:
- 16-expert MoE (~1.5B capacity, ~500M active)
- 4096 sequence length
- 8-bit optimizer states
- Full TinyStories dataset

Usage:
    nohup python -u train_neurips.py > training_neurips.log 2>&1 &
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from esh import ESHModel, ESHConfig
from esh.model import esh_scaled
from esh.training import Trainer, TrainingConfig, print_memory_stats


class TextDataset(Dataset):
    """Simple text dataset with tokenization."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Tokenizing {len(texts)} examples...")
        for i, text in enumerate(texts):
            if i % 50000 == 0:
                print(f"  Progress: {i}/{len(texts)}")
            tokens = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append(tokens["input_ids"].squeeze(0))
        print("Tokenization complete!")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def load_tinystories(tokenizer, max_length: int, split: str, max_samples: int = None):
    """Load TinyStories dataset."""
    from datasets import load_dataset
    
    print(f"Loading TinyStories {split} split...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = [ex["text"] for ex in dataset]
    print(f"Loaded {len(texts)} examples from {split}")
    
    return TextDataset(texts, tokenizer, max_length)


def main():
    print("=" * 70)
    print("ESH NeurIPS Training - Virtual Scaling Edition")
    print("=" * 70)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Training duration  
    MAX_STEPS = 100_000      # 100k steps for thorough training
    
    # Sequence length (4096 for long-context)
    MAX_SEQ_LEN = 4096
    
    # Batch settings (conservative for 12GB)
    BATCH_SIZE = 1           # Small batch for 4096 seq len
    GRAD_ACCUM = 32          # Effective batch = 32
    
    # Data limits (None = full dataset)
    MAX_TRAIN_SAMPLES = None  # ~2.1M examples
    MAX_EVAL_SAMPLES = 1000
    
    # Checkpointing
    SAVE_INTERVAL = 10000    # Every 10k steps
    EVAL_INTERVAL = 5000     # Every 5k steps
    
    # Resume from checkpoint (set path or None)
    RESUME_FROM = None
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Model config (16 experts, 4096 seq len)
    model_config = esh_scaled()
    model_config.max_seq_len = MAX_SEQ_LEN
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_experts: {model_config.n_experts}")
    print(f"  expert_dim: {model_config.expert_dim}")
    print(f"  max_seq_len: {model_config.max_seq_len}")
    
    # Estimate parameters
    # With 16 experts, each expert adds: 3 * d_model * expert_dim params
    # Total MoE params per layer: 16 * 3 * 1024 * 4096 = ~201M
    # Total MoE params: 16 layers * 201M = ~3.2B just in experts!
    # But only 1/16 active = ~200M active MoE params
    
    print(f"\nParameter Estimates:")
    base_params = model_config.estimate_params()
    # Adjust for 16 experts vs 4
    extra_expert_params = (16 - 4) * model_config.n_layers * 3 * model_config.d_model * model_config.expert_dim
    total_params = base_params + extra_expert_params
    active_params = base_params  # Only 1 expert active
    
    print(f"  Total params (all experts): ~{total_params / 1e9:.2f}B")
    print(f"  Active params (1 expert): ~{active_params / 1e6:.0f}M")
    
    # Create model
    print("\nCreating model...")
    model = ESHModel(model_config)
    model.print_model_size()
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    print("\nLoading datasets...")
    train_dataset = load_tinystories(
        tokenizer,
        max_length=MAX_SEQ_LEN,
        split="train",
        max_samples=MAX_TRAIN_SAMPLES,
    )
    
    eval_dataset = load_tinystories(
        tokenizer,
        max_length=MAX_SEQ_LEN,
        split="validation",
        max_samples=MAX_EVAL_SAMPLES,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval: {len(eval_dataset)} examples")
    
    # Training config - NeurIPS schedule
    training_config = TrainingConfig(
        # Learning rate schedule
        learning_rate=3e-4,
        min_lr=1e-5,
        
        # Batch settings
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        
        # Duration
        max_steps=MAX_STEPS,
        warmup_steps=5000,       # 5% warmup
        burn_in_steps=2000,      # Router stabilization
        
        # Memory optimization
        use_amp=True,
        amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        use_8bit_optimizer=True,  # Save ~3GB VRAM
        
        # Logging
        log_interval=50,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        
        # Output
        output_dir="./esh_neurips_checkpoints",
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {training_config.max_steps:,}")
    print(f"  Warmup steps: {training_config.warmup_steps:,}")
    print(f"  Burn-in steps: {training_config.burn_in_steps:,}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Tokens/step: {BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN:,}")
    print(f"  Total tokens: {MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN / 1e9:.2f}B")
    print(f"  8-bit optimizer: {training_config.use_8bit_optimizer}")
    print(f"  AMP dtype: {training_config.amp_dtype}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        eval_dataloader=eval_dataloader,
    )
    
    # Resume if specified
    if RESUME_FROM:
        print(f"\nResuming from: {RESUME_FROM}")
        trainer.load_checkpoint(RESUME_FROM)
    
    # Memory check
    print("\nInitial memory state:")
    print_memory_stats()
    
    # Estimate training time
    estimated_tok_per_sec = 400  # Conservative for 4096 seq
    total_tokens = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN
    hours = total_tokens / estimated_tok_per_sec / 3600
    
    print(f"\nEstimated training time @ {estimated_tok_per_sec} tok/s:")
    print(f"  {hours:.1f} hours ({hours/24:.1f} days)")
    
    # Train!
    print("\n" + "=" * 70)
    print("Starting NeurIPS training run")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print_memory_stats()


if __name__ == "__main__":
    main()
