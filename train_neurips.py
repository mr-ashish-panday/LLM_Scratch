"""
ESH NeurIPS Training Script
===========================
100,000-step training schedule optimized for Virtual Scaling:
- 16-expert MoE (~1.5B capacity, ~400M active)
- 2048 sequence length
- 8-bit optimizer states
- Streaming TinyStories dataset (RAM efficient)

Usage:
    nohup python -u train_neurips.py > training_neurips.log 2>&1 &
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from esh import ESHModel, ESHConfig
from esh.model import esh_scaled
from esh.training import Trainer, TrainingConfig, print_memory_stats


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that tokenizes on-the-fly.
    Doesn't load everything into RAM.
    """
    
    def __init__(self, split: str, tokenizer, max_length: int = 2048, max_samples: int = None):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        
        # Get dataset size for length estimation
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        self._length = min(len(ds), max_samples) if max_samples else len(ds)
        del ds  # Free memory
    
    def __iter__(self):
        from datasets import load_dataset
        
        dataset = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
        
        for i, example in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            
            tokens = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield {"input_ids": tokens["input_ids"].squeeze(0)}
    
    def __len__(self):
        return self._length


class SimpleTextDataset(torch.utils.data.Dataset):
    """Small in-memory dataset for evaluation."""
    
    def __init__(self, split: str, tokenizer, max_length: int, max_samples: int = 1000):
        from datasets import load_dataset
        
        print(f"Loading {max_samples} {split} examples for evaluation...")
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for ex in dataset:
            tokens = tokenizer(
                ex["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append(tokens["input_ids"].squeeze(0))
        print(f"  Loaded {len(self.examples)} eval examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def main():
    print("=" * 70)
    print("ESH NeurIPS Training - Virtual Scaling Edition")
    print("=" * 70)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Training duration (reduced for faster validation)
    MAX_STEPS = 20_000       # 20k steps - ~2-3 days
    
    # Sequence length (2048 for 12GB fit)
    MAX_SEQ_LEN = 2048
    
    # Batch settings - balanced for speed + memory
    BATCH_SIZE = 2           # Balanced (4 OOMs, 1 is too slow)
    GRAD_ACCUM = 16          # Keep effective batch = 32
    
    # Data limits (None = full dataset ~2.1M)
    MAX_TRAIN_SAMPLES = None
    MAX_EVAL_SAMPLES = 500   # Small eval set
    
    # Checkpointing
    SAVE_INTERVAL = 5000     # Every 5k steps
    EVAL_INTERVAL = 2000     # Every 2k steps
    
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
    
    # Model config
    model_config = esh_scaled()
    model_config.max_seq_len = MAX_SEQ_LEN
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_experts: {model_config.n_experts}")
    print(f"  expert_dim: {model_config.expert_dim}")
    print(f"  max_seq_len: {model_config.max_seq_len}")
    
    # Parameter estimates
    print(f"\nParameter Estimates:")
    base_params = model_config.estimate_params()
    extra_expert_params = (16 - 4) * model_config.n_layers * 3 * model_config.d_model * model_config.expert_dim
    total_params = base_params + extra_expert_params
    
    print(f"  Total params (all experts): ~{total_params / 1e9:.2f}B")
    print(f"  Active params (1 expert): ~{base_params / 1e6:.0f}M")
    
    # Create model
    print("\nCreating model...")
    model = ESHModel(model_config)
    model.print_model_size()
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data - STREAMING for training (RAM efficient!)
    print("\nSetting up streaming dataset (RAM efficient)...")
    train_dataset = StreamingTextDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    
    # Small eval set (in-memory is fine)
    eval_dataset = SimpleTextDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        max_samples=MAX_EVAL_SAMPLES,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Streaming doesn't work with multiple workers
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"\nDataset info:")
    print(f"  Train: ~{len(train_dataset):,} examples (streaming)")
    print(f"  Eval: {len(eval_dataset)} examples (in-memory)")
    
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
    estimated_tok_per_sec = 400
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
