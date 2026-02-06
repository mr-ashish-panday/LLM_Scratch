"""
ESH Extended Training Script
============================
Multi-day training configuration for serious runs.

Usage:
    nohup python -u train_extended.py > training_extended.log 2>&1 &
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from esh import ESHModel, ESHConfig
from esh.model import esh_medium, esh_large
from esh.training import Trainer, TrainingConfig, print_memory_stats


class TextDataset(Dataset):
    """Simple text dataset with tokenization."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokens = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append(tokens["input_ids"].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def load_tinystories_full(tokenizer, max_length: int = 1024, max_samples: int = None):
    """Load full TinyStories dataset."""
    from datasets import load_dataset
    
    print("Loading TinyStories dataset (full)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = [ex["text"] for ex in dataset]
    print(f"Loaded {len(texts)} examples")
    
    return TextDataset(texts, tokenizer, max_length)


def load_tinystories_validation(tokenizer, max_length: int = 1024, max_samples: int = 2000):
    """Load TinyStories validation split."""
    from datasets import load_dataset
    
    print("Loading TinyStories validation...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = [ex["text"] for ex in dataset]
    return TextDataset(texts, tokenizer, max_length)


def main():
    # =========================================================================
    # Configuration - EDIT THESE FOR YOUR RUN
    # =========================================================================
    
    # Model size: "medium" (350M) or "large" (500M)
    MODEL_SIZE = "medium"
    
    # Training duration
    MAX_STEPS = 100_000      # ~2-3 days on decent GPU
    
    # Data
    MAX_TRAIN_SAMPLES = None  # None = use all (~2.1M examples)
    MAX_SEQ_LEN = 1024        # Longer context
    
    # Batch settings (adjust for your GPU memory)
    BATCH_SIZE = 2            # Per-GPU batch size
    GRAD_ACCUM = 16           # Effective batch = 2 * 16 = 32
    
    # Checkpointing
    SAVE_INTERVAL = 5000      # Save every 5k steps
    EVAL_INTERVAL = 2500      # Eval every 2.5k steps
    
    # Resume from checkpoint (set path or None)
    RESUME_FROM = None  # e.g., "esh_checkpoints/step_5000.pt"
    
    # =========================================================================
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model config
    if MODEL_SIZE == "medium":
        model_config = esh_medium()
    else:
        model_config = esh_large()
    
    model_config.max_seq_len = MAX_SEQ_LEN
    
    print(f"Model: esh_{MODEL_SIZE}")
    print(f"Estimated parameters: {model_config.estimate_params() / 1e6:.2f}M")
    
    # Create model
    model = ESHModel(model_config)
    model.print_model_size()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_dataset = load_tinystories_full(
        tokenizer,
        max_length=MAX_SEQ_LEN,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    
    eval_dataset = load_tinystories_validation(
        tokenizer,
        max_length=MAX_SEQ_LEN,
        max_samples=2000,
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
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Eval dataset: {len(eval_dataset)} examples")
    
    # Training config
    training_config = TrainingConfig(
        learning_rate=3e-4,
        min_lr=1e-5,           # Decay to lower LR for stability
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        warmup_steps=2000,      # Longer warmup for big runs
        burn_in_steps=1000,     # Balance routing first 1k steps
        use_amp=True,
        amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        log_interval=50,        # Log every 50 steps
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        output_dir="./esh_checkpoints_extended",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        eval_dataloader=eval_dataloader,
    )
    
    # Resume if specified
    if RESUME_FROM:
        trainer.load_checkpoint(RESUME_FROM)
    
    # Memory check
    print_memory_stats()
    
    # Estimate training time
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN
    total_tokens = MAX_STEPS * tokens_per_step
    print(f"\nTraining plan:")
    print(f"  Steps: {MAX_STEPS:,}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {total_tokens/1e9:.2f}B")
    print(f"  Est. time @ 600 tok/s: {total_tokens / 600 / 3600:.1f} hours")
    
    # Train!
    print("\n" + "="*60)
    print("Starting extended training run")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\nTraining complete!")
    print_memory_stats()


if __name__ == "__main__":
    main()
