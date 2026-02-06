"""
ESH Training Example
====================
Example script for training an ESH model on TinyStories/BabyLM.

Usage:
    python train.py

Requirements:
    pip install torch transformers datasets mamba-ssm
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from esh import ESHModel, ESHConfig
from esh.model import esh_small, esh_medium
from esh.training import Trainer, TrainingConfig, print_memory_stats


# =============================================================================
# Dataset (TinyStories example)
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset with tokenization."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
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


def load_tinystories(tokenizer, max_length: int = 512, max_samples: int = 10000):
    """Load TinyStories dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        
        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # Take a subset
        texts = [ex["text"] for ex in dataset.select(range(min(max_samples, len(dataset))))]
        
        return TextDataset(texts, tokenizer, max_length)
    
    except Exception as e:
        print(f"Could not load TinyStories: {e}")
        print("Using dummy data instead...")
        
        # Fallback to dummy data
        dummy_texts = [
            "Once upon a time, there was a little girl named Lucy. She loved to play in the garden.",
            "The big brown dog ran through the park. It was a sunny day and everyone was happy.",
            "Tom found a shiny red ball. He played with it all afternoon with his friends.",
        ] * 100
        
        return TextDataset(dummy_texts, tokenizer, max_length)


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use smaller config for debugging (switch to esh_medium for full training)
    model_config = esh_small()
    model_config.max_seq_len = 512  # Shorter for faster iteration
    
    print(f"Estimated parameters: {model_config.estimate_params() / 1e6:.2f}M")
    
    # Create model
    model = ESHModel(model_config)
    model.print_model_size()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_dataset = load_tinystories(
        tokenizer,
        max_length=model_config.max_seq_len,
        max_samples=10000,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set >0 if not debugging
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Training config (adjusted for 12GB VRAM)
    training_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch = 32
        max_steps=5000,
        warmup_steps=200,
        burn_in_steps=500,  # Balanced routing for first 500 steps
        use_amp=True,
        amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        log_interval=10,
        eval_interval=500,
        save_interval=1000,
        output_dir="./esh_checkpoints",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        eval_dataloader=None,  # Add eval dataloader for proper evaluation
    )
    
    # Print initial memory usage
    print_memory_stats()
    
    # Train!
    trainer.train()
    
    print("Training complete!")
    print_memory_stats()


if __name__ == "__main__":
    main()
