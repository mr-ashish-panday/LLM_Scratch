"""
Mixed Complexity Data Loader for ESH Phase 2
=============================================
Blends multiple datasets for adaptive routing training:
- TinyStories (70%) - Simple syntax, SSM-friendly
- WikiText-103 (20%) - Complex language, needs attention
- GSM8K (10%) - Math reasoning, requires attention

Usage:
    from data_loader import create_mixed_dataloader
    train_loader = create_mixed_dataloader(tokenizer, batch_size=2, max_length=2048)
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Iterator, Dict, Any
import random


class MixedComplexityDataset(IterableDataset):
    """
    Streaming dataset that mixes multiple complexity levels.
    
    Ratios:
    - TinyStories: 70% (simple stories)
    - WikiText-103: 20% (complex text)
    - GSM8K: 10% (math reasoning)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        tiny_ratio: float = 0.70,
        wiki_ratio: float = 0.20,
        gsm_ratio: float = 0.10,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tiny_ratio = tiny_ratio
        self.wiki_ratio = wiki_ratio
        self.gsm_ratio = gsm_ratio
        self.seed = seed
        self.cache_dir = cache_dir
        
        # Validate ratios
        assert abs(tiny_ratio + wiki_ratio + gsm_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        # If cache_dir is set, download to disk (works offline after first run)
        # If cache_dir is None, use streaming (requires constant internet)
        use_streaming = cache_dir is None
        mode_str = "streaming" if use_streaming else f"cached → {cache_dir}"
        
        print(f"Loading TinyStories ({mode_str})...")
        self.tiny_stories = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=use_streaming,
            cache_dir=cache_dir,
        )
        
        print(f"Loading WikiText-103 ({mode_str})...")
        self.wikitext = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=use_streaming,
            cache_dir=cache_dir,
        )
        
        print(f"Loading GSM8K ({mode_str})...")
        self.gsm8k = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            streaming=use_streaming,
            cache_dir=cache_dir,
        )
        
        print("Mixed dataset ready!")
    
    def _format_gsm8k(self, example: Dict[str, Any]) -> str:
        """Format GSM8K as question + answer."""
        question = example.get("question", "")
        answer = example.get("answer", "")
        return f"Question: {question}\nAnswer: {answer}"
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and pad/truncate to max_length."""
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mixed samples based on ratios."""
        rng = random.Random(self.seed)
        
        # Create iterators
        tiny_iter = iter(self.tiny_stories)
        wiki_iter = iter(self.wikitext)
        gsm_iter = iter(self.gsm8k)
        
        while True:
            # Sample dataset based on ratios
            choice = rng.random()
            
            try:
                if choice < self.tiny_ratio:
                    # TinyStories
                    example = next(tiny_iter)
                    text = example.get("text", "")
                    source = "tiny"
                    
                elif choice < self.tiny_ratio + self.wiki_ratio:
                    # WikiText-103
                    example = next(wiki_iter)
                    text = example.get("text", "")
                    source = "wiki"
                    
                else:
                    # GSM8K
                    example = next(gsm_iter)
                    text = self._format_gsm8k(example)
                    source = "gsm"
                
                # Skip empty texts
                if not text or len(text.strip()) < 50:
                    continue
                
                # Tokenize
                tokens = self._tokenize(text)
                tokens["source"] = source  # For debugging/analysis
                
                yield tokens
                
            except StopIteration:
                # Restart exhausted iterator
                if choice < self.tiny_ratio:
                    tiny_iter = iter(self.tiny_stories)
                elif choice < self.tiny_ratio + self.wiki_ratio:
                    wiki_iter = iter(self.wikitext)
                else:
                    gsm_iter = iter(self.gsm8k)


class MixedEvalDataset(torch.utils.data.Dataset):
    """
    In-memory eval dataset with samples from each source.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        samples_per_source: int = 100,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        use_streaming = cache_dir is None
        print(f"Loading {samples_per_source} eval samples per source...")
        
        # TinyStories
        tiny = load_dataset("roneneldan/TinyStories", split="validation",
                           streaming=use_streaming, cache_dir=cache_dir)
        for i, ex in enumerate(tiny):
            if i >= samples_per_source:
                break
            text = ex.get("text", "")
            if text and len(text.strip()) >= 50:
                self.examples.append({"text": text, "source": "tiny"})
        
        # WikiText-103
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                           streaming=use_streaming, cache_dir=cache_dir)
        for i, ex in enumerate(wiki):
            if i >= samples_per_source:
                break
            text = ex.get("text", "")
            if text and len(text.strip()) >= 50:
                self.examples.append({"text": text, "source": "wiki"})
        
        # GSM8K (use test split for eval)
        gsm = load_dataset("openai/gsm8k", "main", split="test",
                          streaming=use_streaming, cache_dir=cache_dir)
        for i, ex in enumerate(gsm):
            if i >= samples_per_source:
                break
            text = f"Question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}"
            self.examples.append({"text": text, "source": "gsm"})
        
        print(f"Loaded {len(self.examples)} eval examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "source": ex["source"],
        }


def create_mixed_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 2,
    max_length: int = 2048,
    tiny_ratio: float = 0.70,
    wiki_ratio: float = 0.20,
    gsm_ratio: float = 0.10,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Create dataloader for mixed complexity training.
    
    If cache_dir is set, datasets are downloaded to disk and loaded locally
    (works offline after first download). Otherwise uses streaming.
    """
    
    dataset = MixedComplexityDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        tiny_ratio=tiny_ratio,
        wiki_ratio=wiki_ratio,
        gsm_ratio=gsm_ratio,
        cache_dir=cache_dir,
    )
    
    def collate_fn(batch):
        """Collate function that handles the source field."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        sources = [b.get("source", "unknown") for b in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sources": sources,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )


def create_mixed_eval_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    samples_per_source: int = 100,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Create eval dataloader with samples from each source."""
    
    dataset = MixedEvalDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        samples_per_source=samples_per_source,
        cache_dir=cache_dir,
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        sources = [b["source"] for b in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sources": sources,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # Test the data loader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n=== Testing Mixed Data Loader ===")
    loader = create_mixed_dataloader(tokenizer, batch_size=2, max_length=512)
    
    source_counts = {"tiny": 0, "wiki": 0, "gsm": 0}
    
    for i, batch in enumerate(loader):
        if i >= 100:
            break
        for source in batch["sources"]:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nSource distribution over 100 batches:")
    total = sum(source_counts.values())
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({100*count/total:.1f}%)")
