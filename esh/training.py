"""
ESH Training Utilities
======================
Training loop, optimizer configuration, and logging utilities
optimized for 12GB VRAM training.
"""

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_decay_style: str = "cosine"  # "cosine" or "linear"
    
    # Batch size (effective = batch_size * gradient_accumulation_steps)
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Memory optimization
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    
    # Paths
    output_dir: str = "./checkpoints"
    
    # Burn-in phase (force balanced routing initially)
    burn_in_steps: int = 1000  # Steps before router takes full control
    initial_attention_ratio: float = 0.5  # Start with 50/50


class Trainer:
    """
    Training orchestrator for ESH models.
    
    Features:
    - Mixed precision training (bfloat16/float16)
    - Gradient accumulation
    - Cosine learning rate schedule with warmup
    - Routing statistics logging
    - Burn-in phase for router stabilization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.callbacks = callbacks or {}
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._configure_optimizer()
        
        # Setup AMP
        self.scaler = GradScaler() if config.use_amp and config.amp_dtype == "float16" else None
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32
        
        # Training state
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW with weight decay on non-bias/norm parameters."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        nodecay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No decay for biases, norms, and embeddings
            if param.ndim == 1 or "emb" in name or "norm" in name or "bias" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=torch.cuda.is_available(),  # Faster on CUDA
        )
        
        return optimizer
    
    def _get_lr(self) -> float:
        """Compute learning rate with warmup and decay."""
        # Warmup phase
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        
        # Decay phase
        decay_steps = self.config.max_steps - self.config.warmup_steps
        current_decay_step = self.step - self.config.warmup_steps
        
        if self.config.lr_decay_style == "cosine":
            # Cosine decay to min_lr
            decay_ratio = current_decay_step / decay_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
        else:
            # Linear decay
            decay_ratio = 1.0 - current_decay_step / decay_steps
            return self.config.min_lr + decay_ratio * (self.config.learning_rate - self.config.min_lr)
    
    def _update_lr(self):
        """Update learning rate for current step."""
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_burn_in_target_ratio(self) -> float:
        """Get target attention ratio during burn-in phase."""
        if self.step >= self.config.burn_in_steps:
            # After burn-in, use model's configured target
            return self.model.config.target_attention_ratio
        
        # Linear interpolation from initial (0.5) to target
        progress = self.step / self.config.burn_in_steps
        return self.config.initial_attention_ratio + progress * (
            self.model.config.target_attention_ratio - self.config.initial_attention_ratio
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        
        # Update target attention ratio during burn-in
        target_ratio = self._get_burn_in_target_ratio()
        self.model.config.target_attention_ratio = target_ratio
        
        # Forward pass with AMP
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
            outputs = self.model(input_ids, labels=labels, return_routing_stats=True)
            loss = outputs["loss"]
            aux_loss = outputs["aux_loss"]
            total_loss = loss + aux_loss
            
            # Scale for gradient accumulation
            total_loss = total_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Collect routing stats
        routing_stats = {}
        if outputs["routing_stats"]:
            # Average across layers
            alpha_means = [s["alpha_mean"] for s in outputs["routing_stats"]]
            attn_ratios = [s["attention_ratio"] for s in outputs["routing_stats"]]
            routing_stats = {
                "alpha_mean": sum(alpha_means) / len(alpha_means),
                "attention_ratio": sum(attn_ratios) / len(attn_ratios),
            }
        
        return {
            "loss": loss.item(),
            "aux_loss": aux_loss.item(),
            "ppl": math.exp(min(loss.item(), 20)),  # Clamp to avoid overflow
            **routing_stats,
        }
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm.item()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataloader."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_routing_stats = []
        
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
                outputs = self.model(input_ids, labels=labels, return_routing_stats=True)
            
            batch_tokens = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * batch_tokens
            total_tokens += batch_tokens
            
            if outputs["routing_stats"]:
                all_routing_stats.append(outputs["routing_stats"])
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        
        result = {
            "eval_loss": avg_loss,
            "eval_ppl": math.exp(min(avg_loss, 20)),
        }
        
        # Average routing stats
        if all_routing_stats:
            avg_attn_ratio = sum(
                sum(s["attention_ratio"] for s in batch_stats) / len(batch_stats)
                for batch_stats in all_routing_stats
            ) / len(all_routing_stats)
            result["eval_attention_ratio"] = avg_attn_ratio
        
        return result
    
    def save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint."""
        name = name or f"step_{self.step}"
        path = Path(self.config.output_dir) / f"{name}.pt"
        
        checkpoint = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from {path} at step {self.step}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        data_iter = iter(self.train_dataloader)
        metrics_buffer = []
        start_time = time.time()
        
        while self.step < self.config.max_steps:
            # Get batch (with cycling)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Accumulate gradients
            metrics = self.train_step(batch)
            metrics_buffer.append(metrics)
            self.tokens_seen += batch["input_ids"].numel()
            
            # Optimizer step
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                lr = self._update_lr()
                grad_norm = self.optimizer_step()
                metrics["grad_norm"] = grad_norm
                metrics["lr"] = lr
            
            self.step += 1
            
            # Logging
            if self.step % self.config.log_interval == 0:
                avg_metrics = self._average_metrics(metrics_buffer)
                elapsed = time.time() - start_time
                tokens_per_sec = self.tokens_seen / elapsed
                
                print(
                    f"Step {self.step:>6} | "
                    f"Loss {avg_metrics['loss']:.4f} | "
                    f"PPL {avg_metrics['ppl']:.2f} | "
                    f"Aux {avg_metrics['aux_loss']:.4f} | "
                    f"Attn% {avg_metrics.get('attention_ratio', 0)*100:.1f} | "
                    f"LR {metrics.get('lr', 0):.2e} | "
                    f"Tok/s {tokens_per_sec:.0f}"
                )
                
                metrics_buffer = []
                
                # Callback
                if "on_log" in self.callbacks:
                    self.callbacks["on_log"](self.step, avg_metrics)
            
            # Evaluation
            if self.step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"  Eval | Loss {eval_metrics['eval_loss']:.4f} | PPL {eval_metrics['eval_ppl']:.2f}")
                    
                    if eval_metrics['eval_loss'] < self.best_val_loss:
                        self.best_val_loss = eval_metrics['eval_loss']
                        self.save_checkpoint("best")
                    
                    if "on_eval" in self.callbacks:
                        self.callbacks["on_eval"](self.step, eval_metrics)
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint()
        
        print("Training complete!")
        self.save_checkpoint("final")
    
    def _average_metrics(self, metrics_buffer: list) -> Dict[str, float]:
        """Average metrics over buffer."""
        if not metrics_buffer:
            return {}
        
        avg = {}
        for key in metrics_buffer[0].keys():
            values = [m[key] for m in metrics_buffer if key in m]
            if values:
                avg[key] = sum(values) / len(values)
        
        return avg


def get_memory_stats() -> Dict[str, float]:
    """Get GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def print_memory_stats():
    """Print GPU memory usage."""
    stats = get_memory_stats()
    if stats:
        print(
            f"GPU Memory: "
            f"Allocated {stats['allocated_gb']:.2f}GB | "
            f"Reserved {stats['reserved_gb']:.2f}GB | "
            f"Peak {stats['max_allocated_gb']:.2f}GB"
        )
