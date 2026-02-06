"""
ESH Model
=========
Full ESH (Entropy-Steered Hybridization) model with:
- Token embeddings with learned positional encoding
- Multiple ESHBlock layers
- Language modeling head
- Training utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from .layers import ESHBlock, RMSNorm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ESHConfig:
    """
    Configuration for ESH model.
    
    Default settings target ~400M parameters for 12GB VRAM training.
    """
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 tokenizer
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    
    # SSM configuration
    d_state: int = 16
    d_conv: int = 4
    ssm_expand: int = 2
    
    # MoE configuration
    n_experts: int = 4
    expert_dim: Optional[int] = None  # Defaults to 4 * d_model
    
    # Sequence length
    max_seq_len: int = 2048
    
    # Training settings
    dropout: float = 0.0
    layer_scale_init: float = 1e-5
    use_flash: bool = True
    use_checkpoint: bool = True
    
    # Initialization
    init_std: float = 0.02
    
    # Auxiliary loss weights
    moe_aux_weight: float = 0.01
    router_balance_weight: float = 0.01
    target_attention_ratio: float = 0.25  # Target 25% tokens to attention
    
    def __post_init__(self):
        if self.expert_dim is None:
            self.expert_dim = self.d_model * 4
            
    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model  # Token embeddings
        embed_params += self.max_seq_len * self.d_model  # Position embeddings
        
        # Per block (rough estimate)
        attn_params = 4 * self.d_model * self.d_model  # QKV + Out
        ssm_params = self.d_model * self.d_model * self.ssm_expand * 2
        moe_params = self.n_experts * 3 * self.d_model * self.expert_dim
        router_params = self.d_model * (self.d_model // 4) + (self.d_model // 4)
        block_params = attn_params + ssm_params + moe_params + router_params
        
        # Total
        total = embed_params + self.n_layers * block_params + self.d_model * self.vocab_size
        return total


# =============================================================================
# ESH Model
# =============================================================================

class ESHModel(nn.Module):
    """
    ESH: Entropy-Steered Hybridization Language Model.
    
    A dynamic hybrid architecture that routes tokens between SSM and Attention
    paths based on learned complexity signals.
    """
    
    def __init__(self, config: ESHConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # Learned positional embeddings (simple, works well for moderate lengths)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # ESH Blocks
        self.blocks = nn.ModuleList([
            ESHBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_state=config.d_state,
                d_conv=config.d_conv,
                ssm_expand=config.ssm_expand,
                n_experts=config.n_experts,
                expert_dim=config.expert_dim,
                dropout=config.dropout,
                layer_scale_init=config.layer_scale_init,
                use_flash=config.use_flash,
                use_checkpoint=config.use_checkpoint,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.d_model)
        
        # Language modeling head (tied with embeddings optionally)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_routing_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: [batch, seq_len] token indices
            labels: [batch, seq_len] target token indices for loss
            return_routing_stats: Whether to return per-layer routing stats
            
        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - loss: Optional cross-entropy loss
                - aux_loss: Auxiliary losses (MoE + routing balance)
                - routing_stats: Optional list of per-layer routing statistics
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Token + positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        # Track auxiliary losses and stats
        total_aux_loss = torch.tensor(0.0, device=device)
        all_routing_stats = [] if return_routing_stats else None
        attention_ratios = []
        
        # Forward through blocks
        for block in self.blocks:
            x, aux_loss, routing_stats = block(x, return_routing_stats)
            total_aux_loss = total_aux_loss + aux_loss
            
            if return_routing_stats:
                all_routing_stats.append(routing_stats)
                attention_ratios.append(routing_stats["attention_ratio"])
        
        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Compute main loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Router balance loss (encourage target attention ratio)
        router_balance_loss = torch.tensor(0.0, device=device)
        if attention_ratios:
            mean_attn_ratio = sum(attention_ratios) / len(attention_ratios)
            router_balance_loss = (
                (mean_attn_ratio - self.config.target_attention_ratio) ** 2
                * self.config.router_balance_weight
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss + router_balance_loss,
            "routing_stats": all_routing_stats,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with sampling.
        
        Args:
            input_ids: [batch, seq_len] initial token indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            [batch, seq_len + max_new_tokens] generated token indices
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(idx_cond)
                logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
    
    @classmethod
    def from_config(cls, **kwargs) -> "ESHModel":
        """Create model from keyword arguments."""
        config = ESHConfig(**kwargs)
        return cls(config)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters."""
        n_params = self.n_params
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params
    
    def print_model_size(self):
        """Print model size information."""
        total = self.n_params
        embedding = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        
        print(f"ESH Model Size:")
        print(f"  Total Parameters: {total / 1e6:.2f}M")
        print(f"  Embedding Parameters: {embedding / 1e6:.2f}M")
        print(f"  Non-Embedding Parameters: {(total - embedding) / 1e6:.2f}M")
        print(f"  Estimated Memory (fp16): {total * 2 / 1e9:.2f}GB")
        print(f"  Estimated Training Memory (fp16, batch=1): {total * 2 * 4 / 1e9:.2f}GB")


# =============================================================================
# Preset Configurations
# =============================================================================

def esh_small() -> ESHConfig:
    """~125M parameters - for debugging and fast iteration."""
    return ESHConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_experts=4,
        max_seq_len=1024,
    )

def esh_medium() -> ESHConfig:
    """~350M parameters - main target for 12GB VRAM."""
    return ESHConfig(
        d_model=1024,
        n_layers=20,
        n_heads=16,
        n_experts=4,
        max_seq_len=2048,
    )

def esh_large() -> ESHConfig:
    """~500M parameters - upper bound for 12GB VRAM with aggressive checkpointing."""
    return ESHConfig(
        d_model=1280,
        n_layers=24,
        n_heads=20,
        n_experts=4,
        max_seq_len=2048,
        use_checkpoint=True,
    )


def esh_scaled() -> ESHConfig:
    """
    ~1.5B total parameters (virtual) - 16 experts for maximum capacity.
    
    Active parameters per forward pass: ~500M
    Total parameters (all experts): ~1.5B
    Fits in 12GB VRAM via:
    - 16-expert Top-1 MoE (only 1 active)
    - 8-bit optimizer states
    - Gradient checkpointing
    - bfloat16 training
    """
    return ESHConfig(
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_experts=16,  # 16 experts = 4x capacity
        expert_dim=4096,  # Each expert: 1024 -> 4096 -> 1024
        max_seq_len=4096,
        use_checkpoint=True,
        dropout=0.0,
        layer_scale_init=1e-5,
    )

