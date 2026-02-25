"""
ESH-Unified Model v2: Hard Routing with Compute Penalty
========================================================

ARCHITECTURAL PIVOT (Feb 25, 2026):
- Soft blending → Hard Gumbel-Softmax routing
- Variance/Balance/Z-loss → Single compute cost penalty
- ACT pondering → Disabled (SSM state collision)

The compute penalty creates an economic incentive:
  L_total = L_lm + λ * mean(attn_masks_across_all_layers)
  
The model pays a "tax" for routing tokens to Attention. It will only
choose Attention when the CE loss reduction exceeds the tax. This is
the mathematical engine that drives genuine specialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from esh.layers import RMSNorm
from .layers import UnifiedBlock


@dataclass
class UnifiedConfig:
    """
    Configuration for the Unified ESH Model v2.

    The key flags are enable_width_routing and enable_depth_routing,
    which control the ablation modes:
      - baseline:    width=False  → random 50/50 hard routing
      - width_only:  width=True   → learned hard routing
    
    Depth routing is disabled (ACT + SSM state collision).
    """
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    d_state: int = 16
    d_conv: int = 4
    ssm_expand: int = 2
    n_experts: int = 4
    expert_dim: Optional[int] = None
    max_seq_len: int = 1024

    # Training
    dropout: float = 0.0
    layer_scale_init: float = 1e-5
    use_flash: bool = True
    use_checkpoint: bool = True
    init_std: float = 0.02

    # Ablation flags
    enable_width_routing: bool = True
    enable_depth_routing: bool = False  # Disabled (ACT+SSM collision)
    max_ponder_steps: int = 1           # Always 1
    use_moe: bool = True

    # Compute penalty (the economic incentive)
    # λ = 0.01 means Attention must improve CE loss by >0.01 to be chosen
    compute_penalty_weight: float = 0.01
    
    # Router temperature for Gumbel-Softmax
    router_temperature: float = 1.0

    # Legacy loss weights (kept for compatibility, not used in v2)
    ponder_cost_weight: float = 0.0
    moe_aux_weight: float = 0.01

    # Ablation mode name (auto-computed)
    mode: str = ""

    def __post_init__(self):
        if self.expert_dim is None:
            self.expert_dim = self.d_model * 4
        # Set mode name
        if self.enable_width_routing:
            self.mode = "width_only"
        else:
            self.mode = "baseline"


class UnifiedModel(nn.Module):
    """
    Unified ESH Language Model v2 with Hard Routing.

    Key differences from v1:
    - Router outputs hard masks via Gumbel-Softmax (no soft blending)
    - Compute penalty replaces variance/balance/z-loss
    - No ACT pondering (disabled)
    - No burn-in needed
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Unified blocks
        self.blocks = nn.ModuleList([
            UnifiedBlock(
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
                enable_width_routing=config.enable_width_routing,
                enable_depth_routing=False,
                max_ponder_steps=1,
                use_moe=config.use_moe,
                router_temperature=config.router_temperature,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm and LM head
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)

        # Initialize
        self.apply(self._init_weights)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def set_global_step(self, step: int):
        """Set training step on all blocks (legacy compatibility)."""
        for block in self.blocks:
            block._global_step = step

    def get_compute_penalty(self) -> torch.Tensor:
        """
        Compute the attention cost penalty across all layers.
        
        Returns the mean attention usage fraction [0, 1].
        0.0 = all tokens routed to SSM (cheapest)
        1.0 = all tokens routed to Attention (most expensive)
        """
        attn_masks = []
        for block in self.blocks:
            if block.current_attn_mask is not None:
                attn_masks.append(block.current_attn_mask)
        
        if not attn_masks:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Mean attention usage across all layers and all tokens
        return torch.cat(attn_masks, dim=0).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_routing_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Returns dict with:
          - logits
          - loss (LM loss only, if labels provided)
          - aux_loss (MoE aux loss)
          - compute_penalty (attention usage fraction)
          - routing_stats (if requested)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Track losses and stats
        total_aux_loss = torch.tensor(0.0, device=device)
        all_stats = []

        # Forward through blocks
        for block in self.blocks:
            x, aux_loss, _, stats = block(x, return_routing_stats)
            total_aux_loss = total_aux_loss + aux_loss
            if return_routing_stats:
                all_stats.append(stats)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # LM Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Compute penalty (attention usage tax)
        compute_penalty = self.get_compute_penalty()

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "compute_penalty": compute_penalty,
            "routing_stats": all_stats if return_routing_stats else None,
        }

    def count_parameters(self) -> int:
        return self.n_params

    def print_model_info(self):
        """Print model configuration and size."""
        total = self.n_params
        emb = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        print(f"╔{'═' * 50}╗")
        print(f"║ ESH-Unified v2 ({self.config.mode.upper()})".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ Routing: HARD (Gumbel-Softmax)".ljust(51) + "║")
        print(f"║ Width Routing: {self.config.enable_width_routing}".ljust(51) + "║")
        print(f"║ Depth Routing: DISABLED (ACT collision)".ljust(51) + "║")
        print(f"║ Compute Penalty λ: {self.config.compute_penalty_weight}".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ d_model: {self.config.d_model}".ljust(51) + "║")
        print(f"║ Layers: {self.config.n_layers}".ljust(51) + "║")
        print(f"║ Heads: {self.config.n_heads}".ljust(51) + "║")
        print(f"║ Experts: {self.config.n_experts}".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ Total Params: {total / 1e6:.2f}M".ljust(51) + "║")
        print(f"║ Non-Embedding: {(total - emb) / 1e6:.2f}M".ljust(51) + "║")
        print(f"╚{'═' * 50}╝")
