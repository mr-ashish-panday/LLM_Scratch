"""
ESH-Unified Model
=================
Full model with configurable ablation modes for the Width × Depth experiment.
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
    Configuration for the Unified ESH Model.

    The key flags are enable_width_routing and enable_depth_routing,
    which control the 4 ablation modes:
      - baseline:    width=False, depth=False
      - width_only:  width=True,  depth=False
      - depth_only:  width=False, depth=True
      - unified:     width=True,  depth=True
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
    enable_depth_routing: bool = True
    max_ponder_steps: int = 3
    use_moe: bool = True  # False = plain SwiGLU FFN (cleaner 2D ablation)

    # Loss weights
    ponder_cost_weight: float = 0.03
    moe_aux_weight: float = 0.01

    # Ablation mode name (auto-computed)
    mode: str = ""

    def __post_init__(self):
        if self.expert_dim is None:
            self.expert_dim = self.d_model * 4
        # Set mode name
        if self.enable_width_routing and self.enable_depth_routing:
            self.mode = "unified"
        elif self.enable_width_routing:
            self.mode = "width_only"
        elif self.enable_depth_routing:
            self.mode = "depth_only"
        else:
            self.mode = "baseline"


class UnifiedModel(nn.Module):
    """
    Unified ESH Language Model.

    Supports 4 ablation modes for the Width × Depth experiment.
    Uses production-grade ESH components (Mamba-2, FlashAttn-2, Top-1 MoE).
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
                enable_depth_routing=config.enable_depth_routing,
                max_ponder_steps=config.max_ponder_steps,
                use_moe=config.use_moe,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm and LM head
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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
        """Set training step on all blocks (for burn-in logic)."""
        for block in self.blocks:
            block._global_step = step

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
          - loss (if labels provided)
          - aux_loss (MoE + router)
          - ponder_cost (mean ponder penalty)
          - avg_ponder_steps
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
        total_ponder_cost = 0.0
        all_stats = []

        # Forward through blocks
        for block in self.blocks:
            x, aux_loss, ponder_cost, stats = block(x, return_routing_stats)
            total_aux_loss = total_aux_loss + aux_loss
            total_ponder_cost += ponder_cost if isinstance(ponder_cost, float) else ponder_cost.item() if hasattr(ponder_cost, 'item') else float(ponder_cost)
            if return_routing_stats:
                all_stats.append(stats)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Aggregate stats
        avg_ponder = 1.0
        if all_stats:
            ponder_vals = [s.get("avg_ponder_steps", 1.0) for s in all_stats]
            avg_ponder = sum(ponder_vals) / len(ponder_vals)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "ponder_cost": total_ponder_cost / max(1, self.config.n_layers),
            "avg_ponder_steps": avg_ponder,
            "routing_stats": all_stats if return_routing_stats else None,
        }

    def count_parameters(self) -> int:
        return self.n_params

    def print_model_info(self):
        """Print model configuration and size."""
        total = self.n_params
        emb = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        print(f"╔{'═' * 50}╗")
        print(f"║ ESH-Unified Model ({self.config.mode.upper()})".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ Mode: {self.config.mode}".ljust(51) + "║")
        print(f"║ Width Routing: {self.config.enable_width_routing}".ljust(51) + "║")
        print(f"║ Depth Routing: {self.config.enable_depth_routing}".ljust(51) + "║")
        print(f"║ Max Ponder Steps: {self.config.max_ponder_steps}".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ d_model: {self.config.d_model}".ljust(51) + "║")
        print(f"║ Layers: {self.config.n_layers}".ljust(51) + "║")
        print(f"║ Heads: {self.config.n_heads}".ljust(51) + "║")
        print(f"║ Experts: {self.config.n_experts}".ljust(51) + "║")
        print(f"╠{'═' * 50}╣")
        print(f"║ Total Params: {total / 1e6:.2f}M".ljust(51) + "║")
        print(f"║ Non-Embedding: {(total - emb) / 1e6:.2f}M".ljust(51) + "║")
        print(f"║ Est. Memory (fp16): {total * 2 / 1e9:.2f} GB".ljust(51) + "║")
        print(f"╚{'═' * 50}╝")
