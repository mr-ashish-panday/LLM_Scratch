"""
Hard Width Routing Model: Hybrid SSM-Attention with Compute Penalty
====================================================================

The compute penalty creates an economic incentive:
  L_total = L_lm + λ * mean(attn_masks_across_all_layers)
  
The model pays a "tax" for routing tokens to Attention. It will only
choose Attention when the CE loss reduction exceeds the tax. This is
the mathematical engine that drives genuine specialization.

Routing is hard at the token level, but execution stays dense so the
ablation remains exact on large-memory GPUs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from esh.layers import RMSNorm
from .layers import UnifiedBlock, PureAttentionBlock, PureSSMBlock


@dataclass
class UnifiedConfig:
    """
    Configuration for Hard Width Routing Model.

    Supported ablation modes:
      - baseline:         Random 50/50 hard routing (control)
      - width_only:       Learned hard routing (the method)
      - pure_transformer: Attention-only (no SSM)
      - pure_ssm:         SSM-only (no Attention)
      - interleaved_1_1:  Alternating SSM/Attn layers (4+4)
      - interleaved_1_5:  1 Attn per 5 SSM layers
      - random_topk:      Budget-matched random routing (~7%)
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
    enable_depth_routing: bool = False
    max_ponder_steps: int = 1
    use_moe: bool = True

    # Compute penalty
    compute_penalty_weight: float = 0.01
    router_temperature: float = 1.0
    ponder_cost_weight: float = 0.0
    moe_aux_weight: float = 0.01

    # Ablation mode name
    mode: str = ""

    # Attention budget for random_topk mode (fraction 0-1)
    attn_budget: float = 0.07

    def __post_init__(self):
        if self.expert_dim is None:
            self.expert_dim = self.d_model * 4
        if not self.mode:
            if self.enable_width_routing:
                self.mode = "width_only"
            else:
                self.mode = "baseline"


class UnifiedModel(nn.Module):
    """
    Hard Width Routing Language Model.

    Supports multiple ablation modes — all share the same embedding layer
    and LM head for fair comparison. Only the block types differ.
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config

        # Embeddings (shared across all modes)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Build blocks based on mode
        self.blocks = nn.ModuleList(self._build_blocks(config))

        # Final norm and LM head (shared across all modes)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _select_interleaved_attn_layers(n_layers: int, ssm_per_attn: int):
        """Place a small number of attention layers evenly through the stack."""
        if n_layers <= 0:
            return set()

        n_attn = max(1, round(n_layers / (ssm_per_attn + 1)))
        layers = set()
        for idx in range(n_attn):
            position = round((idx + 1) * (n_layers + 1) / (n_attn + 1)) - 1
            position = max(0, min(n_layers - 1, position))
            layers.add(position)
        return layers

    def _build_blocks(self, config):
        """Construct block list based on ablation mode."""
        blocks = []
        interleaved_1_5_attn_layers = self._select_interleaved_attn_layers(
            config.n_layers, ssm_per_attn=5
        )
        for i in range(config.n_layers):
            if config.mode == "pure_transformer":
                blocks.append(PureAttentionBlock(
                    d_model=config.d_model, n_heads=config.n_heads,
                    n_experts=config.n_experts, expert_dim=config.expert_dim,
                    dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                    use_flash=config.use_flash, use_checkpoint=config.use_checkpoint,
                    use_moe=config.use_moe,
                ))
            elif config.mode == "pure_ssm":
                blocks.append(PureSSMBlock(
                    d_model=config.d_model, d_state=config.d_state,
                    d_conv=config.d_conv, ssm_expand=config.ssm_expand,
                    n_experts=config.n_experts, expert_dim=config.expert_dim,
                    dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                    use_checkpoint=config.use_checkpoint, use_moe=config.use_moe,
                ))
            elif config.mode == "interleaved_1_1":
                # Alternating: even layers = SSM, odd layers = Attention
                # Following hybrid analysis: don't put Transformer first
                if i % 2 == 0:
                    blocks.append(PureSSMBlock(
                        d_model=config.d_model, d_state=config.d_state,
                        d_conv=config.d_conv, ssm_expand=config.ssm_expand,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, use_checkpoint=config.use_checkpoint,
                        use_moe=config.use_moe,
                    ))
                else:
                    blocks.append(PureAttentionBlock(
                        d_model=config.d_model, n_heads=config.n_heads,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, use_flash=config.use_flash,
                        use_checkpoint=config.use_checkpoint, use_moe=config.use_moe,
                    ))
            elif config.mode == "interleaved_1_5":
                # Approximate 1 attention layer per 5 SSM layers, spaced evenly.
                if i in interleaved_1_5_attn_layers:
                    blocks.append(PureAttentionBlock(
                        d_model=config.d_model, n_heads=config.n_heads,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                        use_flash=config.use_flash,
                        use_checkpoint=config.use_checkpoint, use_moe=config.use_moe,
                    ))
                else:
                    blocks.append(PureSSMBlock(
                        d_model=config.d_model, d_state=config.d_state,
                        d_conv=config.d_conv, ssm_expand=config.ssm_expand,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                        use_checkpoint=config.use_checkpoint,
                        use_moe=config.use_moe,
                    ))
            elif config.mode == "interleaved_1_7":
                # Nemotron/Zamba-style: 1 attention per 7 SSM layers (~12.5%).
                # For 8 layers: layer 3 gets attention, rest SSM.
                interleaved_1_7_attn_layers = self._select_interleaved_attn_layers(
                    config.n_layers, ssm_per_attn=7
                )
                if i in interleaved_1_7_attn_layers:
                    blocks.append(PureAttentionBlock(
                        d_model=config.d_model, n_heads=config.n_heads,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                        use_flash=config.use_flash,
                        use_checkpoint=config.use_checkpoint, use_moe=config.use_moe,
                    ))
                else:
                    blocks.append(PureSSMBlock(
                        d_model=config.d_model, d_state=config.d_state,
                        d_conv=config.d_conv, ssm_expand=config.ssm_expand,
                        n_experts=config.n_experts, expert_dim=config.expert_dim,
                        dropout=config.dropout, layer_scale_init=config.layer_scale_init,
                        use_checkpoint=config.use_checkpoint,
                        use_moe=config.use_moe,
                    ))
            else:
                # baseline, width_only, random_topk, entropy_topk — all use UnifiedBlock
                blocks.append(UnifiedBlock(
                    d_model=config.d_model, n_heads=config.n_heads,
                    d_state=config.d_state, d_conv=config.d_conv,
                    ssm_expand=config.ssm_expand, n_experts=config.n_experts,
                    expert_dim=config.expert_dim, dropout=config.dropout,
                    layer_scale_init=config.layer_scale_init,
                    use_flash=config.use_flash, use_checkpoint=config.use_checkpoint,
                    enable_width_routing=config.enable_width_routing,
                    use_moe=config.use_moe,
                    router_temperature=config.router_temperature,
                    mode=config.mode,
                    attn_budget=config.attn_budget,
                ))
        return blocks

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
        mean_attn_ratio = compute_penalty.detach().item()
        avg_ponder_steps = 1.0
        if all_stats:
            mean_attn_ratio = sum(
                s.get("attention_ratio", s.get("attn_ratio", 0.0))
                for s in all_stats
            ) / len(all_stats)
            avg_ponder_steps = sum(
                s.get("avg_ponder_steps", 1.0) for s in all_stats
            ) / len(all_stats)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "compute_penalty": compute_penalty,
            "attention_ratio": mean_attn_ratio,
            "alpha_mean": mean_attn_ratio,
            "avg_ponder_steps": avg_ponder_steps,
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
        print(f"║ Routing: HARD masks (dense execution)".ljust(51) + "║")
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
