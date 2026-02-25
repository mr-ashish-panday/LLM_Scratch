"""
ESH-Unified Layers v2: Hard Routing Architecture
=================================================

ARCHITECTURAL PIVOT (Feb 25, 2026):
- Soft blending (α·Attn + (1-α)·SSM) REPLACED with hard Gumbel-Softmax routing
- Adaptive pondering (ACT loop) DISABLED — causes SSM state collision
- Each token is routed to EXACTLY ONE path per layer
- No burn-in needed (hard routing doesn't have the 0.5 equilibrium trap)

Reuses production-grade components from the ESH package:
- GatedAttention (FlashAttention-2)
- SSMLayer (Mamba-2 or conv placeholder)
- Top1MoE / ScalableMoE
- RMSNorm, LayerScale
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any

# Import production components from ESH
from esh.layers import (
    GatedAttention,
    SSMLayer,
    Top1MoE,
    ScalableMoE,
    SwiGLUExpert,
    LayerScale,
    RMSNorm,
)
from .router import HardEntropyRouter


class UnifiedBlock(nn.Module):
    """
    Unified ESH Block v2 with Hard Routing.

    Each token is routed to exactly ONE path (SSM or Attention) via
    Gumbel-Softmax. No soft blending, no ensembling attractor.

    Architecture per step:
        x → RMSNorm → HardRouter → (ssm_mask, attn_mask)
          → ssm_mask·SSM(x) + attn_mask·Attention(x) → LayerScale → +Residual
          → RMSNorm → MoE(x) → LayerScale → +Residual
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        n_experts: int = 4,
        expert_dim: Optional[int] = None,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
        use_flash: bool = True,
        use_checkpoint: bool = True,
        use_moe: bool = True,
        # Ablation flags
        enable_width_routing: bool = True,
        enable_depth_routing: bool = False,  # DEFAULT OFF (ACT disabled)
        max_ponder_steps: int = 1,           # DEFAULT 1 (no pondering)
        halt_threshold: float = 0.99,
        # Router temperature
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.enable_width_routing = enable_width_routing
        # Depth routing disabled by default (ACT + SSM state collision)
        self.enable_depth_routing = False  # Hardcoded OFF for now
        self.max_ponder_steps = 1          # Hardcoded 1 for now
        self._global_step = 99999  # No burn-in needed with hard routing

        # Normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Hard Router (Gumbel-Softmax)
        self.router = HardEntropyRouter(
            d_model=d_model,
            temperature=router_temperature,
        )

        # Dual processing paths
        self.attention = GatedAttention(d_model, n_heads, dropout, use_flash)
        self.ssm = SSMLayer(d_model, d_state, d_conv, ssm_expand)

        # LayerScale
        self.ls_attn = LayerScale(d_model, layer_scale_init)
        self.ls_ssm = LayerScale(d_model, layer_scale_init)
        self.ls_moe = LayerScale(d_model, layer_scale_init)

        # FFN: MoE or plain SwiGLU
        self.use_moe = use_moe
        if not use_moe:
            self.ffn = SwiGLUExpert(d_model, expert_dim or d_model * 4)
        elif n_experts > 4:
            self.moe = ScalableMoE(d_model, n_experts, expert_dim)
        else:
            self.moe = Top1MoE(d_model, n_experts, expert_dim)

        self.dropout_layer = nn.Dropout(dropout)

        # Store attention mask for compute penalty (set during forward)
        self.current_attn_mask = None

    def _compute_step(
        self,
        x: torch.Tensor,
        ssm_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One processing step with hard routing."""
        residual = x
        normed = self.norm1(x)

        # Compute both paths
        attn_out = self.ls_attn(self.attention(normed))
        ssm_out = self.ls_ssm(self.ssm(normed))

        # HARD selection: exactly one path per token
        # ssm_mask and attn_mask are mutually exclusive {0, 1}
        # Ensembling is IMPOSSIBLE
        mixed_out = ssm_mask * ssm_out + attn_mask * attn_out

        # First residual
        x = residual + self.dropout_layer(mixed_out)

        # FFN (MoE or SwiGLU)
        residual = x
        normed = self.norm2(x)
        if self.use_moe:
            ffn_out, moe_aux_loss = self.moe(normed)
        else:
            ffn_out = self.ffn(normed)
            moe_aux_loss = torch.tensor(0.0, device=x.device)
        ffn_out = self.ls_moe(ffn_out)
        x = residual + self.dropout_layer(ffn_out)

        return x, moe_aux_loss

    def forward(
        self,
        x: torch.Tensor,
        return_routing_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, Any]]:
        """
        Args:
            x: [B, L, D] input tensor
            return_routing_stats: whether to collect stats

        Returns:
            output: [B, L, D]
            aux_loss: scalar (MoE aux loss only — router has no aux loss)
            ponder_cost: 0.0 (pondering disabled)
            stats: dict with routing statistics
        """
        B, L, D = x.shape
        device = x.device

        if self.enable_width_routing:
            # Hard routing via Gumbel-Softmax
            router_out = self.router(x)
            ssm_mask = router_out.ssm_mask      # [B, L, 1]
            attn_mask = router_out.attn_mask     # [B, L, 1]
        else:
            # Baseline: random 50/50 routing (no learning)
            # Use hard random masks, not soft 0.5 blend
            random_choice = (torch.rand(B, L, 1, device=device) > 0.5).float()
            ssm_mask = random_choice
            attn_mask = 1.0 - random_choice

        # Store attn_mask for compute penalty in the training loop
        self.current_attn_mask = attn_mask

        # Forward through the block
        if self.use_checkpoint and self.training:
            output, moe_loss = checkpoint(
                self._compute_step, x, ssm_mask, attn_mask,
                use_reentrant=False
            )
        else:
            output, moe_loss = self._compute_step(x, ssm_mask, attn_mask)

        # Stats
        stats = {}
        if return_routing_stats:
            with torch.no_grad():
                stats = {
                    "attn_ratio": attn_mask.mean().item(),
                    "ssm_ratio": ssm_mask.mean().item(),
                    "avg_ponder_steps": 1.0,
                }
                if self.enable_width_routing:
                    logits = self.router.complexity_net(x)
                    probs = F.softmax(logits, dim=-1)
                    stats["ssm_logit_mean"] = logits[..., 0].mean().item()
                    stats["attn_logit_mean"] = logits[..., 1].mean().item()
                    stats["ssm_prob_mean"] = probs[..., 0].mean().item()
                    stats["attn_prob_mean"] = probs[..., 1].mean().item()

        return output, moe_loss, 0.0, stats
