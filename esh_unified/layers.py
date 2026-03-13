"""
Hard Width Routing Layers: Token-Level SSM/Attention Assignment
================================================================

Each token is routed to exactly one path per layer via Gumbel-Softmax.
On large-memory GPUs we keep both paths dense and use the hard masks for
selection, so the ablation behavior stays exact and we do not claim sparse
attention speedups.

Reuses production-grade components:
- GatedAttention (FlashAttention-2)
- SSMLayer (Mamba-2 or pure-PyTorch fallback)
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


def _build_routing_stats(attn_ratio: float, ssm_ratio: float) -> Dict[str, float]:
    """Return both current and legacy routing keys for local tooling."""
    return {
        "attn_ratio": attn_ratio,
        "ssm_ratio": ssm_ratio,
        "attention_ratio": attn_ratio,
        "alpha_mean": attn_ratio,
        "avg_ponder_steps": 1.0,
    }


class PureAttentionBlock(nn.Module):
    """Pure Transformer block: Attention + FFN only (no SSM). Upper-bound baseline."""

    def __init__(self, d_model=768, n_heads=12, n_experts=4, expert_dim=None,
                 dropout=0.0, layer_scale_init=1e-5, use_flash=True,
                 use_checkpoint=True, use_moe=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attention = GatedAttention(d_model, n_heads, dropout, use_flash)
        self.ls_attn = LayerScale(d_model, layer_scale_init)
        self.ls_ffn = LayerScale(d_model, layer_scale_init)
        self.use_moe = use_moe
        if not use_moe:
            self.ffn = SwiGLUExpert(d_model, expert_dim or d_model * 4)
        elif n_experts > 4:
            self.moe = ScalableMoE(d_model, n_experts, expert_dim)
        else:
            self.moe = Top1MoE(d_model, n_experts, expert_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.current_attn_mask = None  # Always 1.0 for pure attention

    def forward(self, x, return_routing_stats=False):
        B, L, _ = x.shape
        residual = x
        normed = self.norm1(x)
        attn_out = self.ls_attn(self.attention(normed))
        x = residual + self.dropout_layer(attn_out)
        residual = x
        normed = self.norm2(x)
        if self.use_moe:
            ffn_out, moe_aux = self.moe(normed)
        else:
            ffn_out = self.ffn(normed)
            moe_aux = torch.tensor(0.0, device=x.device)
        x = residual + self.dropout_layer(self.ls_ffn(ffn_out))
        self.current_attn_mask = torch.ones(B, L, 1, device=x.device)
        stats = _build_routing_stats(1.0, 0.0) if return_routing_stats else {}
        return x, moe_aux, 0.0, stats


class PureSSMBlock(nn.Module):
    """Pure SSM block: Mamba + FFN only (no Attention). Lower-bound baseline."""

    def __init__(self, d_model=768, d_state=16, d_conv=4, ssm_expand=2,
                 n_experts=4, expert_dim=None, dropout=0.0,
                 layer_scale_init=1e-5, use_checkpoint=True, use_moe=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ssm = SSMLayer(d_model, d_state, d_conv, ssm_expand)
        self.ls_ssm = LayerScale(d_model, layer_scale_init)
        self.ls_ffn = LayerScale(d_model, layer_scale_init)
        self.use_moe = use_moe
        if not use_moe:
            self.ffn = SwiGLUExpert(d_model, expert_dim or d_model * 4)
        elif n_experts > 4:
            self.moe = ScalableMoE(d_model, n_experts, expert_dim)
        else:
            self.moe = Top1MoE(d_model, n_experts, expert_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.current_attn_mask = None  # Always 0.0 for pure SSM

    def forward(self, x, return_routing_stats=False):
        B, L, _ = x.shape
        residual = x
        normed = self.norm1(x)
        ssm_out = self.ls_ssm(self.ssm(normed))
        x = residual + self.dropout_layer(ssm_out)
        residual = x
        normed = self.norm2(x)
        if self.use_moe:
            ffn_out, moe_aux = self.moe(normed)
        else:
            ffn_out = self.ffn(normed)
            moe_aux = torch.tensor(0.0, device=x.device)
        x = residual + self.dropout_layer(self.ls_ffn(ffn_out))
        self.current_attn_mask = torch.zeros(B, L, 1, device=x.device)
        stats = _build_routing_stats(0.0, 1.0) if return_routing_stats else {}
        return x, moe_aux, 0.0, stats


class UnifiedBlock(nn.Module):
    """
    Hard Width Routing Block with dense dual-path execution.

    Architecture:
        x → RMSNorm → HardRouter → {ssm_mask, attn_mask}
          → SSM(x) + Attention(x)  [both dense]
          → Hard-select one output per token → LayerScale → +Residual
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
        # Mode and budget for random_topk
        mode: str = "width_only",
        attn_budget: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.enable_width_routing = enable_width_routing
        self.enable_depth_routing = False  # Out of scope for this paper
        self.max_ponder_steps = 1
        self._global_step = 99999
        self.mode = mode
        self.attn_budget = attn_budget

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
        """One processing step with dense execution and hard output selection."""
        residual = x
        normed = self.norm1(x)
        # Keep both paths dense on large GPUs so we do not change compute semantics.
        attn_out = self.ls_attn(self.attention(normed))
        ssm_out = self.ls_ssm(self.ssm(normed))
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
        elif self.mode == "random_topk":
            # Budget-matched random routing: match learned router's attn fraction
            attn_mask = torch.zeros(B, L, 1, device=device)
            k = min(L, max(0, int(round(L * self.attn_budget))))
            if k > 0:
                rand_scores = torch.rand(B, L, device=device)
                topk_idx = rand_scores.topk(k, dim=-1).indices
                attn_mask.scatter_(1, topk_idx.unsqueeze(-1), 1.0)
            ssm_mask = 1.0 - attn_mask
        elif self.mode == "entropy_topk":
            # Heuristic routing: route highest-complexity tokens to attention.
            # Uses hidden-state L2 norm as complexity proxy (no learned params).
            # High norm = high activation magnitude = likely complex/uncertain.
            with torch.no_grad():
                token_scores = x.norm(dim=-1)  # [B, L]
            k = min(L, max(0, int(round(L * self.attn_budget))))
            attn_mask = torch.zeros(B, L, 1, device=device)
            if k > 0:
                topk_idx = token_scores.topk(k, dim=-1).indices
                attn_mask.scatter_(1, topk_idx.unsqueeze(-1), 1.0)
            ssm_mask = 1.0 - attn_mask
        else:
            # Baseline: random 50/50 routing (no learning)
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
                stats = _build_routing_stats(
                    attn_mask.mean().item(),
                    ssm_mask.mean().item(),
                )
                if self.enable_width_routing:
                    logits = self.router.complexity_net(x)
                    probs = F.softmax(logits, dim=-1)
                    stats["ssm_logit_mean"] = logits[..., 0].mean().item()
                    stats["attn_logit_mean"] = logits[..., 1].mean().item()
                    stats["ssm_prob_mean"] = probs[..., 0].mean().item()
                    stats["attn_prob_mean"] = probs[..., 1].mean().item()

        return output, moe_loss, 0.0, stats
