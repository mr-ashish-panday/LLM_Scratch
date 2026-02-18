"""
ESH-Unified Layers
==================
UnifiedBlock: The core building block supporting 4 ablation modes.

Reuses production-grade components from the ESH package:
- GatedAttention (FlashAttention-2)
- SSMLayer (Mamba-2)
- Top1MoE / ScalableMoE
- RMSNorm, LayerScale

Adds adaptive pondering loop from ESH-Loop with ACT-style halting.
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
from .router import UnifiedEntropyRouter


class UnifiedBlock(nn.Module):
    """
    Unified ESH Block with configurable Width and Depth routing.

    Modes (controlled by config flags):
      - baseline:    alpha fixed at 0.5, ponder_steps=1
      - width_only:  alpha learned, ponder_steps=1
      - depth_only:  alpha fixed at 0.5, ponder_steps=1-K
      - unified:     alpha learned, ponder_steps=1-K

    Architecture per ponder step:
        x → RMSNorm → Router(α, p_halt)
          → [α·Attention(x) + (1-α)·SSM(x)] → LayerScale → +Residual
          → RMSNorm → MoE(x) → LayerScale → +Residual
        → ACT accumulate with halt weight
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
        enable_depth_routing: bool = True,
        max_ponder_steps: int = 3,
        halt_threshold: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.enable_width_routing = enable_width_routing
        self.enable_depth_routing = enable_depth_routing
        self.max_ponder_steps = max_ponder_steps
        self.halt_threshold = halt_threshold
        self._global_step = 99999  # Set by model before forward
        self.burn_in_steps = 1500  # Force α=0.5 during burn-in

        # Normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Unified Router (produces both alpha and halt_prob)
        self.router = UnifiedEntropyRouter(dim=d_model)

        # Dual processing paths
        self.attention = GatedAttention(d_model, n_heads, dropout, use_flash)
        self.ssm = SSMLayer(d_model, d_state, d_conv, ssm_expand)

        # LayerScale (one per path + one for MoE)
        self.ls_attn = LayerScale(d_model, layer_scale_init)
        self.ls_ssm = LayerScale(d_model, layer_scale_init)
        self.ls_moe = LayerScale(d_model, layer_scale_init)

        # FFN: MoE or plain SwiGLU (toggle for ablation cleanliness)
        self.use_moe = use_moe
        if not use_moe:
            # Single SwiGLU FFN — no routing overhead, pure 2D test
            self.ffn = SwiGLUExpert(d_model, expert_dim or d_model * 4)
        elif n_experts > 4:
            self.moe = ScalableMoE(d_model, n_experts, expert_dim)
        else:
            self.moe = Top1MoE(d_model, n_experts, expert_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def _single_step(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One processing step: blend paths + MoE FFN."""
        residual = x
        normed = self.norm1(x)

        # Width blend: α·Attn + (1-α)·SSM
        attn_out = self.ls_attn(self.attention(normed))
        ssm_out = self.ls_ssm(self.ssm(normed))
        blended = alpha * attn_out + (1 - alpha) * ssm_out

        # First residual
        x = residual + self.dropout_layer(blended)

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
            aux_loss: scalar (MoE + router aux losses)
            ponder_cost: scalar (mean ponder steps as a differentiable penalty)
            stats: dict with routing/pondering statistics
        """
        B, L, D = x.shape
        device = x.device
        total_aux_loss = torch.tensor(0.0, device=device)

        # =====================================================================
        # NO DEPTH ROUTING: Single step (baseline or width_only)
        # =====================================================================
        if not self.enable_depth_routing:
            # Get alpha from router (or fix to 0.5)
            router_out = self.router(x, ponder_step=0, max_steps=1,
                                     training=self.training)
            if self.enable_width_routing:
                # BURN-IN: Force α=0.5 for first N steps
                if self._global_step < self.burn_in_steps:
                    alpha = torch.full((B, L, 1), 0.5, device=device, dtype=x.dtype)
                else:
                    alpha = router_out.alpha  # [B, L, 1] learned
            else:
                alpha = torch.full((B, L, 1), 0.5, device=device, dtype=x.dtype)

            total_aux_loss = total_aux_loss + router_out.aux_loss

            if self.use_checkpoint and self.training:
                output, moe_loss = checkpoint(
                    self._single_step, x, alpha, use_reentrant=False
                )
            else:
                output, moe_loss = self._single_step(x, alpha)

            total_aux_loss = total_aux_loss + moe_loss

            stats = {}
            if return_routing_stats:
                with torch.no_grad():
                    stats = {
                        "alpha_mean": alpha.mean().item(),
                        "alpha_std": alpha.std().item(),
                        "attention_ratio": (alpha > 0.5).float().mean().item(),
                        "avg_ponder_steps": 1.0,
                        "halt_prob_mean": 1.0,
                    }

            return output, total_aux_loss, 0.0, stats

        # =====================================================================
        # DEPTH ROUTING: Adaptive pondering loop (depth_only or unified)
        # =====================================================================
        accumulated = torch.zeros_like(x)
        # per-token halted probability (how much prob mass has been "spent")
        halted_prob = torch.zeros(B, L, 1, device=device, dtype=x.dtype)
        current = x

        all_alphas = []
        # Track remainders for DIFFERENTIABLE ponder cost
        remainders = []

        for step in range(self.max_ponder_steps):
            # Route
            router_out = self.router(
                current, ponder_step=step, max_steps=self.max_ponder_steps,
                training=self.training
            )

            if self.enable_width_routing:
                # BURN-IN: Force α=0.5 for first N steps
                if self._global_step < self.burn_in_steps:
                    alpha = torch.full((B, L, 1), 0.5, device=device, dtype=x.dtype)
                else:
                    alpha = router_out.alpha
            else:
                alpha = torch.full((B, L, 1), 0.5, device=device, dtype=x.dtype)

            halt_prob = router_out.halt_prob  # [B, L, 1]
            total_aux_loss = total_aux_loss + router_out.aux_loss
            all_alphas.append(alpha.detach())

            # Process step
            if self.use_checkpoint and self.training:
                step_output, moe_loss = checkpoint(
                    self._single_step, current, alpha, use_reentrant=False
                )
            else:
                step_output, moe_loss = self._single_step(current, alpha)

            total_aux_loss = total_aux_loss + moe_loss

            # Remaining probability for this token (DIFFERENTIABLE through halt_prob)
            remainder = 1.0 - halted_prob
            remainders.append(remainder)

            # ACT accumulation
            if step < self.max_ponder_steps - 1:
                # Normal step: halt_prob determines how much to commit
                weight = halt_prob * remainder
            else:
                # Last step: dump all remaining probability
                weight = remainder

            accumulated = accumulated + weight * step_output
            halted_prob = halted_prob + weight

            # Update current for next step
            current = step_output

            # Early exit if all tokens have halted (>threshold committed)
            if (halted_prob > self.halt_threshold).all():
                break

        # =====================================================================
        # Ponder cost: SUM OF REMAINDERS (differentiable!)
        #
        # remainder_t = 1 - halted_prob_t, which depends on halt_prob (sigmoid)
        # so gradient FLOWS through this cost.
        #
        # If halt_prob_0 is high → remainder_1 is low → cost is low (good)
        # If halt_prob is always low → all remainders are high → cost is high
        #
        # Normalized by max_steps so cost is in [0.33, 1.0] for 3 steps:
        #   1 step:  remainder_0=1.0, ~0.33
        #   3 steps: all remainders~1.0, ~1.0
        #
        # OLD BUG: n_updates was a discrete counter with ZERO gradient.
        # =====================================================================
        ponder_cost = torch.stack(remainders).mean(dim=0).mean() / self.max_ponder_steps * len(remainders)

        # Alpha balance losses (two components):
        #
        # 1. L_mean = (α.mean() - 0.5)²
        #    Penalizes the BATCH mean deviating from 0.5.
        #    Allows individual tokens to be 0.1 or 0.9 as long as
        #    the macro-level usage of SSM vs Attention is balanced.
        #
        # 2. L_var = clamp(0.15 - α.var(), min=0)
        #    ENCOURAGES high variance — forces the router to make
        #    decisive, polarized choices (not mushy 0.5 for everything).
        #
        # NOTE: The old formula ((α-0.5)²).mean() was wrong because it
        # penalized each token individually, forcing ALL tokens to 0.5
        # and destroying the routing entirely.
        alpha_balance_loss = torch.tensor(0.0, device=device)
        if self.enable_width_routing and all_alphas and self._global_step >= self.burn_in_steps:
            # Only apply alpha balance after burn-in (during burn-in, α is forced to 0.5)
            stacked_alpha = torch.cat(all_alphas, dim=-1)
            alpha_mean_loss = (stacked_alpha.mean() - 0.5) ** 2
            alpha_var_loss = torch.clamp(0.15 - stacked_alpha.var(), min=0.0)
            alpha_balance_loss = alpha_mean_loss + alpha_var_loss
            total_aux_loss = total_aux_loss + 1.0 * alpha_mean_loss + 1.0 * alpha_var_loss

        stats = {}
        if return_routing_stats:
            with torch.no_grad():
                stacked_alpha = torch.cat(all_alphas, dim=-1)  # [B, L, steps]
                avg_alpha = stacked_alpha.mean()
                # Per-token ponder steps: sum of remainders across steps → [B, L, 1]
                per_token_ponder = torch.stack(remainders).sum(dim=0).squeeze(-1)  # [B, L]
                # Per-token alpha: average across ponder steps → [B, L]
                per_token_alpha = stacked_alpha.mean(dim=-1)  # [B, L]
                stats = {
                    "alpha_mean": avg_alpha.item(),
                    "alpha_std": stacked_alpha[:, :, 0].std().item(),
                    "attention_ratio": (stacked_alpha[:, :, 0] > 0.5).float().mean().item(),
                    "avg_ponder_steps": per_token_ponder.mean().item(),
                    "halt_prob_mean": halted_prob.mean().item(),
                    "alpha_balance_loss": alpha_balance_loss.item() if torch.is_tensor(alpha_balance_loss) else 0.0,
                    # Per-token data for entropy analysis (detached, cpu)
                    "per_token_ponder": per_token_ponder.cpu(),  # [B, L]
                    "per_token_alpha": per_token_alpha.cpu(),  # [B, L]
                }

        return accumulated, total_aux_loss, ponder_cost, stats
