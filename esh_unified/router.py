"""
ESH-Unified Router v2: Hard Routing via Gumbel-Softmax
======================================================

ARCHITECTURAL PIVOT (Feb 25, 2026):
The original soft-sigmoid router (α·Attn + (1-α)·SSM) was proven to be a
dead coin flip. Root cause: soft blending creates an "ensembling attractor"
at α=0.5 because the LM loss prefers maximum representational bandwidth.

This rewrite uses Gumbel-Softmax with Straight-Through Estimator (STE):
- Forward pass: hard one-hot routing (exactly one path per token)
- Backward pass: continuous gradients through softmax approximation
- Ensembling is IMPOSSIBLE: ssm_mask ∈ {0,1}, attn_mask ∈ {0,1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RouterOutput:
    ssm_mask: torch.Tensor      # [B, L, 1] — 1 means route to SSM
    attn_mask: torch.Tensor     # [B, L, 1] — 1 means route to Attention
    logits: torch.Tensor        # [B, L, 2] — raw logits for debugging


class HardEntropyRouter(nn.Module):
    """
    Hard routing via Gumbel-Softmax with Straight-Through Estimator.

    Outputs mutually exclusive masks: each token goes to EXACTLY one path.
    No ensembling. No soft blending. Forces genuine specialization.

    The model MUST learn to route because:
    1. Only one path is visible per token (no ensemble safety net)
    2. A compute penalty taxes Attention usage
    3. The router must discover that SSM is "cheaper" and sufficient 
       for simple tokens
    """

    def __init__(self, d_model: int, hidden_dim: int = None, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim or max(d_model // 4, 32)

        # 2-class output: [SSM_preference, Attention_preference]
        # NOTE: bias=True (unlike the old router) to break the symmetry
        self.complexity_net = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

        self.temperature = temperature
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: [B, L, D] hidden states

        Returns:
            RouterOutput with ssm_mask [B,L,1], attn_mask [B,L,1], logits [B,L,2]
        """
        logits = self.complexity_net(x)  # [B, L, 2]

        if self.training:
            # Gumbel-Softmax with STE:
            # Forward: hard one-hot [1,0] or [0,1]
            # Backward: continuous gradients through the softmax
            routing_weights = F.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )
        else:
            # Deterministic greedy routing during inference
            indices = torch.argmax(logits, dim=-1)
            routing_weights = F.one_hot(indices, num_classes=2).float()

        ssm_mask = routing_weights[..., 0:1]    # [B, L, 1]
        attn_mask = routing_weights[..., 1:2]   # [B, L, 1]

        return RouterOutput(
            ssm_mask=ssm_mask,
            attn_mask=attn_mask,
            logits=logits,
        )

    def get_routing_stats(self, x: torch.Tensor):
        """Get stats without Gumbel noise (deterministic)."""
        with torch.no_grad():
            logits = self.complexity_net(x)
            probs = F.softmax(logits, dim=-1)
            ssm_prob = probs[..., 0].mean().item()
            attn_prob = probs[..., 1].mean().item()
        return {
            "ssm_prob": ssm_prob,
            "attn_prob": attn_prob,
            "temp": self.temperature,
        }
