import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class RouterOutput:
    alpha: torch.Tensor          # [B, L, 1] Blending ratio (0=SSM, 1=Attn)
    halt_prob: torch.Tensor      # [B, L, 1] Halting probability
    aux_loss: torch.Tensor       # Scalar auxiliary loss (variance + balance)
    entropy: torch.Tensor        # [B, L, 1] Predictive entropy (informational)

class UnifiedEntropyRouter(nn.Module):
    """
    The core of the Unified ESH Architecture.
    
    A single router that produces two signals:
    1. Width Control (alpha): Determines the mix of SSM (efficient) vs Attention (expressive).
    2. Depth Control (halt_prob): Determines whether to stop processing (adaptive depth).
    
    This unifies 'where to route' with 'how much to ponder'.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        init_scale: float = 0.02,
        route_method: str = "sigmoid",  # sigmoid or softmax
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or max(dim // 4, 32)
        
        # 1. Complexity Network (Width Routing)
        # No bias in first layer to keep it centered around 0 initially
        self.complexity_net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1, bias=False)
        )
        
        # 2. Halt Network (Depth Routing)
        # Takes input + ponder_step_encoding (scalar)
        self.halt_net = nn.Sequential(
            nn.Linear(dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Learnable temperature for sharpness control
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # Init temp=1.0

        # Loss weights
        self.variance_loss_weight = 0.05
        self.balance_loss_weight = 0.01
        self.z_loss_weight = 0.001
        
        # Initialize weights
        self._init_weights(init_scale)

    def _init_weights(self, scale):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, 
        x: torch.Tensor, 
        ponder_step: int = 0,
        max_steps: int = 3,
        training: bool = True
    ) -> RouterOutput:
        """
        Args:
            x: Input tensor [B, L, D]
            ponder_step: Current step in the adaptive loop (0-indexed)
            max_steps: Maximum allowed steps (for normalization)
            training: Whether to compute aux losses
        """
        B, L, D = x.shape
        
        # --- 1. Compute Width Routing (Alpha) ---
        logits = self.complexity_net(x)  # [B, L, 1]
        
        # Temperature scaling
        temperature = torch.exp(self.log_temperature).clamp(0.1, 10.0)
        scaled_logits = logits / temperature
        
        # Sigmoid routing (0=SSM, 1=Attn)
        alpha = torch.sigmoid(scaled_logits)
        
        # --- 2. Compute Depth Routing (Halt Probability) ---
        # Encode step as a normalized scalar feature
        step_val = ponder_step / max(1, max_steps)
        step_encoding = torch.full(
            (B, L, 1), step_val,
            device=x.device, dtype=x.dtype
        )
        
        halt_input = torch.cat([x, step_encoding], dim=-1)  # [B, L, D+1]
        halt_prob = self.halt_net(halt_input)            # [B, L, 1]
        
        # --- 3. Compute Auxiliary Losses ---
        total_aux_loss = torch.tensor(0.0, device=x.device)
        entropy = torch.zeros_like(alpha)
        
        if training:
            # A) Variance Loss: Encourage decisive routing (not 0.5 everywhere)
            # We want variance of alpha to be around 0.15 (standard deviation ~0.38)
            # This pushes alpha towards 0 or 1, but not hard binary
            alpha_variance = alpha.var()
            target_variance = 0.15
            var_loss = (alpha_variance - target_variance).pow(2)
            
            # B) Balance Loss: Prevent collapse to all-SSM or all-Attention
            # Target mean alpha = 0.25 (25% attention, efficient)
            mean_alpha = alpha.mean()
            target_mean = 0.25
            bal_loss = (mean_alpha - target_mean).abs()
            
            # C) Z-Loss: Prevent logits from exploding
            z_loss = logits.pow(2).mean()
            
            # Combine
            total_aux_loss = (
                self.variance_loss_weight * var_loss +
                self.balance_loss_weight * bal_loss +
                self.z_loss_weight * z_loss
            )
            
            # Compute predictive entropy for logging (info metric only)
            p = alpha
            entropy = -p * torch.log(p + 1e-6) - (1 - p) * torch.log(1 - p + 1e-6)

        return RouterOutput(
            alpha=alpha,
            halt_prob=halt_prob,
            aux_loss=total_aux_loss,
            entropy=entropy
        )

    def get_routing_stats(self, x: torch.Tensor):
        """Helper to get stats without full forward pass overhead."""
        with torch.no_grad():
            out = self.forward(x, training=False)
        return {
            "mean_alpha": out.alpha.mean().item(),
            "std_alpha": out.alpha.std().item(),
            "mean_halt": out.halt_prob.mean().item(),
            "temp": torch.exp(self.log_temperature).item()
        }
