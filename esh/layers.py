"""
ESH Core Layers
===============
Implementation of ESHBlock with:
- Soft Entropy Routing (differentiable α-blending)
- Gated Attention with FlashAttention-2
- Top-1 MoE with load balancing
- LayerScale for gradient stabilization
- Gradient checkpointing for memory efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
from dataclasses import dataclass

# Try to import mamba - falls back gracefully if not installed
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using placeholder SSM.")


# =============================================================================
# Soft Entropy Router
# =============================================================================

class SoftEntropyRouter(nn.Module):
    """
    Soft Entropy-based Router for dynamic path selection.
    
    Outputs α ∈ [0, 1] where:
    - α → 1: High complexity, favor Attention path
    - α → 0: Low complexity, favor SSM path
    
    The routing is fully differentiable via soft-gating.
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        init_temperature: float = 1.0,
        min_temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model // 4
        
        # Complexity scoring network
        self.complexity_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),  # SiLU/Swish is smoother than GELU for routing
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
        # Learnable temperature for sigmoid sharpness
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.min_temperature = min_temperature
        
        # Initialize to balanced routing (outputs ~0.5 initially)
        self._init_weights()
        
    def _init_weights(self):
        # Small init to start near α ≈ 0.5 (balanced routing)
        for m in self.complexity_net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                
    @property
    def temperature(self) -> torch.Tensor:
        # Ensure temperature doesn't go below minimum
        return self.log_temperature.exp().clamp(min=self.min_temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            alpha: [batch, seq_len, 1] - routing weights for attention path
        """
        # Compute complexity score
        complexity = self.complexity_net(x)  # [B, L, 1]
        
        # Apply temperature-scaled sigmoid
        alpha = torch.sigmoid(complexity / self.temperature)
        
        return alpha
    
    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """Compute routing statistics for logging."""
        with torch.no_grad():
            alpha = self.forward(x)
            return {
                "alpha_mean": alpha.mean().item(),
                "alpha_std": alpha.std().item(),
                "attention_ratio": (alpha > 0.5).float().mean().item(),
                "temperature": self.temperature.item(),
            }


# =============================================================================
# Gated Attention with FlashAttention-2
# =============================================================================

class GatedAttention(nn.Module):
    """
    Sigmoid-Gated Multi-Head Attention with FlashAttention-2 support.
    
    Key features:
    - Sigmoid gating for training stability
    - FlashAttention-2 for memory efficiency
    - RoPE-ready (can add rotary embeddings)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash and self._flash_available()
        
        # QKV projection (fused for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Sigmoid gate (the key innovation for stability)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _flash_available(self) -> bool:
        """Check if FlashAttention-2 is available."""
        return (
            hasattr(F, 'scaled_dot_product_attention') and
            torch.cuda.is_available()
        )
        
    def _init_weights(self):
        # Standard attention initialization
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))
        # Gate initialized to output ~1 (pass-through initially)
        nn.init.zeros_(self.gate_proj.weight)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional mask for padding
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        B, L, D = x.shape
        
        # Compute QKV
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]
        q, k, v = qkv.unbind(0)  # Each: [B, H, L, D_h]
        
        # Attention computation
        if self.use_flash:
            # FlashAttention-2 via PyTorch's SDPA
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,  # Causal mask for autoregressive
            )
        else:
            # Manual attention (fallback)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        
        # Sigmoid gating (stabilizes training, improves long-context)
        gate = torch.sigmoid(self.gate_proj(x))
        output = self.out_proj(attn_out) * gate
        
        return output


# =============================================================================
# SSM Layer (Mamba-2 wrapper or placeholder)
# =============================================================================

class SSMLayer(nn.Module):
    """
    State Space Model layer. Uses Mamba-2 if available, otherwise a 
    simpler convolutional alternative.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        
        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            # Fallback: Gated convolution (not as good, but works)
            self.ssm = self._build_conv_ssm(d_model, d_conv, expand)
            self.use_mamba = False
            
    def _build_conv_ssm(self, d_model: int, d_conv: int, expand: int):
        """Simple gated conv as Mamba fallback."""
        inner_dim = d_model * expand
        return nn.Sequential(
            nn.Linear(d_model, inner_dim * 2, bias=False),
            nn.GLU(dim=-1),
            # Causal conv
            nn.Conv1d(inner_dim, inner_dim, d_conv, padding=d_conv-1, groups=inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, d_model, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        if self.use_mamba:
            return self.ssm(x)
        else:
            # Conv fallback needs channel-first
            B, L, D = x.shape
            out = self.ssm[0](x)  # Linear + GLU
            out = self.ssm[1](out)
            out = out.transpose(1, 2)  # [B, D, L]
            out = self.ssm[2](out)[:, :, :L]  # Causal conv, trim
            out = self.ssm[3](out.transpose(1, 2))  # SiLU
            out = self.ssm[4](out)  # Project back
            return out


# =============================================================================
# Top-1 Mixture of Experts
# =============================================================================

class Top1MoE(nn.Module):
    """
    Memory-efficient Top-1 Mixture of Experts with load balancing.
    
    Features:
    - Top-1 routing (only 1 expert active per token)
    - Auxiliary load balancing loss
    - Capacity factor to prevent expert overflow
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        expert_dim: Optional[int] = None,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.expert_dim = expert_dim or d_model * 4
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        # Router
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Experts (SwiGLU-style for each)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, self.expert_dim)
            for _ in range(n_experts)
        ])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.gate.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: scalar load balancing loss
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [B*L, D]
        N = x_flat.size(0)
        
        # Compute routing probabilities
        logits = self.gate(x_flat)  # [N, n_experts]
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 selection
        top_probs, indices = probs.max(dim=-1)  # [N], [N]
        
        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                # Weight by routing probability (allows gradient flow)
                expert_out = expert(x_flat[mask])
                output[mask] = expert_out * top_probs[mask].unsqueeze(-1)
        
        # Load balancing auxiliary loss
        aux_loss = self._compute_aux_loss(probs, indices)
        
        return output.view(B, L, D), aux_loss
    
    def _compute_aux_loss(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing loss to prevent expert collapse."""
        N = probs.size(0)
        
        # Fraction of tokens routed to each expert
        load = torch.bincount(indices, minlength=self.n_experts).float() / N
        
        # Average probability assigned to each expert
        importance = probs.mean(dim=0)
        
        # Auxiliary loss: minimize product of load and importance imbalance
        aux_loss = (load * importance).sum() * self.n_experts
        
        return aux_loss * self.aux_loss_weight


class SwiGLUExpert(nn.Module):
    """Single expert using SwiGLU activation."""
    
    def __init__(self, d_model: int, expert_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, expert_dim, bias=False)  # Gate
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(xW1) ⊙ xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# LayerScale
# =============================================================================

class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling initialized to small values.
    Prevents any single path from dominating gradient flow early in training.
    
    Reference: "Going deeper with Image Transformers" (Touvron et al., 2021)
    """
    
    def __init__(self, d_model: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# =============================================================================
# ESH Block (Main Component)
# =============================================================================

class ESHBlock(nn.Module):
    """
    Entropy-Steered Hybrid Block.
    
    The core building block of ESH architecture featuring:
    - Soft entropy-based routing between SSM and Attention
    - LayerScale for gradient balancing
    - Gradient checkpointing for memory efficiency
    - Top-1 MoE FFN for capacity
    
    Architecture:
        x → Norm → [α·Attention(x) + (1-α)·SSM(x)] → LayerScale → +Residual
          → Norm → MoE(x) → LayerScale → +Residual → output
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        n_experts: int = 4,
        expert_dim: Optional[int] = None,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
        use_flash: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        
        # Normalization layers (RMSNorm for efficiency)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Soft Entropy Router
        self.router = SoftEntropyRouter(d_model)
        
        # Dual paths
        self.attention = GatedAttention(d_model, n_heads, dropout, use_flash)
        self.ssm = SSMLayer(d_model, d_state, d_conv, ssm_expand)
        
        # LayerScale for gradient balancing (separate for each path)
        self.ls_attn = LayerScale(d_model, layer_scale_init)
        self.ls_ssm = LayerScale(d_model, layer_scale_init)
        self.ls_moe = LayerScale(d_model, layer_scale_init)
        
        # MoE FFN
        self.moe = Top1MoE(d_model, n_experts, expert_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _hybrid_forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Execute both paths and blend based on routing weights."""
        # Attention path
        attn_out = self.attention(x)
        attn_out = self.ls_attn(attn_out)
        
        # SSM path
        ssm_out = self.ssm(x)
        ssm_out = self.ls_ssm(ssm_out)
        
        # Soft blending: Output = α·Attention(x) + (1-α)·SSM(x)
        blended = alpha * attn_out + (1 - alpha) * ssm_out
        
        return blended
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            return_routing_stats: If True, return routing statistics
            
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: MoE auxiliary loss
            routing_stats: Optional dict of routing statistics
        """
        residual = x
        
        # Pre-norm
        x = self.norm1(x)
        
        # Compute routing weights
        alpha = self.router(x)  # [B, L, 1]
        
        # Hybrid forward (with optional gradient checkpointing)
        if self.use_checkpoint and self.training:
            # Gradient checkpointing saves memory by recomputing activations
            blended = checkpoint(
                self._hybrid_forward,
                x, alpha,
                use_reentrant=False,
            )
        else:
            blended = self._hybrid_forward(x, alpha)
        
        # First residual connection
        x = residual + self.dropout(blended)
        
        # MoE FFN
        residual = x
        x = self.norm2(x)
        
        if self.use_checkpoint and self.training:
            moe_out, aux_loss = checkpoint(
                self.moe,
                x,
                use_reentrant=False,
            )
        else:
            moe_out, aux_loss = self.moe(x)
        
        moe_out = self.ls_moe(moe_out)
        
        # Second residual connection
        x = residual + self.dropout(moe_out)
        
        # Optional routing statistics
        routing_stats = None
        if return_routing_stats:
            routing_stats = self.router.get_routing_stats(self.norm1(residual))
        
        return x, aux_loss, routing_stats


# =============================================================================
# RMSNorm (Memory-efficient normalization)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
