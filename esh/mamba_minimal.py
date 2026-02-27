"""
Pure-PyTorch Mamba (Selective State Space Model)
================================================

This is a drop-in replacement for `mamba_ssm.Mamba` that requires
NO custom CUDA kernels. It implements the exact same math:

    h_t = A_bar · h_{t-1} + B_bar · x_t
    y_t = C · h_t + D · x_t

Where A_bar, B_bar, C are INPUT-DEPENDENT (selective), making this
fundamentally different from a convolution:
- Conv: fixed kernel, local receptive field
- Mamba: input-dependent transitions, infinite receptive field

The sequential scan is O(L·D·N) vs the CUDA kernel's O(L·D·N) but
without hardware-level parallelism. ~3-5x slower, but mathematically
identical — which is what matters for proving routing works.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           (Gu & Dao, 2023) — Algorithm 2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaMinimal(nn.Module):
    """
    Pure-PyTorch implementation of Mamba (Selective SSM).
    
    Same interface as mamba_ssm.Mamba:
        input:  [B, L, D]
        output: [B, L, D]
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        if dt_rank == "auto":
            self.dt_rank = max(math.ceil(d_model / 16), 1)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection: x → (z, x_proj) where z is the gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal depthwise conv (like in the original Mamba)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters (input-dependent / selective)
        # x → (Δ, B, C) — these are computed per-token, making it SELECTIVE
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        
        # Δ (timestep) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # A parameter (not input-dependent, but discretized with input-dependent Δ)
        # Initialize A as negative log-uniform (standard Mamba init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        # dt_proj bias init (from original Mamba code)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            output: [B, L, D]
        """
        B, L, D = x.shape
        
        # 1. Input projection → split into x_proj and gate z
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]
        
        # 2. Causal depthwise conv
        x_proj = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_proj = self.conv1d(x_proj)[:, :, :L]  # Causal: trim padding
        x_proj = x_proj.transpose(1, 2)  # [B, L, d_inner]
        x_proj = F.silu(x_proj)
        
        # 3. SSM parameters (SELECTIVE: computed per-token)
        ssm_params = self.x_proj(x_proj)  # [B, L, dt_rank + 2*d_state]
        dt, B_param, C_param = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 4. Discretize
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)    # Ensure positive timestep
        
        A = -torch.exp(self.A_log)  # [d_inner, d_state] — negative for stability
        
        # 5. Selective scan (the core SSM recurrence)
        y = self._selective_scan(x_proj, dt, A, B_param, C_param)
        
        # 6. Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_proj
        
        # 7. Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan(self, x, dt, A, B, C):
        """
        Memory-efficient sequential selective scan.
        
        Computes per-timestep to avoid materializing [B, L, d_inner, d_state]
        tensors (which caused OOM at seq_len=512 with 8 layers).
        
        Args:
            x:  [B, L, d_inner] — input
            dt: [B, L, d_inner] — timestep (per-token, per-channel)
            A:  [d_inner, d_state] — state transition (negative)
            B:  [B, L, d_state] — input matrix (per-token)
            C:  [B, L, d_state] — output matrix (per-token)
        
        Returns:
            y: [B, L, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Sequential scan — compute per-timestep to save memory
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(seq_len):
            # Discretize at this timestep only (no big pre-computed tensors)
            dt_t = dt[:, t].unsqueeze(-1)          # [B, d_inner, 1]
            dA_t = torch.exp(dt_t * A)              # [B, d_inner, d_state]
            dBx_t = dt_t * B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)  # [B, d_inner, d_state]
            
            h = dA_t * h + dBx_t                    # [B, d_inner, d_state]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1) # [B, d_inner]
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        return y
