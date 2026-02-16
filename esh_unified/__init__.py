"""
ESH-Unified: Entropy-Steered Hybridization with 2D Adaptive Compute
====================================================================
Unified model combining width routing (SSM vs Attention) and
depth routing (adaptive pondering) via a single entropy signal.

Supports 4 ablation modes:
  - baseline:    α=0.5 fixed, ponder=1 fixed
  - width_only:  α=learned, ponder=1 fixed
  - depth_only:  α=0.5 fixed, ponder=1-K learned
  - unified:     α=learned, ponder=1-K learned
"""

from .router import UnifiedEntropyRouter
from .layers import UnifiedBlock
from .model import UnifiedConfig, UnifiedModel

__version__ = "0.1.0"
__all__ = [
    "UnifiedConfig",
    "UnifiedModel",
    "UnifiedBlock",
    "UnifiedEntropyRouter",
]
