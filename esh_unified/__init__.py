"""
ESH-Unified v2: Hard Routing via Gumbel-Softmax
================================================
Unified model with hard routing (SSM vs Attention) via Gumbel-Softmax.
Depth routing disabled (ACT + SSM state collision).
"""

from .router import HardEntropyRouter
from .layers import UnifiedBlock
from .model import UnifiedConfig, UnifiedModel

__version__ = "2.0.0"
__all__ = [
    "UnifiedConfig",
    "UnifiedModel",
    "UnifiedBlock",
    "HardEntropyRouter",
]
