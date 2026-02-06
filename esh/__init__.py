# ESH: Entropy-Steered Hybridization
# A dynamic hybrid architecture for efficient LLM training

from .model import ESHModel, ESHConfig, esh_small, esh_medium, esh_large, esh_scaled
from .layers import ESHBlock, SoftEntropyRouter, GatedAttention, Top1MoE, ScalableMoE

__version__ = "0.1.0"
__all__ = [
    "ESHModel",
    "ESHConfig",
    "ESHBlock",
    "SoftEntropyRouter",
    "GatedAttention",
    "Top1MoE",
    "ScalableMoE",
    "esh_small",
    "esh_medium",
    "esh_large",
    "esh_scaled",
]

