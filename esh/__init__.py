# ESH: Entropy-Steered Hybridization
# A dynamic hybrid architecture for efficient LLM training

from .model import ESHModel, ESHConfig
from .layers import ESHBlock, SoftEntropyRouter, GatedAttention, Top1MoE

__version__ = "0.1.0"
__all__ = [
    "ESHModel",
    "ESHConfig", 
    "ESHBlock",
    "SoftEntropyRouter",
    "GatedAttention",
    "Top1MoE",
]
