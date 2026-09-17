"""
B8: DP-FL (Differential Privacy FL)
"""
import torch
from typing import List


def dp_fl(gradients: List[torch.Tensor], clip_norm: float = 1.0, noise_std: float = 0.1) -> torch.Tensor:
    """
    Differential Privacy FL: Gradient clipping + Gaussian noise addition.
    
    Args:
        gradients: List of client gradient tensors
        clip_norm: L2 gradient clipping threshold C
        noise_std: Noise scale factor sigma (Gaussian noise std = sigma * C / N)
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N, d = stacked.shape

    # Clip gradients
    norms = torch.norm(stacked, dim=1, keepdim=True) + 1e-8
    clip_factors = torch.clamp(clip_norm / norms, max=1.0)
    clipped_grads = stacked * clip_factors

    # Average clipped gradients
    avg_grad = clipped_grads.mean(dim=0)

    # Add Gaussian noise: N(0, (sigma * C / N)^2 I)
    noise = torch.randn(d) * (noise_std * clip_norm / N)
    return avg_grad + noise
