"""
B4: Trimmed Mean Baseline
"""
import torch
from typing import List


def trimmed_mean(gradients: List[torch.Tensor], beta: float = 0.1) -> torch.Tensor:
    """
    Coordinate-wise trimmed mean.
    
    Args:
        gradients: List of client gradient tensors
        beta: Fraction of extreme values to trim from top and bottom (0.0 to 0.5)
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N, d = stacked.shape

    k = int(N * beta)
    if k == 0 or 2 * k >= N:
        return stacked.mean(dim=0)

    sorted_stacked, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_stacked[k : N - k, :]
    return trimmed.mean(dim=0)
