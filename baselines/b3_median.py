"""
B3: Coordinate Median Baseline
"""
import torch
from typing import List


def coordinate_median(gradients: List[torch.Tensor]) -> torch.Tensor:
    """Coordinate-wise median across all client gradients."""
    stacked = torch.stack([g.flatten().float() for g in gradients])
    median_vals, _ = torch.median(stacked, dim=0)
    return median_vals
