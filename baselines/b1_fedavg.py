"""
B1: FedAvg Baseline
"""
import torch
from typing import List


def fedavg(gradients: List[torch.Tensor]) -> torch.Tensor:
    """Arithmetic mean of client gradients."""
    stacked = torch.stack([g.flatten().float() for g in gradients])
    return stacked.mean(dim=0)
