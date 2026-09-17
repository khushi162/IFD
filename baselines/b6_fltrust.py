"""
B6: FLTrust [Cao et al., NDSS 2021]
"""
import torch
from typing import List, Optional


def fltrust(gradients: List[torch.Tensor], server_gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    FLTrust Defense using trusted server root dataset gradient.
    
    Args:
        gradients: List of client gradient tensors
        server_gradient: Trusted server gradient vector (if None, mean honest fallback)
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N, d = stacked.shape

    if server_gradient is None:
        g0 = stacked.mean(dim=0)
    else:
        g0 = server_gradient.flatten().float()

    g0_norm = torch.norm(g0) + 1e-8
    unit_g0 = g0 / g0_norm

    weights = torch.zeros(N)
    normalized_grads = torch.zeros_like(stacked)

    for i in range(N):
        gi = stacked[i]
        gi_norm = torch.norm(gi) + 1e-8
        unit_gi = gi / gi_norm

        # ReLU of cosine similarity with server gradient
        cos_sim = torch.sum(unit_gi * unit_g0)
        ts = torch.relu(cos_sim)
        weights[i] = ts

        # Scale client gradient to server gradient magnitude
        normalized_grads[i] = unit_gi * g0_norm

    total_weight = weights.sum()
    if total_weight < 1e-8:
        return g0

    weighted_sum = (normalized_grads * weights.unsqueeze(1)).sum(dim=0)
    return weighted_sum / total_weight
