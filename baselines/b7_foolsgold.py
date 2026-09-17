"""
B7: FoolsGold [Fung et al., 2020]
"""
import torch
from typing import List


def foolsgold(gradients: List[torch.Tensor]) -> torch.Tensor:
    """
    FoolsGold Sybil defense based on pairwise gradient cosine similarity.
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N, d = stacked.shape

    if N <= 1:
        return stacked.mean(dim=0)

    # Normalize gradients
    norms = torch.norm(stacked, dim=1, keepdim=True) + 1e-8
    normed = stacked / norms

    # Pairwise cosine similarity matrix S (NxN)
    S = torch.mm(normed, normed.T)

    # For each client i, find maximum similarity with any other client j != i
    max_sim = torch.zeros(N)
    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        max_sim[i] = torch.max(S[i, mask])

    # Diversity score a_i = 1 - max_j!=i S_ij
    alpha = 1.0 - torch.clamp(max_sim, 0.0, 1.0)
    
    # Softmax / rescale weighting
    weights = torch.softmax(alpha * 5.0, dim=0)
    
    return (stacked * weights.unsqueeze(1)).sum(dim=0)
