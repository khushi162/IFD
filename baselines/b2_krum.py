"""
B2: Krum & Multi-Krum [Blanchard et al., NeurIPS 2017]
"""
import torch
from typing import List, Optional


def krum(gradients: List[torch.Tensor], f: int = 1, multi_krum: bool = False, m: Optional[int] = None) -> torch.Tensor:
    """
    Krum / Multi-Krum Defense.
    
    Args:
        gradients: List of client gradient tensors
        f: Number of assumed Byzantine clients
        multi_krum: If True, average top m selected clients
        m: Number of clients to select in Multi-Krum
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N = stacked.shape[0]
    
    if N <= 2 * f + 2:
        # Fallback if N too small for Krum bound N >= 2f + 3
        return stacked.mean(dim=0)

    # Compute pairwise squared L2 distances
    dists = torch.cdist(stacked, stacked, p=2) ** 2

    # For each client i, sum distances to N - f - 2 closest peers
    k = N - f - 2
    scores = torch.zeros(N)
    for i in range(N):
        sorted_dists, _ = torch.sort(dists[i])
        scores[i] = sorted_dists[1 : k + 1].sum()  # exclude self distance at 0

    if not multi_krum:
        best_idx = int(torch.argmin(scores).item())
        return stacked[best_idx]
    else:
        num_select = m if m is not None else N - f - 2
        num_select = max(1, min(num_select, N))
        _, top_indices = torch.topk(scores, k=num_select, largest=False)
        return stacked[top_indices].mean(dim=0)
