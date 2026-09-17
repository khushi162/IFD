"""
B5: Bulyan [El Mhamdi et al., ICML 2018]
"""
import torch
from typing import List


def bulyan(gradients: List[torch.Tensor], f: int = 1) -> torch.Tensor:
    """
    Bulyan Defense: Multi-Krum selection + Trimmed Mean.
    
    Args:
        gradients: List of client gradient tensors
        f: Number of assumed Byzantine clients
    """
    stacked = torch.stack([g.flatten().float() for g in gradients])
    N = stacked.shape[0]

    if N <= 4 * f + 2:
        # Fallback if N too small for Bulyan bound N >= 4f + 3
        return stacked.mean(dim=0)

    # Step 1: Multi-Krum selection to select theta = N - 2f clients
    theta = N - 2 * f
    selected_indices = []
    remaining_indices = list(range(N))

    for _ in range(theta):
        rem_stacked = stacked[remaining_indices]
        M = rem_stacked.shape[0]
        if M <= 2:
            break
        dists = torch.cdist(rem_stacked, rem_stacked, p=2) ** 2
        k = max(1, M - f - 2)
        scores = torch.zeros(M)
        for i in range(M):
            sorted_dists, _ = torch.sort(dists[i])
            scores[i] = sorted_dists[1 : k + 1].sum()
        best_local_idx = int(torch.argmin(scores).item())
        best_global_idx = remaining_indices[best_local_idx]
        selected_indices.append(best_global_idx)
        remaining_indices.remove(best_global_idx)

    selected_grads = stacked[selected_indices]
    S = selected_grads.shape[0]

    # Step 2: Trimmed mean on selected set (trim 2f extreme values)
    trim_count = f
    if 2 * trim_count >= S:
        return selected_grads.mean(dim=0)

    sorted_selected, _ = torch.sort(selected_grads, dim=0)
    trimmed = sorted_selected[trim_count : S - trim_count, :]
    return trimmed.mean(dim=0)
