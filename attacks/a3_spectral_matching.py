"""
A3: Spectral Matching Attack

Crafts malicious gradient vector to lie strictly inside the top-k peer subspace
of Layer 2 (Spectral LOO Anomaly Detector). By projecting the target direction
onto the peer subspace, reconstruction error is guaranteed to be near zero.
"""

import torch
from typing import List, Optional


class SpectralMatching:
    """
    Spectral Matching Attack.
    
    Projects desired target direction onto top-k principal components of peer subspace
    to evade SVD reconstruction error anomaly detection in Layer 2.
    """

    def __init__(self, gamma: float = 0.95):
        """
        Args:
            gamma: Target cumulative variance threshold for subspace components (default 0.95)
        """
        self.gamma = gamma

    def generate_gradient(
        self,
        peer_gradients: List[torch.Tensor],
        target_direction: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate attack gradient lying inside peer subspace.
        
        Args:
            peer_gradients: List of honest client gradient tensors in current round
            target_direction: 1D torch Tensor representing desired target payload
            top_k: Optional manual top-k component count override
            
        Returns:
            g_adv: 1D torch Tensor gradient vector with zero peer subspace reconstruction error
        """
        stacked_peers = torch.stack([g.flatten().float() for g in peer_gradients])
        N, d = stacked_peers.shape
        target = target_direction.flatten().float()

        if N < 2:
            return target * (torch.norm(stacked_peers.mean(dim=0)) / (torch.norm(target) + 1e-8))

        # Center peer matrix
        peer_mean = stacked_peers.mean(dim=0, keepdim=True)
        centered_peers = stacked_peers - peer_mean

        # SVD on centered peer matrix
        U, S, Vh = torch.linalg.svd(centered_peers, full_matrices=False)

        # Select k components explaining gamma variance
        if top_k is None:
            var_ratio = (S ** 2) / ((S ** 2).sum() + 1e-8)
            cum_var = torch.cumsum(var_ratio, dim=0)
            k = int(torch.searchsorted(cum_var, self.gamma).item()) + 1
            k = min(k, min(N - 1, d))
        else:
            k = min(top_k, min(N - 1, d))

        Vk = Vh[:k, :]  # shape (k, d)

        # Project centered target onto Vk
        centered_target = target - peer_mean.squeeze(0)
        proj_coeff = torch.matmul(centered_target, Vk.T)  # shape (k,)
        centered_proj = torch.matmul(proj_coeff, Vk)      # strictly in span(Vk)

        # Scale centered_proj to typical peer deviation scale
        avg_peer_dev = torch.norm(centered_peers, dim=1).mean()
        proj_norm = torch.norm(centered_proj) + 1e-8
        scaled_centered_proj = centered_proj * (avg_peer_dev / proj_norm)

        # Add peer mean back
        g_adv = scaled_centered_proj + peer_mean.squeeze(0)

        return g_adv
