"""
A1: Oracle White-Box PGD Attack

Adversary knows full defense state (ref, peer subspace, historical trajectories)
and uses projected gradient descent (PGD) to optimize a malicious gradient vector
that maximizes alignment with target direction while evading cascade detection layers.
"""

import torch
from typing import List, Optional


class OracleWhiteBoxPGD:
    """
    Oracle White-Box PGD Attack.
    
    Optimizes adversarial gradient vector g_adv to maximize alignment with target_direction
    subject to passing Layer 1 (Norm/Cosine), Layer 2 (Spectral), and Layer 3 (Temporal).
    """

    def __init__(
        self,
        eps: float = 0.5,
        step_size: float = 0.05,
        num_steps: int = 30,
        penalty_weight: float = 2.0,
    ):
        """
        Args:
            eps: Maximum L2 perturbation radius from mean honest gradient
            step_size: PGD optimization step size
            num_steps: Number of PGD iterations
            penalty_weight: Weight of defense rejection penalty in loss
        """
        self.eps = eps
        self.step_size = step_size
        self.num_steps = num_steps
        self.penalty_weight = penalty_weight

    def attack(
        self,
        honest_gradients: List[torch.Tensor],
        target_direction: torch.Tensor,
        layer1=None,
        layer2=None,
        layer3=None,
        client_id: str = "adv_0",
    ) -> torch.Tensor:
        """
        Craft adversarial gradient g_adv.
        
        Args:
            honest_gradients: List of 1D torch Tensors representing honest client gradients
            target_direction: 1D torch Tensor representing desired attack direction
            layer1: Instance of Layer1NormCosine (optional)
            layer2: Instance of Layer2Spectral (optional)
            layer3: Instance of Layer3Temporal (optional)
            client_id: Identifier for adversary (for Layer 3 state)
            
        Returns:
            g_adv: Crafted 1D adversarial gradient tensor
        """
        stacked_honest = torch.stack([g.flatten() for g in honest_gradients])
        mean_honest = stacked_honest.mean(dim=0)
        target_normed = target_direction.flatten() / (torch.norm(target_direction.flatten()) + 1e-8)

        # Initialize g_adv near target_direction scaled to mean honest norm
        target_scale = torch.norm(mean_honest)
        g_adv = (target_normed * target_scale).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([g_adv], lr=self.step_size)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Loss: Maximize alignment with target direction
            cos_target = torch.sum(g_adv * target_normed) / (torch.norm(g_adv) + 1e-8)
            loss_target = -cos_target  # Minimize negative cosine

            loss_penalty = torch.tensor(0.0)

            # Differentiable soft penalties if defense layers provided
            if layer1 is not None:
                # Norm penalty
                g_all = torch.cat([stacked_honest, g_adv.unsqueeze(0)], dim=0)
                norms = torch.norm(g_all, dim=1)
                mu_norm = norms[:-1].mean()
                sigma_norm = norms[:-1].std() + 1e-8
                z_norm = torch.abs(torch.norm(g_adv) - mu_norm) / sigma_norm
                s_norm = 1.0 - torch.sigmoid(z_norm - 3.0)
                loss_penalty = loss_penalty + torch.relu(0.5 - s_norm)

            if layer2 is not None and stacked_honest.shape[0] >= 3:
                # Spectral projection penalty
                # Peer matrix excluding g_adv is stacked_honest
                peer_mat = stacked_honest - stacked_honest.mean(dim=0, keepdim=True)
                U, S, Vh = torch.linalg.svd(peer_mat, full_matrices=False)
                # Retain top 95% variance components
                var_ratio = (S ** 2) / (S ** 2).sum()
                cum_var = torch.cumsum(var_ratio, dim=0)
                k = int(torch.searchsorted(cum_var, 0.95).item()) + 1
                Vk = Vh[:k, :]  # shape (k, d)
                
                proj = torch.matmul(g_adv, Vk.T)
                recon = torch.matmul(proj, Vk)
                recon_err = torch.norm(g_adv - recon)
                loss_penalty = loss_penalty + recon_err * 0.1

            total_loss = loss_target + self.penalty_weight * loss_penalty
            total_loss.backward()
            optimizer.step()

            # Projection step: clamp g_adv within L2 ball around mean_honest
            with torch.no_grad():
                delta = g_adv - mean_honest
                delta_norm = torch.norm(delta)
                if delta_norm > self.eps * torch.norm(mean_honest):
                    delta = delta * (self.eps * torch.norm(mean_honest) / delta_norm)
                g_adv.copy_(mean_honest + delta)

        return g_adv.detach()
