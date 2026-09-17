"""
Reputation Tracking and Weighted Aggregation

Maintains per-client reputation scores with EWMA updates and steady-state damping.
Provides reputation-weighted gradient aggregation for robust federated learning.

Mathematical Specification:
---------------------------
Reputation update (locked formula):
    R_i^(t+1) = (1-α)·R_i^(t) + α·a_i^(t) + γ·(R_SS − R_i^(t))

Where:
- a_i^(t) is the combined cascade acceptance score (computed by CascadeRouter)
- R_SS = 0.85 is a FIXED DESIGN CONSTANT (hardcoded, never derived)
- α = 0.1 (placeholder, flagged as needing tuning)
- γ = 0.05 (placeholder, flagged as needing tuning)

CRITICAL: R_SS = 0.85 is hardcoded. Do NOT derive R_SS from any other equation.

Aggregation:
    g_global = Σ (R_i · g_i) / Σ R_i

Default Hyperparameters (placeholders needing tuning):
------------------------------------------------------
- α = 0.1        # EWMA weight for acceptance score
- γ = 0.05       # Damping factor toward steady-state
- R_SS = 0.85    # FIXED steady-state reputation (design constant, NOT derived)
"""

import torch
from typing import Dict, List, Optional


class ReputationTracker:
    """
    Per-client reputation tracker with conditional EWMA updates and steady-state damping.
    
    Tracks reputation for each client based on their cascade acceptance scores.
    New clients are initialized at R_i = 1.0 (full trust).
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        R_SS: float = 0.85,
        min_reputation: float = 0.1,
        K: int = 5,
    ):
        """
        Initialize reputation tracker.
        
        Args:
            alpha: EWMA weight for acceptance score (placeholder)
            gamma: Damping factor toward steady-state (placeholder)
            R_SS: FIXED steady-state reputation (design constant, hardcoded: 0.85)
            min_reputation: Minimum reputation threshold for streak tracking (default: 0.1)
            K: Consecutive rounds below min_reputation before zeroing weight (default: 5)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.R_SS = R_SS  # FIXED DESIGN CONSTANT: 0.85
        self.min_reputation = min_reputation
        self.K = K
        
        # Per-client reputation state
        self.reputations: Dict[str, float] = {}
        # Per-client streak of rounds with R_i < min_reputation
        self.consecutive_low: Dict[str, int] = {}
    
    def update(
        self,
        client_ids: List[str],
        acceptance_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update reputations based on cascade acceptance scores.
        
        Args:
            client_ids: List of client identifiers
            acceptance_scores: Tensor of combined acceptance scores a_i, shape [num_clients]
                              Each score is in [0, 1]
        
        Returns:
            Dictionary mapping client_id to updated reputation
        """
        if len(client_ids) != acceptance_scores.shape[0]:
            raise ValueError(
                f"Mismatch: {len(client_ids)} client IDs but "
                f"{acceptance_scores.shape[0]} scores"
            )
        
        updated_reputations = {}
        
        for client_id, score in zip(client_ids, acceptance_scores):
            # Initialize new clients at R_i = 1.0 (full trust)
            if client_id not in self.reputations:
                self.reputations[client_id] = 1.0
                self.consecutive_low[client_id] = 0
            
            # Apply conditional reputation update formula:
            # R_i_new = R_i + alpha*(a_i - R_i) + gamma*(R_SS - R_i) * (1 if a_i > 0.5 else 0)
            R_prev = self.reputations[client_id]
            a_i = score.item() if torch.is_tensor(score) else score
            
            pull = self.gamma * (self.R_SS - R_prev) if a_i > 0.5 else 0.0
            R_new = R_prev + self.alpha * (a_i - R_prev) + pull
            
            # Clamp to [0, 1] for numerical stability
            R_new = max(0.0, min(1.0, R_new))
            
            self.reputations[client_id] = R_new
            
            # Track consecutive rounds where R_i < min_reputation
            if R_new < self.min_reputation:
                self.consecutive_low[client_id] = self.consecutive_low.get(client_id, 0) + 1
            else:
                self.consecutive_low[client_id] = 0
                
            updated_reputations[client_id] = R_new
        
        return updated_reputations
    
    def get_reputation(self, client_id: str) -> float:
        """Get current reputation for a client (returns 1.0 if new)."""
        return self.reputations.get(client_id, 1.0)
    
    def get_weight(self, client_id: str) -> float:
        """Get aggregation weight for a client (0.0 if streak >= K, else reputation)."""
        if self.consecutive_low.get(client_id, 0) >= self.K:
            return 0.0
        return self.reputations.get(client_id, 1.0)

    def get_weights(self, client_ids: List[str]) -> torch.Tensor:
        """Get aggregation weights tensor for a list of client IDs."""
        return torch.tensor([self.get_weight(cid) for cid in client_ids], dtype=torch.float32)

    def get_all_reputations(self) -> Dict[str, float]:
        """Get all tracked client reputations."""
        return self.reputations.copy()
    
    def reset(self):
        """Reset all client reputations and streaks."""
        self.reputations.clear()
        self.consecutive_low.clear()


def reputation_weighted_aggregate(
    gradients: List[torch.Tensor],
    reputations: torch.Tensor,
    client_ids: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Aggregate gradients weighted by client reputations.
    
    Formula:
        g_global = Σ (R_i · g_i) / Σ R_i
    
    Args:
        gradients: List of gradient tensors, one per client
                  Each tensor can be any shape (will be flattened for aggregation)
        reputations: Tensor of reputation scores, shape [num_clients]
        client_ids: Optional list of client IDs (for debugging/logging)
    
    Returns:
        Aggregated gradient tensor with same shape as input gradients
    """
    if len(gradients) != reputations.shape[0]:
        raise ValueError(
            f"Mismatch: {len(gradients)} gradients but "
            f"{reputations.shape[0]} reputations"
        )
    
    if len(gradients) == 0:
        raise ValueError("Cannot aggregate zero gradients")
    
    # Get reference shape from first gradient
    ref_shape = gradients[0].shape
    device = gradients[0].device
    
    # Stack all gradients (flatten if needed for weighted sum)
    # For simplicity, we'll work with the original shape and use torch operations
    stacked_grads = torch.stack(gradients, dim=0)  # [num_clients, *grad_shape]
    
    # Reshape reputations for broadcasting: [num_clients, 1, 1, ...]
    rep_shape = [reputations.shape[0]] + [1] * len(ref_shape)
    rep_weights = reputations.view(rep_shape).to(device)
    
    # Weighted sum: Σ (R_i · g_i)
    weighted_sum = (stacked_grads * rep_weights).sum(dim=0)
    
    # Normalization: / Σ R_i
    total_reputation = reputations.sum().item()
    
    if total_reputation == 0:
        # Fallback: uniform average if all reputations are zero
        aggregated = stacked_grads.mean(dim=0)
    else:
        aggregated = weighted_sum / total_reputation
    
    return aggregated
