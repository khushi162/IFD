"""
Layer 3: Dual-Channel Temporal CUSUM Consistency Scoring

Tracks per-client gradient trajectory using an anomaly-gated EMA (local channel)
and per-round robust global consensus deviation (global channel) with tabular CUSUM.

Mathematical Specification:
1. Local channel:
   d_local = 1 - cos(g_i^(t), trajectory_i^(t-1))
   S_local = max(0, S_local^(t-1) + d_local - (mu0_local + k_local))
   a3_local = 1 - sigmoid(slope * (S_local - h_local))
   alpha_eff = alpha * a3_local
   trajectory_i^(t) = (1 - alpha_eff) * trajectory_i^(t-1) + alpha_eff * g_i^(t)

2. Global channel:
   d_global = 1 - cos(g_i^(t), median_j(g_j^(t)))  (no-op if N==1)
   S_global = max(0, S_global^(t-1) + d_global - (mu0_global + k_global))
   a3_global = 1 - sigmoid(slope * (S_global - h_global))

3. Combined score & confidence:
   a3 = min(a3_local, a3_global)
   c3 = min(rounds_seen_i / maturity_rounds, 1.0)
"""

import math
from typing import Any, Dict, List, Tuple, Union
import torch


class Layer3Temporal:
    """
    Dual-channel temporal consistency tracker for federated learning gradients.
    
    Combines a fast local trajectory channel (CUSUM on deviation from client's own EMA)
    with an independent global consensus channel (CUSUM on deviation from robust median).
    
    Args:
        alpha: Base EMA decay constant for local trajectory tracking (default: 0.1)
        maturity_rounds: Rounds required for full confidence (default: 20)
        mu0_local: Expected honest local cosine deviation baseline (calibrated: 0.33)
        k_local: Local CUSUM slack / allowance parameter (calibrated: 0.03)
        h_local: Local CUSUM decision threshold (calibrated: 1.2)
        mu0_global: Expected honest global cosine deviation baseline (calibrated: 0.28)
        k_global: Global CUSUM slack / allowance parameter (calibrated: 0.02)
        h_global: Global CUSUM decision threshold (calibrated: 1.2)
        slope: Sigmoid scaling factor for continuous anomaly scoring (default: 2.0)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        maturity_rounds: int = 20,
        mu0_local: float = 0.33,
        k_local: float = 0.03,
        h_local: float = 1.2,
        mu0_global: float = 0.28,
        k_global: float = 0.02,
        h_global: float = 1.2,
        slope: float = 2.0,
    ):
        self.alpha = alpha
        self.maturity_rounds = maturity_rounds
        self.mu0_local = mu0_local
        self.k_local = k_local
        self.h_local = h_local
        self.mu0_global = mu0_global
        self.k_global = k_global
        self.h_global = h_global
        self.slope = slope

        # Per-client state: {client_id: {'trajectory': Tensor, 'cusum_local': float, 'cusum_global': float, 'rounds_seen': int}}
        self.client_state: Dict[Any, Dict[str, Any]] = {}

    def score(
        self,
        gradients: Union[List[torch.Tensor], torch.Tensor],
        client_ids: Union[List, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score current round gradients for temporal and global consistency.

        Args:
            gradients: List of 1D tensors or 2D tensor (N, d)
            client_ids: List/array of client identifiers

        Returns:
            a3: Anomaly scores [0, 1], shape (N,)
            c3: Confidence scores [0, 1], shape (N,)
        """
        # Normalize inputs
        if isinstance(gradients, list):
            gradients = torch.stack([g.flatten() for g in gradients])
        elif gradients.dim() == 1:
            gradients = gradients.unsqueeze(0)

        if isinstance(client_ids, torch.Tensor):
            client_ids = client_ids.tolist()

        N = len(client_ids)
        device = gradients.device

        a3 = torch.full((N,), float('nan'), device=device)
        c3 = torch.zeros(N, device=device)

        # Global consensus reference (coordinate-wise median)
        if N > 1:
            ref_global = torch.median(gradients, dim=0).values
        else:
            ref_global = None

        for idx, client_id in enumerate(client_ids):
            g_current = gradients[idx]

            if client_id not in self.client_state:
                # First round: initialize trajectory with current gradient
                self.client_state[client_id] = {
                    "trajectory": g_current.clone(),
                    "cusum_local": 0.0,
                    "cusum_global": 0.0,
                    "rounds_seen": 1,
                }
                # First round: perfect self-similarity → a3 = 1.0
                a3[idx] = 1.0
                c3[idx] = 1.0 / self.maturity_rounds
            else:
                state = self.client_state[client_id]
                rounds_seen = state["rounds_seen"] + 1
                state["rounds_seen"] = rounds_seen
                c3[idx] = min(rounds_seen / self.maturity_rounds, 1.0)

                # --- 1. Local Trajectory Channel ---
                trajectory_prev = state["trajectory"]
                cos_local = self._cosine_similarity(g_current, trajectory_prev)
                # Deviation from client's own history
                d_local = 1.0 - cos_local
                s_local = max(0.0, state["cusum_local"] + d_local - (self.mu0_local + self.k_local))
                state["cusum_local"] = s_local

                # Continuous anomaly score for local channel
                a3_local = 1.0 - 1.0 / (1.0 + math.exp(-self.slope * (s_local - self.h_local)))

                # Anomaly-gated EMA update: slow trajectory adaptation when anomalous
                alpha_eff = self.alpha * a3_local
                state["trajectory"] = (1.0 - alpha_eff) * trajectory_prev + alpha_eff * g_current

                # --- 2. Global Consensus Channel ---
                # N == 1 case: with no peers, the global channel is intentionally a no-op (always accepted)
                if N > 1 and ref_global is not None:
                    cos_global = self._cosine_similarity(g_current, ref_global)
                    d_global = 1.0 - cos_global
                    s_global = max(0.0, state["cusum_global"] + d_global - (self.mu0_global + self.k_global))
                    state["cusum_global"] = s_global
                    a3_global = 1.0 - 1.0 / (1.0 + math.exp(-self.slope * (s_global - self.h_global)))
                else:
                    a3_global = 1.0

                # Combined acceptance score: min(a3_local, a3_global)
                a3[idx] = min(a3_local, a3_global)

        return a3, c3

    def _cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute cosine similarity: (a · b) / (‖a‖₂ · ‖b‖₂ + ε)
        """
        norm_a = torch.norm(a, p=2)
        norm_b = torch.norm(b, p=2)

        if norm_a.item() < eps or norm_b.item() < eps:
            return 0.0

        dot_product = torch.dot(a, b)
        denominator = norm_a * norm_b + eps
        cosine = dot_product / denominator
        return max(-1.0, min(1.0, cosine.item()))

    def reset_client(self, client_id: Any) -> None:
        """Remove a client's state (for testing or client removal)."""
        if client_id in self.client_state:
            del self.client_state[client_id]

    def get_client_state(self, client_id: Any) -> Union[Dict[str, Any], None]:
        """Get a client's current state (for inspection/debugging)."""
        return self.client_state.get(client_id, None)
