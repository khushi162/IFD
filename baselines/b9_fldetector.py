"""
B9: FLDetector [Zhang et al., CCS 2022]
"""
import torch
from typing import List, Dict, Optional


class FLDetector:
    """
    FLDetector: Model update consistency tracker based on historical predictions.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.history: Dict[str, torch.Tensor] = {}

    def aggregate(self, gradients: List[torch.Tensor], client_ids: Optional[List[str]] = None) -> torch.Tensor:
        stacked = torch.stack([g.flatten().float() for g in gradients])
        N, d = stacked.shape

        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(N)]

        weights = torch.ones(N)

        for i, cid in enumerate(client_ids):
            gi = stacked[i]
            if cid in self.history:
                prev_gi = self.history[cid]
                cos_sim = torch.sum(gi * prev_gi) / (torch.norm(gi) * torch.norm(prev_gi) + 1e-8)
                if cos_sim < 0.0:  # Direction inversion anomaly
                    weights[i] = 0.1
            self.history[cid] = gi.detach().clone()

        total_weight = weights.sum() + 1e-8
        return (stacked * weights.unsqueeze(1)).sum(dim=0) / total_weight
