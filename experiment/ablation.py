"""
Ablation Study Runner isolating L1, L2, L3, threshold controller, and reputation.
"""

from typing import Dict, Any
import torch
import flwr as fl

from experiment.simulation import run_simulation
from layers.layer1_norm_cosine import Layer1NormCosine
from layers.layer2_spectral import Layer2Spectral
from layers.layer3_temporal import Layer3Temporal
from orchestration.flower_strategy import CascadeRouter
from orchestration.reputation import ReputationTracker
from orchestration.threshold_controller import ThresholdController


class PassThroughLayer1(Layer1NormCosine):
    def score(self, gradients):
        N = len(gradients)
        return torch.ones(N), torch.ones(N)


def run_ablation_configs(num_clients: int = 5, num_rounds: int = 2) -> Dict[str, Any]:
    """
    Run ablation configurations:
      1. Full Cascade (L1 + L2 + L3 + Threshold + Reputation)
      2. No Layer 1 (L2 + L3)
    """
    results = {}

    # 1. Full Cascade
    print("\n--- Ablation Config 1: Full Cascade ---")
    strat_full = CascadeRouter()
    hist_full = run_simulation(num_clients=num_clients, num_rounds=num_rounds, strategy=strat_full)
    results["full_cascade"] = hist_full

    # 2. No Layer 1
    print("\n--- Ablation Config 2: No Layer 1 ---")
    strat_no_l1 = CascadeRouter(layer1=PassThroughLayer1())
    hist_no_l1 = run_simulation(num_clients=num_clients, num_rounds=num_rounds, strategy=strat_no_l1)
    results["no_layer1"] = hist_no_l1

    return results
