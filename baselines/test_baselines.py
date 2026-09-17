"""
Numerical Verification for Chunk 7: Baselines B1–B9
"""

import torch
import numpy as np

from baselines.adapter import BaselineAdapter


def test_all_baselines():
    print("\n--- Test: Baseline Defenses B1–B9 ---")
    d = 100
    N_honest = 10
    N_adv = 3

    # Generate synthetic gradients
    mu = torch.randn(d)
    honest_grads = [mu + torch.randn(d) * 0.05 for _ in range(N_honest)]
    adv_grads = [-mu * 5.0 + torch.randn(d) * 1.0 for _ in range(N_adv)]
    all_grads = honest_grads + adv_grads
    client_ids = [f"client_{i}" for i in range(len(all_grads))]

    baselines_to_test = [
        "b1_fedavg",
        "b2_krum",
        "b2_multikrum",
        "b3_median",
        "b4_trimmed_mean",
        "b5_bulyan",
        "b6_fltrust",
        "b7_foolsgold",
        "b8_dpfl",
        "b9_fldetector",
    ]

    for name in baselines_to_test:
        adapter = BaselineAdapter(name)
        g_agg = adapter.aggregate(all_grads, client_ids=client_ids)
        
        # Checks
        assert g_agg.shape == (d,), f"[{name}] Output shape mismatch: expected ({d},), got {g_agg.shape}"
        assert not torch.isnan(g_agg).any(), f"[{name}] Output contains NaN values"
        assert not torch.isinf(g_agg).any(), f"[{name}] Output contains Inf values"
        
        cos_with_mu = (torch.sum(g_agg * mu) / (torch.norm(g_agg) * torch.norm(mu))).item()
        print(f"  {name:15s} -> shape: {list(g_agg.shape)}, cos(g_agg, mu): {cos_with_mu:+.4f}")

    print("✓ All Baselines B1–B9 tests PASSED")


if __name__ == "__main__":
    test_all_baselines()
    print("\n==========================================")
    print("ALL CHUNK 7 BASELINE TESTS PASSED ✓")
    print("==========================================")
