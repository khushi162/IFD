"""
Numerical Verification for Chunk 6: Attacks A1, A2, A3
"""

import math
import torch
import numpy as np

from attacks.a1_oracle_whitebox import OracleWhiteBoxPGD
from attacks.a2_grinding import TemporalGrinding
from attacks.a3_spectral_matching import SpectralMatching
from layers.layer1_norm_cosine import Layer1NormCosine
from layers.layer2_spectral import Layer2Spectral
from layers.layer3_temporal import Layer3Temporal


def test_a1_oracle_whitebox():
    print("\n--- Test: A1 Oracle White-Box PGD ---")
    d = 50
    N = 10
    honest_grads = [torch.randn(d) * 0.1 + 1.0 for _ in range(N)]
    target_dir = torch.randn(d)

    l1 = Layer1NormCosine()
    l2 = Layer2Spectral()

    attacker = OracleWhiteBoxPGD(eps=0.3, num_steps=20)
    g_adv = attacker.attack(honest_grads, target_dir, layer1=l1, layer2=l2)

    assert g_adv.shape == (d,), f"Shape mismatch: {g_adv.shape}"
    cos_target = (torch.sum(g_adv * target_dir) / (torch.norm(g_adv) * torch.norm(target_dir))).item()
    print(f"Crafted A1 gradient cosine with target direction: {cos_target:.4f}")
    assert cos_target > 0.0, "A1 attack should achieve positive cosine alignment with target"
    print("✓ A1 Oracle White-Box PGD test PASSED")


def test_a2_temporal_grinding():
    print("\n--- Test: A2 Temporal Grinding ---")
    d = 50
    base_grad = torch.ones(d)
    target_dir = -torch.ones(d)  # Opposite direction

    attacker = TemporalGrinding(drift_angle_deg=5.0, warmup_rounds=10)

    prev_angle = 0.0
    for r in range(1, 15):
        g_adv = attacker.generate_gradient(round_num=r, base_gradient=base_grad, target_direction=target_dir)
        unit_g = g_adv / torch.norm(g_adv)
        unit_b = base_grad / torch.norm(base_grad)
        angle_deg = math.degrees(torch.acos(torch.clamp(torch.sum(unit_g * unit_b), -1.0, 1.0)).item())

        if r in [1, 5, 9, 10, 11]:
            print(f"  Round {r:2d}: angle from base = {angle_deg:.2f}°")

        if r < 10:
            assert angle_deg >= prev_angle - 1e-4, "Angle should increase during warmup"
            prev_angle = angle_deg
        elif r >= 10:
            # Post warmup: angle should be 180 degrees (full target flip)
            assert abs(angle_deg - 180.0) < 1e-2, f"Expected 180° post warmup, got {angle_deg:.2f}°"

    print("✓ A2 Temporal Grinding test PASSED")


def test_a3_spectral_matching():
    print("\n--- Test: A3 Spectral Matching ---")
    d = 100
    N = 20
    mu = torch.randn(d)
    # Peer subspace defined by 5 basis vectors
    basis = torch.randn(5, d)
    honest_grads = [mu + torch.matmul(torch.randn(5), basis) * 0.1 for _ in range(N)]
    target_dir = torch.randn(d)

    attacker = SpectralMatching(gamma=0.95)
    g_adv = attacker.generate_gradient(peer_gradients=honest_grads, target_direction=target_dir)

    l2 = Layer2Spectral(gamma=0.95)
    all_grads = honest_grads + [g_adv]
    a2, c2 = l2.score(all_grads)

    adv_score = a2[-1].item()
    print(f"Layer 2 acceptance score for Spectral Matching gradient: {adv_score:.4f}")
    assert adv_score > 0.5, f"Spectral matching gradient should pass Layer 2 (a2 > 0.5), got {adv_score}"
    print("✓ A3 Spectral Matching test PASSED")


if __name__ == "__main__":
    test_a1_oracle_whitebox()
    test_a2_temporal_grinding()
    test_a3_spectral_matching()
    print("\n==========================================")
    print("ALL CHUNK 6 ATTACK TESTS PASSED ✓")
    print("==========================================")
