"""
Test suite for Layer 3 Temporal Consistency

Verifies:
1. Honest clients maintain high a3 (mean > 0.9) after warmup
2. Inconsistent clients get low a3 (< 0.3)
3. Edge cases: first round, single client, maturity ramp-up
"""

import torch
import numpy as np
from layers.layer3_temporal import Layer3Temporal


def test_consistency_honest_clients():
    """
    Simulate 30 rounds of 10 honest clients with consistent gradients.
    Verify mean(a3) > 0.9 for rounds 5-30.
    """
    print("\n" + "="*70)
    print("TEST 1: Honest Clients Consistency")
    print("="*70)
    
    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    
    n_clients = 10
    n_rounds = 30
    dim = 100
    
    # Honest clients: gradients sampled from N(μ, 0.05²I)
    # Same mean direction, low variance → high consistency
    mu = torch.randn(dim) * 10.0  # Fixed direction
    
    client_ids = [f"client_{i}" for i in range(n_clients)]
    
    # Track one client's evolution
    track_client = "client_0"
    a3_history = []
    c3_history = []
    
    print(f"\nSimulating {n_rounds} rounds of {n_clients} honest clients")
    print(f"Gradient distribution: N(μ, 0.05²I), dim={dim}")
    print(f"\nTracking {track_client}:")
    print(f"{'Round':<8} {'a3':<10} {'c3':<10} {'Mean(a3)':<12}")
    print("-" * 45)
    
    all_a3_after_warmup = []
    
    for round_idx in range(1, n_rounds + 1):
        # Sample honest gradients: small noise around fixed direction
        gradients = []
        for _ in range(n_clients):
            noise = torch.randn(dim) * 0.05
            g = mu + noise
            gradients.append(g)
        
        gradients = torch.stack(gradients)
        a3, c3 = layer3.score(gradients, client_ids)
        
        # Track specific client
        a3_history.append(a3[0].item())
        c3_history.append(c3[0].item())
        
        mean_a3 = a3.mean().item()
        
        # After warmup (round 5+), collect for final check
        if round_idx >= 5:
            all_a3_after_warmup.extend(a3.tolist())
        
        # Print evolution
        if round_idx == 1 or round_idx % 5 == 0 or round_idx == n_rounds:
            print(f"{round_idx:<8} {a3[0].item():<10.4f} {c3[0].item():<10.4f} {mean_a3:<12.4f}")
    
    # Verify requirements
    mean_a3_warmup = np.mean(all_a3_after_warmup)
    
    print("\n" + "-" * 45)
    print(f"VERIFICATION:")
    print(f"  Round 1 expectations:")
    print(f"    c3 = 0.05 (1/20): {'✓' if abs(c3_history[0] - 0.05) < 0.01 else '✗'} (got {c3_history[0]:.4f})")
    print(f"    a3 ≈ 1.0 (self-sim): {'✓' if a3_history[0] > 0.95 else '✗'} (got {a3_history[0]:.4f})")
    print(f"  Round 20 expectations:")
    print(f"    c3 = 1.0 (mature): {'✓' if abs(c3_history[19] - 1.0) < 0.01 else '✗'} (got {c3_history[19]:.4f})")
    print(f"  Rounds 5-30 mean(a3): {mean_a3_warmup:.4f}")
    print(f"    Requirement > 0.9: {'✓ PASS' if mean_a3_warmup > 0.9 else '✗ FAIL'}")
    
    assert mean_a3_warmup > 0.9, f"Honest clients mean(a3) = {mean_a3_warmup:.4f} < 0.9"
    print("\n✓ Test PASSED: Honest clients maintain high temporal consistency")


def test_inconsistency_attack():
    """
    Inject one client with flipped gradient at round 15.
    Verify that client's a3 drops below 0.3.
    """
    print("\n" + "="*70)
    print("TEST 2: Inconsistency Detection (Direction Flip)")
    print("="*70)
    
    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    
    n_clients = 10
    n_rounds = 25
    dim = 100
    attack_round = 15
    attack_client_idx = 5
    
    mu = torch.randn(dim) * 10.0
    client_ids = [f"client_{i}" for i in range(n_clients)]
    
    print(f"\nSimulating {n_rounds} rounds, injecting attack at round {attack_round}")
    print(f"Attack: client_{attack_client_idx} gradient flips to N(-μ, 0.1²I)")
    print(f"\n{'Round':<8} {'Attacker a3':<15} {'Mean(others a3)':<18}")
    print("-" * 45)
    
    attacker_a3_at_attack = None
    
    for round_idx in range(1, n_rounds + 1):
        gradients = []
        
        for client_idx in range(n_clients):
            if round_idx == attack_round and client_idx == attack_client_idx:
                # Attack: flip direction
                noise = torch.randn(dim) * 0.1
                g = -mu + noise
            else:
                # Honest
                noise = torch.randn(dim) * 0.05
                g = mu + noise
            gradients.append(g)
        
        gradients = torch.stack(gradients)
        a3, c3 = layer3.score(gradients, client_ids)
        
        attacker_a3 = a3[attack_client_idx].item()
        others_a3 = torch.cat([a3[:attack_client_idx], a3[attack_client_idx+1:]]).mean().item()
        
        if round_idx == attack_round:
            attacker_a3_at_attack = attacker_a3
        
        # Print key rounds
        if round_idx in [1, attack_round - 1, attack_round, attack_round + 1] or round_idx % 10 == 0:
            marker = " ← ATTACK" if round_idx == attack_round else ""
            print(f"{round_idx:<8} {attacker_a3:<15.4f} {others_a3:<18.4f}{marker}")
    
    print("\n" + "-" * 45)
    print(f"VERIFICATION:")
    print(f"  Attacker a3 at round {attack_round}: {attacker_a3_at_attack:.4f}")
    print(f"    Requirement < 0.3: {'✓ PASS' if attacker_a3_at_attack is not None and attacker_a3_at_attack < 0.3 else '✗ FAIL'}")
    
    assert attacker_a3_at_attack is not None and attacker_a3_at_attack < 0.3, f"Attacker a3 = {attacker_a3_at_attack:.4f} >= 0.3"
    print("\n✓ Test PASSED: Direction flip detected with low a3")


def test_edge_cases():
    """
    Edge case verification:
    1. First round: all clients return c3=0.05, a3=1.0
    2. Single client across 5 rounds: c3 ramps, a3 stays high
    """
    print("\n" + "="*70)
    print("TEST 3: Edge Cases")
    print("="*70)
    
    # Edge case 1: First round behavior
    print("\nEdge Case 1: First Round")
    print("-" * 45)
    
    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    
    n_clients = 5
    dim = 50
    gradients = torch.randn(n_clients, dim)
    client_ids = [f"client_{i}" for i in range(n_clients)]
    
    a3, c3 = layer3.score(gradients, client_ids)
    
    print(f"First round scores (n={n_clients} clients):")
    for i in range(n_clients):
        print(f"  client_{i}: a3={a3[i].item():.4f}, c3={c3[i].item():.4f}")
    
    all_c3_correct = all(abs(c3[i].item() - 0.05) < 0.01 for i in range(n_clients))
    all_a3_correct = all(a3[i].item() == 1.0 for i in range(n_clients))
    
    print(f"\nVerification:")
    print(f"  All c3 = 0.05: {'✓' if all_c3_correct else '✗'}")
    print(f"  All a3 = 1.0: {'✓' if all_a3_correct else '✗'}")
    
    assert all_c3_correct, "First round c3 should be 0.05 for all clients"
    assert all_a3_correct, "First round a3 should be 1.0 for all clients"
    
    # Edge case 2: Single client maturity ramp
    print("\n" + "-" * 45)
    print("Edge Case 2: Single Client Maturity Ramp")
    print("-" * 45)
    
    layer3_single = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    
    client_id = "solo_client"
    mu = torch.randn(dim) * 5.0
    
    print(f"\n{'Round':<8} {'a3':<10} {'c3':<10} {'Expected c3':<15}")
    print("-" * 45)
    
    for round_idx in range(1, 6):
        noise = torch.randn(dim) * 0.05
        g = mu + noise
        
        a3, c3 = layer3_single.score([g], [client_id])
        
        expected_c3 = min(round_idx / 20.0, 1.0)
        
        print(f"{round_idx:<8} {a3[0].item():<10.4f} {c3[0].item():<10.4f} {expected_c3:<15.4f}")
        
        # Verify c3 ramps correctly
        assert abs(c3[0].item() - expected_c3) < 0.01, f"Round {round_idx} c3 mismatch"
        
        # Verify a3 stays high for consistent gradients
        if round_idx > 1:
            assert a3[0].item() > 0.8, f"Round {round_idx} a3 too low for consistent client"
    
    print("\nVerification:")
    print(f"  c3 ramps from 0.05 → 0.25: ✓")
    print(f"  a3 remains high (> 0.8): ✓")
    
    print("\n✓ Test PASSED: Edge cases verified")


def test_zero_norm_edge_case():
    """
    Edge case: zero-norm gradient.
    Should handle zero norm gracefully without NaN and return valid score in [0, 1].
    """
    print("\n" + "="*70)
    print("TEST 4: Zero-Norm Edge Case")
    print("="*70)
    
    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    
    dim = 50
    client_id = "zero_client"
    
    # First round: normal gradient
    g1 = torch.randn(dim) * 5.0
    a3, c3 = layer3.score([g1], [client_id])
    print(f"Round 1: a3={a3[0].item():.4f}, c3={c3[0].item():.4f} (normal)")
    
    # Second round: zero gradient
    g2 = torch.zeros(dim)
    a3, c3 = layer3.score([g2], [client_id])
    print(f"Round 2: a3={a3[0].item():.4f}, c3={c3[0].item():.4f} (zero gradient)")
    
    assert not torch.isnan(a3).any(), "a3 should not contain NaN for zero-norm gradient"
    assert not torch.isnan(c3).any(), "c3 should not contain NaN for zero-norm gradient"
    assert 0.0 <= a3[0].item() <= 1.0, f"a3 should be in [0, 1], got {a3[0].item()}"
    
    print("\n✓ Test PASSED: Zero-norm edge case handled without NaN")


def test_gradual_drift_grinding():
    """
    Test dual-channel CUSUM detection on a 20-round gradual drift (A2 grinding attack).
    Verifies that the attacker is detected (a3 < 0.5) during the drift.
    """
    print("\n" + "="*70)
    print("TEST 5: Gradual Drift Detection (A2 Grinding)")
    print("="*70)

    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    n_clients = 10
    dim = 100
    n_rounds = 35
    attacker_idx = 0
    client_ids = [f"client_{i}" for i in range(n_clients)]

    mu = torch.randn(dim)
    mu = mu / torch.norm(mu)
    target_dir = -mu

    attacker_detected = False
    for r in range(1, n_rounds + 1):
        grads = []
        for i in range(n_clients):
            if i == attacker_idx and r >= 10:
                # Gradual drift from round 10 to 30
                frac = min(1.0, (r - 10) / 20.0)
                g_blend = (1.0 - frac) * mu + frac * target_dir
                g = g_blend + 0.05 * torch.randn(dim)
            else:
                g = mu + 0.05 * torch.randn(dim)
            grads.append(g)
        grads = torch.stack(grads)
        a3, c3 = layer3.score(grads, client_ids)

        if a3[attacker_idx].item() < 0.5:
            attacker_detected = True

    assert attacker_detected, "Layer 3 dual-channel CUSUM should detect gradual 20-round drift"
    print("✓ Test PASSED: Gradual drift attack detected by Layer 3 dual-channel CUSUM")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Layer 3 Temporal Dual-Channel CUSUM Test Suite")
    print("="*70)
    
    try:
        test_consistency_honest_clients()
        test_inconsistency_attack()
        test_edge_cases()
        test_zero_norm_edge_case()
        test_gradual_drift_grinding()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        raise
