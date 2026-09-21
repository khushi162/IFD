"""
Test Layer 2: Spectral LOO Anomaly Detection

Verifies the mathematical specification and expected behavior:
1. Inlier test: honest gradients should have high a2 scores (mean > 0.9)
2. Outlier test: adversarial gradients should have low a2 scores (< 0.3)
3. Edge cases: N=1, N=2, zero-variance peer matrices
"""

import torch
import numpy as np
from layers.layer2_spectral import Layer2Spectral


def test_inliers():
    """Test that honest gradients from same distribution get high a2 scores."""
    print("\n=== TEST 1: INLIER TEST ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # Generate 50 honest gradients from N(μ, 0.05²I)
    d = 100  # gradient dimension
    mu = torch.randn(d) * 2.0  # mean vector
    std = 0.05
    
    n_honest = 50
    gradients = []
    for _ in range(n_honest):
        g = mu + torch.randn(d) * std
        gradients.append(g)
    
    gradients_tensor = torch.stack(gradients)
    
    # Score with Layer 2
    a2, c2 = layer2.score(gradients_tensor)
    
    print(f"Number of gradients: {n_honest}")
    print(f"a2 scores - mean: {a2.mean().item():.4f}, std: {a2.std().item():.4f}")
    print(f"a2 scores - min: {a2.min().item():.4f}, max: {a2.max().item():.4f}")
    print(f"c2 scores - mean: {c2.mean().item():.4f}, std: {c2.std().item():.4f}")
    
    # Check: mean(a2) > 0.9
    mean_a2 = a2.mean().item()
    passed = mean_a2 > 0.9
    
    print(f"\n✓ PASS: mean(a2) = {mean_a2:.4f} > 0.9" if passed else f"✗ FAIL: mean(a2) = {mean_a2:.4f} ≤ 0.9")
    
    return passed


def test_outliers():
    """Test that adversarial gradients orthogonal to honest ones get low a2 scores."""
    print("\n=== TEST 2: OUTLIER TEST ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # Generate honest and adversarial gradients
    d = 100
    mu_honest = torch.randn(d) * 2.0
    mu_honest = mu_honest / torch.norm(mu_honest)  # normalize
    
    # Adversarial mean orthogonal to honest mean
    mu_adv = torch.randn(d) * 2.0
    mu_adv = mu_adv - (mu_adv @ mu_honest) * mu_honest  # Gram-Schmidt orthogonalization
    mu_adv = mu_adv / torch.norm(mu_adv) * 3.0  # normalize and scale
    
    n_honest = 50
    n_adv = 5
    
    gradients = []
    labels = []
    
    # Honest gradients: N(μ_honest, 0.05²I)
    for _ in range(n_honest):
        g = mu_honest + torch.randn(d) * 0.05
        gradients.append(g)
        labels.append('honest')
    
    # Adversarial gradients: N(μ_adv, 0.1²I)
    for _ in range(n_adv):
        g = mu_adv + torch.randn(d) * 0.1
        gradients.append(g)
        labels.append('adversarial')
    
    gradients_tensor = torch.stack(gradients)
    
    # Score with Layer 2
    a2, c2 = layer2.score(gradients_tensor)
    
    # Split scores by type
    a2_honest = a2[:n_honest]
    a2_adv = a2[n_honest:]
    
    print(f"Number of honest gradients: {n_honest}")
    print(f"Number of adversarial gradients: {n_adv}")
    print(f"\nHonest a2 scores - mean: {a2_honest.mean().item():.4f}, std: {a2_honest.std().item():.4f}")
    print(f"Adversarial a2 scores - mean: {a2_adv.mean().item():.4f}, std: {a2_adv.std().item():.4f}")
    print(f"Adversarial a2 scores - min: {a2_adv.min().item():.4f}, max: {a2_adv.max().item():.4f}")
    
    # Check: all adversarial a2 < 0.3
    passed = (a2_adv < 0.3).all().item()
    max_adv_a2 = a2_adv.max().item()
    
    print(f"\n✓ PASS: all adversarial a2 < 0.3 (max={max_adv_a2:.4f})" if passed 
          else f"✗ FAIL: some adversarial a2 ≥ 0.3 (max={max_adv_a2:.4f})")
    
    return passed


def test_edge_case_n1():
    """Test N=1 edge case: should return a2=1.0, c2=1.0."""
    print("\n=== TEST 3: EDGE CASE N=1 ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # Single gradient
    g = torch.randn(100)
    gradients = g.unsqueeze(0)
    
    a2, c2 = layer2.score(gradients)
    
    print(f"Input shape: {gradients.shape}")
    print(f"a2: {a2.item():.4f}")
    print(f"c2: {c2.item():.4f}")
    
    passed = (a2.item() == 1.0) and (c2.item() == 1.0)
    
    print(f"\n✓ PASS: a2=1.0, c2=1.0" if passed else f"✗ FAIL: expected a2=1.0, c2=1.0")
    
    return passed


def test_edge_case_n2():
    """Test N=2 edge case: should return a2=1.0, c2=1.0."""
    print("\n=== TEST 4: EDGE CASE N=2 ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # Two gradients
    g1 = torch.randn(100)
    g2 = torch.randn(100)
    gradients = torch.stack([g1, g2])
    
    a2, c2 = layer2.score(gradients)
    
    print(f"Input shape: {gradients.shape}")
    print(f"a2: {a2}")
    print(f"c2: {c2}")
    
    passed = (a2 == 1.0).all().item() and (c2 == 1.0).all().item()
    
    print(f"\n✓ PASS: all a2=1.0, all c2=1.0" if passed else f"✗ FAIL: expected all a2=1.0, all c2=1.0")
    
    return passed


def test_edge_case_zero_variance():
    """Test zero-variance peer matrix: all gradients identical."""
    print("\n=== TEST 5: EDGE CASE ZERO VARIANCE ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # All gradients identical
    g = torch.randn(100)
    n = 10
    gradients = g.unsqueeze(0).expand(n, -1).clone()
    
    a2, c2 = layer2.score(gradients)
    
    print(f"Input shape: {gradients.shape}")
    print(f"All gradients identical: {torch.allclose(gradients[0], gradients[1])}")
    print(f"a2 scores: {a2}")
    print(f"c2 scores: {c2}")
    
    passed = (a2 == 1.0).all().item()
    
    print(f"\n✓ PASS: all a2=1.0" if passed else f"✗ FAIL: expected all a2=1.0")
    
    return passed


def test_edge_case_near_zero_variance():
    """Test near-zero variance: gradients very similar."""
    print("\n=== TEST 6: EDGE CASE NEAR-ZERO VARIANCE ===")
    
    layer2 = Layer2Spectral(gamma=0.95)
    
    # Very similar gradients (tiny noise)
    g_base = torch.randn(100)
    n = 20
    gradients = []
    for _ in range(n):
        g = g_base + torch.randn(100) * 1e-10  # extremely small noise
        gradients.append(g)
    gradients = torch.stack(gradients)
    
    a2, c2 = layer2.score(gradients)
    
    print(f"Input shape: {gradients.shape}")
    print(f"Gradient variance: {gradients.var().item():.2e}")
    print(f"a2 scores - mean: {a2.mean().item():.4f}, std: {a2.std().item():.4f}")
    print(f"a2 scores - min: {a2.min().item():.4f}, max: {a2.max().item():.4f}")
    
    # Should handle gracefully, all a2 should be high (near 1.0)
    passed = a2.mean().item() > 0.95
    
    print(f"\n✓ PASS: mean(a2) > 0.95" if passed else f"✗ FAIL: mean(a2) ≤ 0.95")
    
    return passed


def main():
    """Run all tests and report results."""
    print("=" * 70)
    print("LAYER 2 SPECTRAL LOO VERIFICATION TESTS")
    print("=" * 70)
    
    results = []
    
    # Core functionality tests
    results.append(("Inlier test", test_inliers()))
    results.append(("Outlier test", test_outliers()))
    
    # Edge case tests
    results.append(("Edge case N=1", test_edge_case_n1()))
    results.append(("Edge case N=2", test_edge_case_n2()))
    results.append(("Edge case zero variance", test_edge_case_zero_variance()))
    results.append(("Edge case near-zero variance", test_edge_case_near_zero_variance()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
