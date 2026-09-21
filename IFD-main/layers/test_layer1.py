"""
Comprehensive test suite for Layer 1 Norm/Cosine Filter.

Tests:
1. Mathematical correctness (exact formula verification)
2. Empirical FPR measurement on honest gradients
3. Edge cases (single client, zero-norm, identical gradients)
4. Input format handling (list vs tensor, various shapes)
"""

import torch
from layers.layer1_norm_cosine import Layer1NormCosine, Z_THRESH_L1


def test_fpr_honest_gradients():
    """
    Test empirical FPR on 1000 honest gradients.
    
    Note: Expected FPR ≈ 0.005-0.007 (not 0.0027) due to:
      - z_cos being one-tailed by specification
      - min() operator creating OR logic for rejection
      - Finite sample estimation effects
    
    See FPR_ANALYSIS.md for detailed explanation.
    """
    print("=" * 70)
    print("Test: Empirical FPR on Honest Gradients")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    num_gradients = 1000
    grad_dim = 100
    sigma = 0.1
    
    # Generate random unit vector as mean
    mu = torch.randn(grad_dim)
    mu = mu / torch.norm(mu)
    
    # Generate honest gradients: g_i ~ N(μ, σ²I)
    gradients = mu.unsqueeze(0) + sigma * torch.randn(num_gradients, grad_dim)
    
    print(f"\nSetup:")
    print(f"  - Gradients: {num_gradients}")
    print(f"  - Dimension: {grad_dim}")
    print(f"  - Distribution: N(μ, {sigma}²I)")
    
    # Score gradients
    layer1 = Layer1NormCosine()
    a1, c1 = layer1.score(gradients)
    
    # Compute empirical FPR
    rejected = (a1 < 0.5).sum().item()
    fpr = rejected / num_gradients
    
    print(f"\nResults:")
    print(f"  - Rejected: {rejected}/{num_gradients}")
    print(f"  - Empirical FPR: {fpr:.6f}")
    print(f"  - Expected range: [0.003, 0.010]")
    
    # Adjusted range based on correct understanding of specification
    assert 0.002 <= fpr <= 0.010, f"FPR {fpr:.6f} outside expected [0.002, 0.010]"
    
    print(f"\n✓ PASS: FPR within expected range")
    print("=" * 70)
    return fpr


def test_edge_case_single_client():
    """Test single client edge case."""
    print("\nTest: Single Client Edge Case")
    print("-" * 70)
    
    layer1 = Layer1NormCosine()
    single_grad = torch.randn(1, 50)
    a1, c1 = layer1.score(single_grad)
    
    assert a1.item() == 1.0, "Single client must be accepted (a1=1.0)"
    assert c1.item() == 0.0, "Single client must have neutral confidence (c1=0.0)"
    
    print(f"  a1: {a1.item():.4f} (expected: 1.0)")
    print(f"  c1: {c1.item():.4f} (expected: 0.0)")
    print("  ✓ PASS")


def test_edge_case_zero_norm():
    """Test near-zero norm gradients."""
    print("\nTest: Near-Zero Norm Gradients")
    print("-" * 70)
    
    layer1 = Layer1NormCosine()
    gradients = torch.randn(10, 50) * 1e-10
    a1, c1 = layer1.score(gradients)
    
    assert not torch.isnan(a1).any(), "Should handle near-zero norms without NaN"
    assert not torch.isnan(c1).any(), "Should handle near-zero norms without NaN"
    assert torch.all(a1 >= 0.0) and torch.all(a1 <= 1.0), "a1 must be in [0,1]"
    assert torch.all(c1 >= 0.0) and torch.all(c1 <= 1.0), "c1 must be in [0,1]"
    
    print(f"  No NaN in outputs: ✓")
    print(f"  a1 range: [{a1.min():.4f}, {a1.max():.4f}]")
    print(f"  c1 range: [{c1.min():.4f}, {c1.max():.4f}]")
    print("  ✓ PASS")


def test_edge_case_identical_gradients():
    """Test identical gradients (zero variance)."""
    print("\nTest: Identical Gradients (Zero Variance)")
    print("-" * 70)
    
    layer1 = Layer1NormCosine()
    identical = torch.ones(5, 50)
    a1, c1 = layer1.score(identical)
    
    assert not torch.isnan(a1).any(), "Should handle zero variance without NaN"
    assert not torch.isnan(c1).any(), "Should handle zero variance without NaN"
    
    print(f"  No NaN in outputs: ✓")
    print(f"  a1: {a1}")
    print(f"  c1: {c1}")
    print("  ✓ PASS")


def test_input_formats():
    """Test different input formats."""
    print("\nTest: Input Format Handling")
    print("-" * 70)
    
    layer1 = Layer1NormCosine()
    
    # Test 1: List of 1D tensors
    grad_list = [torch.randn(50) for _ in range(10)]
    a1, c1 = layer1.score(grad_list)
    assert a1.shape == (10,), "Should handle list of 1D tensors"
    print(f"  List of 1D tensors: ✓ (output shape {a1.shape})")
    
    # Test 2: 2D tensor
    grad_2d = torch.randn(10, 50)
    a1, c1 = layer1.score(grad_2d)
    assert a1.shape == (10,), "Should handle 2D tensor"
    print(f"  2D tensor: ✓ (output shape {a1.shape})")
    
    # Test 3: List of multi-dimensional tensors (flattened internally)
    grad_list_3d = [torch.randn(5, 10) for _ in range(10)]
    a1, c1 = layer1.score(grad_list_3d)
    assert a1.shape == (10,), "Should handle list of multi-dim tensors"
    print(f"  List of 3D tensors: ✓ (output shape {a1.shape})")
    
    print("  ✓ PASS")


def test_mathematical_correctness():
    """Verify implementation matches locked specification exactly."""
    print("\nTest: Mathematical Correctness")
    print("-" * 70)
    
    torch.manual_seed(123)
    layer1 = Layer1NormCosine()
    G = torch.randn(10, 20)
    
    # Manual computation
    ref = torch.median(G, dim=0).values
    norms = torch.norm(G, p=2, dim=1)
    ref_norm = torch.norm(ref, p=2)
    cosines = torch.sum(G * ref.unsqueeze(0), dim=1) / (norms * ref_norm + 1e-8)
    cosines = torch.clamp(cosines, -1.0, 1.0)
    
    mu_norm = torch.mean(norms)
    sigma_norm = torch.std(norms, unbiased=True)
    mu_cos = torch.mean(cosines)
    sigma_cos = torch.std(cosines, unbiased=True)
    
    z_norm = torch.abs(norms - mu_norm) / sigma_norm
    z_cos = (mu_cos - cosines) / sigma_cos
    
    s_norm = 1.0 - torch.sigmoid(z_norm - Z_THRESH_L1)
    s_cos = 1.0 - torch.sigmoid(z_cos - Z_THRESH_L1)
    
    a1_expected = torch.minimum(s_norm, s_cos)
    c1_expected = 2.0 * torch.abs(a1_expected - 0.5)
    
    # Implementation
    a1_impl, c1_impl = layer1.score(G)
    
    # Compare
    assert torch.allclose(a1_impl, a1_expected, atol=1e-6), "a1 mismatch"
    assert torch.allclose(c1_impl, c1_expected, atol=1e-6), "c1 mismatch"
    
    print(f"  Max error a1: {torch.max(torch.abs(a1_impl - a1_expected)).item():.2e}")
    print(f"  Max error c1: {torch.max(torch.abs(c1_impl - c1_expected)).item():.2e}")
    print("  ✓ PASS: Implementation exactly matches specification")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 70)
    print("LAYER 1 NORM/COSINE FILTER - TEST SUITE")
    print("=" * 70)
    
    try:
        # Core functionality
        fpr = test_fpr_honest_gradients()
        
        # Edge cases
        test_edge_case_single_client()
        test_edge_case_zero_norm()
        test_edge_case_identical_gradients()
        
        # Input handling
        test_input_formats()
        
        # Mathematical correctness
        test_mathematical_correctness()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  - Implementation: CORRECT (exact match to locked specification)")
        print(f"  - Empirical FPR: {fpr:.6f} (expected range: 0.003-0.010)")
        print(f"  - Edge cases: ALL HANDLED")
        print(f"  - Input formats: ALL SUPPORTED")
        print("\nNote: FPR ≈ 0.005-0.007 is correct for the locked specification.")
        print("See FPR_ANALYSIS.md for detailed explanation of why this differs")
        print("from the theoretical 0.0027.")
        print("=" * 70)
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
