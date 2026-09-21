"""
Layer 1: Norm/Cosine Filtering (CORRECTED)

Original defect (see AUDIT_NOTES.md for full derivation):
    The original implementation computed z-scores using the SAMPLE MEAN and
    SAMPLE STANDARD DEVIATION of the N client updates themselves:

        z = |x_i - mean(x)| / std(x)

    For any finite sample of size N, the maximum possible value of this
    statistic is bounded (Laguerre-Samuelson inequality):

        z_max <= (N - 1) / sqrt(N)

    At N = 10, z_max <= 9/sqrt(10) ~= 2.846. With Z_THRESH_L1 = 3.3 (or the
    docstring-specified 3.0), it is ALGEBRAICALLY IMPOSSIBLE for any input,
    including an arbitrarily extreme outlier, to produce an acceptance score
    a1 below 0.5. Layer 1 was mathematically inert for any N <= 12.

Fix:
    Replace mean/std with MEDIAN and MEDIAN ABSOLUTE DEVIATION (MAD). The
    median/MAD z-score has no such bound: because the estimator of location
    and scale is not itself dragged toward the outlier the way mean/std are,
    a single extreme point can produce an arbitrarily large z-score even at
    small N. This is the standard robust-statistics fix for exactly this
    failure mode.

    A secondary IQR-based fallback is used when MAD collapses to (near) zero,
    which can happen when more than half of a small client set happens to
    share the same value.

Mathematical Specification (CORRECTED):
    med_norm = median(‖g_i‖₂)
    mad_norm = median(| ‖g_i‖₂ - med_norm |)
    z_norm   = | ‖g_i‖₂ - med_norm | / (1.4826 * mad_norm)

    med_cos = median(cos(g_i, ref))
    mad_cos = median(| cos(g_i, ref) - med_cos |)
    z_cos   = (med_cos - cos(g_i, ref)) / (1.4826 * mad_cos)      # one-tailed:
                                                                   # only LOW
                                                                   # cosine is
                                                                   # suspicious

    s_norm = 1 - sigmoid(z_norm - Z_THRESH_L1)
    s_cos  = 1 - sigmoid(z_cos  - Z_THRESH_L1)
    a1 = min(s_norm, s_cos)
    c1 = 2 * |a1 - 0.5|

    where ref = coordinate-wise median of all client updates, and 1.4826 is
    the standard constant that makes MAD a consistent estimator of the
    standard deviation under Gaussian data.

IMPORTANT: Z_THRESH_L1 must be recalibrated empirically at the DEPLOYED
client count (see calibrate_z_threshold() below). The original 3.3 (and the
docstring's 3.0) were derived from a 1000-sample calibration harness and
deployed at N=10 -- an entirely different regime. Do not reuse either number
without recalibrating for your actual N.
"""

import torch
from typing import List, Optional, Tuple, Union


# NOTE: this default is a placeholder. It must be recalibrated for the
# deployed client count using calibrate_z_threshold() before being used in
# any reported experiment. See AUDIT_NOTES.md item #1.
Z_THRESH_L1: float = 3.3


class Layer1NormCosine:
    """
    Layer 1 gradient filter based on robust (median/MAD) L2 norm and cosine
    similarity statistics.

    Computes per-client acceptance scores (a1) and confidence scores (c1).
    Unlike the mean/std formulation, this estimator is not bounded by the
    Laguerre-Samuelson inequality and can reject outliers at any N >= 2.
    """

    def __init__(self, epsilon: float = 1e-8, z_thresh: float = Z_THRESH_L1):
        """
        Args:
            epsilon: Small constant for numerical stability in divisions.
            z_thresh: Calibrated z-score threshold. MUST be recalibrated for
                the deployed client count -- see calibrate_z_threshold().
        """
        self.epsilon = epsilon
        self.z_thresh = z_thresh

    def __call__(
        self,
        gradients: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.score(gradients)

    def _robust_z(self, v: torch.Tensor) -> torch.Tensor:
        """
        Median/MAD z-score. Unbounded in N (unlike (x - mean)/std), so a
        single extreme outlier can be detected even at small sample sizes.

        Falls back to an IQR-based scale estimate if MAD collapses to
        (near) zero -- which happens when more than half the sample shares
        one value. Falls back to an all-zero z-score only if BOTH MAD and
        IQR collapse, in which case there is genuinely no dispersion signal
        to detect anomalies from.
        """
        med = torch.median(v)
        abs_dev = torch.abs(v - med)
        mad = torch.median(abs_dev)

        scale = 1.4826 * mad

        if scale.item() < self.epsilon:
            # MAD collapsed (>=50% of values tied at the median). Try IQR.
            q = torch.quantile(
                v, torch.tensor([0.25, 0.75], device=v.device, dtype=v.dtype)
            )
            iqr = q[1] - q[0]
            scale = iqr / 1.349  # consistency constant for Gaussian IQR

        if scale.item() < self.epsilon:
            # No dispersion signal at all (e.g. all values identical).
            # Returning zero z-scores means every point looks "typical" --
            # correct, since there is nothing to distinguish it from.
            return torch.zeros_like(v)

        return (v - med) / scale

    def score(
        self,
        gradients: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Layer 1 scores for a batch of gradients (or gradient deltas
        -- callers should pass Δw_i = w_i - w_global_prev, not absolute
        weights; see AUDIT_NOTES.md item #3 for why absolute weights
        saturate cosine similarity near +1.0 regardless of attack).

        Args:
            gradients: List of gradient tensors or 2D tensor [num_clients, grad_dim]

        Returns:
            a1: Acceptance scores [num_clients], in [0, 1]
            c1: Confidence scores [num_clients], in [0, 1]. 0 means "no
                information" (e.g. degenerate input), consistent with the
                confidence convention used by Layer 2 and Layer 3.
        """
        if isinstance(gradients, list):
            grad_list = []
            for g in gradients:
                grad_list.append(g.flatten() if g.dim() > 1 else g)
            G = torch.stack(grad_list)  # [N, D]
        else:
            G = gradients
            if G.dim() > 2:
                G = G.reshape(G.shape[0], -1)

        N, D = G.shape
        device, dtype = G.device, G.dtype

        # Edge case: single client. No peers to compare against, so accept
        # with ZERO confidence (there is no information to score against).
        if N == 1:
            return (
                torch.ones(1, device=device, dtype=dtype),
                torch.zeros(1, device=device, dtype=dtype),
            )

        # Step 1: coordinate-wise median reference
        ref = torch.median(G, dim=0).values  # [D]

        # Step 2: L2 norms
        norms = torch.norm(G, p=2, dim=1)  # [N]
        ref_norm = torch.norm(ref, p=2)

        # Step 3: cosine similarities to the reference
        dot_products = torch.sum(G * ref.unsqueeze(0), dim=1)  # [N]
        cosines = dot_products / (norms * ref_norm + self.epsilon)
        cosines = torch.clamp(cosines, -1.0, 1.0)

        # Step 4: robust z-scores (CORRECTED -- median/MAD, not mean/std)
        z_norm = torch.abs(self._robust_z(norms))
        z_cos = -self._robust_z(cosines)  # one-tailed: only low cosine is suspicious

        # Step 5: sigmoid scores
        s_norm = 1.0 - torch.sigmoid(z_norm - self.z_thresh)
        s_cos = 1.0 - torch.sigmoid(z_cos - self.z_thresh)

        # Step 6: acceptance and confidence
        a1 = torch.minimum(s_norm, s_cos)
        c1 = 2.0 * torch.abs(a1 - 0.5)

        return a1, c1


def calibrate_z_threshold(
    n_clients: int,
    n_trials: int = 20000,
    target_fpr: float = 0.0027,
    dim: int = 100,
    sigma: float = 0.1,
    seed: int = 0,
) -> float:
    """
    Empirically calibrate Z_THRESH_L1 for a SPECIFIC client count.

    This replaces the original calibration harness, which measured FPR on
    1000 i.i.d. samples and then deployed the resulting threshold at N=10 --
    a different regime where the Laguerre-Samuelson bound makes rejection
    impossible in the first place. Calibration must be performed at the
    actual deployed N.

    Returns the (1 - target_fpr) quantile of the worst-case (max) robust
    z-score observed across n_trials simulated honest rounds of n_clients
    clients each, so that under the null (all-honest) hypothesis the
    probability of a false rejection in a given round is approximately
    target_fpr.
    """
    import numpy as np

    torch.manual_seed(seed)
    layer = Layer1NormCosine(z_thresh=0.0)  # z_thresh unused for calibration

    worst_z = []
    for _ in range(n_trials):
        mu = torch.randn(dim)
        mu = mu / torch.norm(mu)
        G = mu.unsqueeze(0) + sigma * torch.randn(n_clients, dim)

        norms = torch.norm(G, p=2, dim=1)
        z_n = torch.abs(layer._robust_z(norms))
        worst_z.append(z_n.max().item())

    return float(np.quantile(worst_z, 1.0 - target_fpr))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_layer1_rejects_extreme_outlier():
    """
    Regression test for the original defect: an extreme-norm outlier at
    N=10 MUST be rejected. This test FAILS on the original mean/std
    implementation (a1 >= 0.6116 always) and PASSES on the corrected
    median/MAD implementation.
    """
    print("=" * 70)
    print("Test: Layer 1 rejects extreme-norm outlier at N=10")
    print("=" * 70)

    torch.manual_seed(0)
    layer1 = Layer1NormCosine()

    G = torch.randn(10, 1000) * 0.1
    G[0] *= 1000.0  # one client 1000x the norm of the rest

    a1, c1 = layer1.score(G)

    print(f"  Outlier a1:       {a1[0].item():.4f} (must be < 0.5)")
    print(
        f"  Honest a1 range:  [{a1[1:].min().item():.4f}, {a1[1:].max().item():.4f}] (must be >= 0.5)"
    )

    assert a1[0].item() < 0.5, (
        f"Layer 1 failed to reject outlier (a1={a1[0].item():.4f})"
    )
    assert (a1[1:] >= 0.5).all().item(), "Layer 1 false-positived on an honest client"

    print("  PASS")
    print("=" * 70)


def _test_layer1_true_update_sign_flip():
    """
    Verifies detection works on gradient DELTAS (this is what callers should
    pass -- see the module docstring and AUDIT_NOTES.md item #3). A true
    sign-flip attack, Δw* = -Δw, must be detected once cosine similarity is
    computed on deltas rather than absolute weights dominated by w_prev.
    """
    print("\n" + "=" * 70)
    print("Test: True update sign-flip detected on deltas")
    print("=" * 70)

    torch.manual_seed(1)
    layer1 = Layer1NormCosine()

    delta_w = torch.randn(1000) * 0.05  # honest per-round update
    deltas = torch.stack(
        [delta_w + torch.randn(1000) * 0.01 for _ in range(9)] + [-delta_w]
    )

    a1, c1 = layer1.score(deltas)
    print(f"  Sign-flipped client a1: {a1[-1].item():.4f} (expected < 0.5)")
    assert a1[-1].item() < 0.5, "Sign-flip on deltas should be detected"

    print("  PASS")
    print("=" * 70)


def _test_edge_cases():
    print("\n" + "=" * 70)
    print("Layer 1 Edge Cases Test")
    print("=" * 70)

    layer1 = Layer1NormCosine()

    # Single client
    print("\nTest 1: Single client")
    single_grad = torch.randn(1, 50)
    a1, c1 = layer1.score(single_grad)
    assert a1.item() == 1.0, "Single client should be fully accepted"
    assert c1.item() == 0.0, "Single client should have zero confidence (no peers)"
    print("  PASS")

    # Near-zero norm gradients
    print("\nTest 2: Near-zero norm gradients")
    gradients = torch.randn(10, 50) * 1e-10
    a1, c1 = layer1.score(gradients)
    assert not torch.isnan(a1).any() and not torch.isnan(c1).any()
    print("  PASS")

    # Identical gradients (zero variance / MAD collapse)
    print("\nTest 3: Identical gradients (MAD/IQR collapse)")
    identical = torch.ones(5, 50)
    a1, c1 = layer1.score(identical)
    assert not torch.isnan(a1).any() and not torch.isnan(c1).any()
    assert torch.allclose(a1, torch.ones_like(a1)), (
        "Identical inputs should all be typical (z=0)"
    )
    print("  PASS")

    # List input
    print("\nTest 4: List of tensors input")
    grad_list = [torch.randn(50) for _ in range(10)]
    a1, c1 = layer1.score(grad_list)
    assert a1.shape == (10,) and c1.shape == (10,)
    print("  PASS")

    # >=50% of clients tied at the median (MAD collapse, IQR fallback path)
    print("\nTest 5: Majority-tied norms (MAD collapse, IQR fallback)")
    G = torch.randn(11, 50)
    G[:6] = G[0]  # 6 of 11 clients share an identical vector
    a1, c1 = layer1.score(G)
    assert not torch.isnan(a1).any() and not torch.isnan(c1).any()
    print("  PASS")

    print("\n" + "=" * 70)
    print("All edge cases passed!")
    print("=" * 70)


if __name__ == "__main__":
    _test_layer1_rejects_extreme_outlier()
    _test_layer1_true_update_sign_flip()
    _test_edge_cases()

    print("\nRecalibrating Z_THRESH_L1 at N=10 (this may take a few seconds)...")
    z10 = calibrate_z_threshold(n_clients=10, n_trials=2000)
    print(f"Calibrated Z_THRESH_L1 for N=10, target FPR=0.0027: {z10:.4f}")
    print(
        "Use this value (recalibrated with n_trials=20000+ for the paper) "
        "instead of the placeholder default."
    )
