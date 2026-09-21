"""
Layer 2: Spectral LOO Anomaly Detection (CORRECTED)

Original defect (see AUDIT_NOTES.md item #8):
    When the median absolute deviation (MAD) of reconstruction errors
    collapsed to (near) zero -- which happens whenever >=50% of clients
    produce near-identical residuals, including via deliberate collusion --
    the layer returned a2 = 1.0 for EVERY client unconditionally:

        if MAD < self.epsilon:
            return torch.ones(N), torch.ones(N)

    This is a fail-open path: a colluding subset that equalizes its
    residuals disables the layer entirely, for itself AND for every honest
    client in the same round, with the returned confidence (c2 = 1.0)
    falsely signalling certainty.

Fix:
    Fall back to a secondary scale estimate (sample std) when MAD collapses,
    and only fall back to "no signal" (uniform accept) when BOTH MAD and std
    collapse -- i.e. when the reconstruction errors are genuinely all
    identical, in which case there is no information to distinguish clients
    on and treating them as low-confidence (c2 = 0.0) rather than
    high-confidence is the honest description of that state. This also
    aligns Layer 2's confidence convention with Layer 1 and Layer 3, where
    c = 0 means "no information."

The closed-form Gram-matrix / LOO submatrix-slicing algebra in the original
implementation (Steps 1-2 below) was audited and found correct -- it is
unchanged here.
"""

import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class Layer2Spectral:
    """
    Layer 2 gradient filter based on spectral LOO anomaly detection.

    Uses full Gram matrix pre-computation and closed-form LOO submatrix
    slicing to reduce runtime complexity to O(N^2*d + N^4) (N independent
    eigh() calls on (N-1)x(N-1) matrices remain the dominant term; a true
    O(N^3) implementation would require a rank-1 LOO eigenvalue downdate
    rather than independent eigh() calls per client -- not implemented
    here).

    Args:
        gamma: Variance threshold for selecting top-k components (default 0.95)
        epsilon: Small constant for numerical stability (default 1e-8)
        z_thresh: Calibrated z-score threshold for sigmoid score mapping
            (default 3.5 for standard, 2.2 for log-transform)
        use_log_transform: If True, log-transform reconstruction errors
            before MAD normalization, to make right-skewed Chi-distributed
            residuals closer to symmetric/Gaussian before scoring (default False)
    """

    def __init__(
        self,
        gamma: float = 0.95,
        epsilon: float = 1e-8,
        z_thresh: Optional[float] = None,
        use_log_transform: bool = False,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_log_transform = use_log_transform

        if z_thresh is None:
            self.z_thresh = 3.5 if not use_log_transform else 2.2
        else:
            self.z_thresh = z_thresh
            if use_log_transform and z_thresh > 3.0:
                warnings.warn(
                    f"z_thresh={z_thresh} with use_log_transform=True has been "
                    f"measured to produce a near-zero rejection rate."
                )

    def score(
        self, gradients: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(gradients, list):
            gradients = torch.stack([g.flatten() for g in gradients])
        elif gradients.dim() == 1:
            gradients = gradients.unsqueeze(0)

        N, d = gradients.shape
        device = gradients.device
        dtype = gradients.dtype

        # Edge cases: too few clients for a meaningful peer comparison.
        # Confidence is 0, not 1 -- there is no information to score from,
        # not certainty that the client is honest.
        if N == 1:
            return (
                torch.ones(1, device=device, dtype=dtype),
                torch.zeros(1, device=device, dtype=dtype),
            )
        if N == 2:
            return (
                torch.ones(2, device=device, dtype=dtype),
                torch.zeros(2, device=device, dtype=dtype),
            )

        reconstruction_errors = torch.zeros(N, device=device, dtype=dtype)

        # ---- Step 1: Pre-compute Full Gram Matrix (O(N^2 d) once) ----
        K_full = gradients @ gradients.T  # (N, N)
        row_sums = K_full.sum(dim=1)      # (N,)
        total_sum = K_full.sum()          # scalar

        # ---- Step 2: Leave-One-Out (LOO) via Closed-Form Submatrix Slicing ----
        for i in range(N):
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[i] = False
            idx_peers = torch.where(mask)[0]  # (N-1,)

            peer_row_sums = row_sums[idx_peers] - K_full[idx_peers, i]  # (N-1,)
            sum_K_peers = total_sum - 2.0 * row_sums[i] + K_full[i, i]
            mean_norm_sq = sum_K_peers / ((N - 1) ** 2)

            K_sub = K_full[idx_peers][:, idx_peers]
            means_outer = (peer_row_sums.unsqueeze(1) + peer_row_sums.unsqueeze(0)) / (N - 1)
            K_peer = K_sub - means_outer + mean_norm_sq

            g_i_mean_peer = (row_sums[i] - K_full[i, i]) / (N - 1)
            y_i = K_full[idx_peers, i] - g_i_mean_peer - (peer_row_sums / (N - 1)) + mean_norm_sq
            g_i_norm_sq = K_full[i, i] - 2.0 * g_i_mean_peer + mean_norm_sq

            try:
                L, U = torch.linalg.eigh(K_peer)
            except Exception:
                reconstruction_errors[i] = float('inf')
                continue

            L = torch.flip(L, dims=[0])
            U = torch.flip(U, dims=[1])

            valid_mask = L > self.epsilon
            if not valid_mask.any():
                reconstruction_errors[i] = torch.sqrt(torch.clamp(g_i_norm_sq, min=0.0))
                continue

            L_valid = L[valid_mask]
            U_valid = U[:, valid_mask]

            total_var = L_valid.sum()
            cum_var = torch.cumsum(L_valid, dim=0) / total_var
            k_idx = torch.searchsorted(cum_var, self.gamma).item() + 1
            max_k = min(N - 2, d, len(L_valid))
            k = min(k_idx, max_k)

            if k == 0:
                reconstruction_errors[i] = torch.sqrt(torch.clamp(g_i_norm_sq, min=0.0))
                continue

            L_k = L_valid[:k]
            U_k = U_valid[:, :k]

            proj_y = U_k.T @ y_i  # (k,)
            coeffs = proj_y / torch.sqrt(L_k)  # (k,)

            proj_norm_sq = torch.sum(coeffs ** 2)
            err_sq = torch.clamp(g_i_norm_sq - proj_norm_sq, min=0.0)
            reconstruction_errors[i] = torch.sqrt(err_sq)

        if torch.isinf(reconstruction_errors).any():
            finite_mask = torch.isfinite(reconstruction_errors)
            if finite_mask.any():
                max_finite = reconstruction_errors[finite_mask].max()
            else:
                max_finite = torch.tensor(1.0, device=device)
            reconstruction_errors[~finite_mask] = max_finite * 10.0

        # ---- Step 3: Robust Normalization & Scoring (CORRECTED fallback) ----
        if self.use_log_transform:
            scores_to_norm = torch.log(reconstruction_errors + self.epsilon)
        else:
            scores_to_norm = reconstruction_errors

        median_e = torch.median(scores_to_norm)
        abs_deviations = torch.abs(scores_to_norm - median_e)
        MAD = torch.median(abs_deviations)

        if MAD < self.epsilon:
            # MAD collapsed: >=50% of residuals are tied at the median. This
            # can happen via honest coincidence OR deliberate collusion that
            # equalizes residuals to disable the layer -- do not blindly
            # accept everyone. Fall back to sample std as the scale estimate.
            std_fallback = scores_to_norm.std(unbiased=True)

            if std_fallback < self.epsilon:
                # Residuals are genuinely all identical -- no dispersion
                # signal exists to detect anomalies from. Accept, but with
                # ZERO confidence (there is nothing to be confident about),
                # not full confidence as the original code returned.
                return (
                    torch.ones(N, device=device, dtype=dtype),
                    torch.zeros(N, device=device, dtype=dtype),
                )

            z = (scores_to_norm - median_e) / (std_fallback + self.epsilon)
        else:
            z = (scores_to_norm - median_e) / (1.4826 * MAD + self.epsilon)

        sigmoid_z = torch.sigmoid(z - self.z_thresh)
        s_spectral = 1.0 - sigmoid_z

        a2 = s_spectral
        c2 = 2.0 * torch.abs(a2 - 0.5)

        return a2, c2

    def __call__(
        self, gradients: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.score(gradients)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_mad_collapse_does_not_blindly_accept():
    """
    Regression test for the original defect: when >=50% of clients collude
    to equalize their reconstruction error (MAD -> 0), the layer must NOT
    unconditionally accept everyone with full confidence. It should fall
    back to a std-based estimate, or -- only if there is truly zero
    dispersion -- accept with ZERO confidence, not full confidence.
    """
    print("=" * 70)
    print("Test: MAD collapse does not blindly accept")
    print("=" * 70)

    torch.manual_seed(0)
    layer2 = Layer2Spectral(gamma=0.95)

    d = 50
    # 6 "colluding" clients share an identical vector (forces >=50% of
    # residuals to tie), plus 4 honest clients with normal spread.
    base = torch.randn(d)
    colluders = base.unsqueeze(0).expand(6, -1).clone()
    honest = base.unsqueeze(0) + torch.randn(4, d) * 0.5
    G = torch.cat([colluders, honest], dim=0)

    a2, c2 = layer2.score(G)

    print(f"  a2: {a2}")
    print(f"  c2: {c2}")

    # The key regression check: this must NOT be all-ones with all-ones
    # confidence (the original fail-open behavior).
    blindly_accepted = torch.allclose(a2, torch.ones_like(a2)) and torch.allclose(
        c2, torch.ones_like(c2)
    )
    assert not blindly_accepted, "Layer 2 fell back to blind full-acceptance"

    print("  PASS (did not blindly accept)")
    print("=" * 70)


def test_inliers():
    print("\n=== TEST: INLIER TEST ===")
    layer2 = Layer2Spectral(gamma=0.95)

    d = 100
    mu = torch.randn(d) * 2.0
    std = 0.05
    n_honest = 50
    gradients = torch.stack([mu + torch.randn(d) * std for _ in range(n_honest)])

    a2, c2 = layer2.score(gradients)
    mean_a2 = a2.mean().item()
    print(f"mean(a2) = {mean_a2:.4f} (expect > 0.9)")
    assert mean_a2 > 0.9
    print("PASS")


def test_outliers():
    print("\n=== TEST: OUTLIER TEST ===")
    layer2 = Layer2Spectral(gamma=0.95)

    d = 100
    mu_honest = torch.randn(d) * 2.0
    mu_honest = mu_honest / torch.norm(mu_honest)

    mu_adv = torch.randn(d) * 2.0
    mu_adv = mu_adv - (mu_adv @ mu_honest) * mu_honest
    mu_adv = mu_adv / torch.norm(mu_adv) * 3.0

    n_honest, n_adv = 50, 5
    gradients = [mu_honest + torch.randn(d) * 0.05 for _ in range(n_honest)]
    gradients += [mu_adv + torch.randn(d) * 0.1 for _ in range(n_adv)]
    gradients = torch.stack(gradients)

    a2, c2 = layer2.score(gradients)
    a2_adv = a2[n_honest:]
    print(f"adversarial a2 max = {a2_adv.max().item():.4f} (expect < 0.3)")
    assert (a2_adv < 0.3).all().item()
    print("PASS")


def test_edge_cases_n1_n2():
    print("\n=== TEST: EDGE CASES N=1, N=2 ===")
    layer2 = Layer2Spectral(gamma=0.95)

    a2, c2 = layer2.score(torch.randn(1, 100))
    assert a2.item() == 1.0 and c2.item() == 0.0, "N=1 should be accept, zero confidence"

    a2, c2 = layer2.score(torch.randn(2, 100))
    assert (a2 == 1.0).all().item() and (c2 == 0.0).all().item(), "N=2 should be accept, zero confidence"

    print("PASS")


if __name__ == "__main__":
    _test_mad_collapse_does_not_blindly_accept()
    test_inliers()
    test_outliers()
    test_edge_cases_n1_n2()
    print("\nAll Layer 2 tests passed.")
