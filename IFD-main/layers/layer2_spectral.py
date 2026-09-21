import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class Layer2Spectral:
    """
    Layer 2 gradient filter based on spectral LOO anomaly detection.

    Refactored to solve O(N^3 d) computational scaling via Full Gram Matrix pre-computation
    and closed-form LOO submatrix slicing, reducing runtime complexity to O(N^2*d + N^4).
    # TODO: A true O(N^3) fix would require a rank-1 LOO eigenvalue downdate instead of N independent eigh() calls.
    Supports empirical threshold calibration and optional log-transformed error normalization
    to correct right-skewed Chi-distributed FPR miscalibration.

    Args:
        gamma: Variance threshold for selecting top k components (default 0.95)
        epsilon: Small constant for numerical stability (default 1e-8)
        z_thresh: Calibrated z-score threshold for sigmoid score mapping (default 3.5 for standard, 2.2 for log-transform)
        use_log_transform: If True, log-transform reconstruction errors before MAD normalization
                           to transform Chi-distributed residuals into symmetric near-Gaussian scores (default False)
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

        if N == 1:
            return torch.ones(1, device=device, dtype=dtype), torch.ones(
                1, device=device, dtype=dtype
            )

        if N == 2:
            return torch.ones(2, device=device, dtype=dtype), torch.ones(
                2, device=device, dtype=dtype
            )

        reconstruction_errors = torch.zeros(N, device=device, dtype=dtype)

        # ---- Step 1: Pre-compute Full Gram Matrix (O(N^2 d) once) ----
        K_full = gradients @ gradients.T  # shape: (N, N)
        row_sums = K_full.sum(dim=1)      # shape: (N,)
        total_sum = K_full.sum()          # scalar

        # ---- Step 2: Leave-One-Out (LOO) via Closed-Form Submatrix Slicing ----
        for i in range(N):
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[i] = False
            idx_peers = torch.where(mask)[0]  # shape: (N-1,)

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

        # ---- Step 3: Robust MAD Normalization & Scoring ----
        if self.use_log_transform:
            scores_to_norm = torch.log(reconstruction_errors + self.epsilon)
        else:
            scores_to_norm = reconstruction_errors

        median_e = torch.median(scores_to_norm)
        abs_deviations = torch.abs(scores_to_norm - median_e)
        MAD = torch.median(abs_deviations)

        if MAD < self.epsilon:
            return torch.ones(N, device=device, dtype=dtype), torch.ones(
                N, device=device, dtype=dtype
            )

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