"""
Layer 3: Dual-Channel Temporal CUSUM Consistency Scoring (CORRECTED)

Two defects addressed here (see AUDIT_NOTES.md items #7 and #J):

1. Score floor bug (slope too shallow):
   The per-round CUSUM increment d = 1 - cos(...) is bounded in [0, 2]
   (maximized at cos = -1, the most extreme possible single-round attack).
   From a clean state (S=0), the ORIGINAL default slope=2.0 gives:

       S_global = 2 - 0.30 = 1.70  ->  a3_global = 1 - sigmoid(2*(1.70-1.2))
                                      = 1 - sigmoid(1.0) = 0.2689
       S_local  = 2 - 0.36 = 1.64  ->  a3_local  = 1 - sigmoid(2*(1.64-1.2))
                                      = 1 - sigmoid(0.88) = 0.2932

   So a3 >= 0.2689 after ANY single round, for ANY attack -- there is a hard
   mathematical floor on how low a3 can go in one round, and it sits only
   ~0.03 below the layer's own 0.3 test threshold. A small hyperparameter
   change (e.g. h_global: 1.2 -> 1.5) would push the floor above the
   detection threshold and silently disable single-round detection.

   Fix: raise `slope` so a single maximally-adversarial round can actually
   approach 0, giving real margin instead of ~0.03. A slope of 6.0 gives a
   single-round floor of 1 - sigmoid(6*0.5) = 0.047.

2. Hardcoded, uncalibrated baselines (mu0_local, mu0_global):
   The original mu0 values (0.33, 0.28) are fixed constants never estimated
   from the actual data distribution. Every existing test uses gradients
   with cos ~= 1.0 (d ~= 0), two orders of magnitude below the allowance, so
   the CUSUM sits pinned at zero and tests pass regardless of whether mu0 is
   well-calibrated. Under real non-IID (Dirichlet) client data, typical
   honest round-to-round cosine deviation may be much higher than these
   constants assume, which would cause every honest client to accumulate
   CUSUM linearly and eventually be flagged.

   Fix: add an optional warmup-calibration mode that estimates mu0_local and
   mu0_global from the observed median deviation over the first
   `warmup_rounds` rounds, before CUSUM accumulation begins. Fixed values
   remain the default for backward compatibility, but calibrated values
   should be used for any reported experiment.

The dual-channel CUSUM recursion itself (tabular CUSUM: S = max(0, S + d -
(mu0+k))) was audited and found mathematically standard and correctly
implemented; it is unchanged here.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


class Layer3Temporal:
    """
    Dual-channel temporal consistency tracker for federated learning
    gradients (or gradient deltas -- see AUDIT_NOTES.md item #3 for why
    absolute weights are unsuitable input for anomaly scoring).

    Combines a local trajectory channel (CUSUM on deviation from the
    client's own EMA) with a global consensus channel (CUSUM on deviation
    from the robust cross-client median).

    Args:
        alpha: Base EMA decay constant for local trajectory tracking.
        maturity_rounds: Rounds required for full confidence.
        mu0_local: Expected honest local cosine-deviation baseline. Ignored
            if `warmup_rounds > 0` (baseline is estimated instead).
        k_local: Local CUSUM slack / allowance parameter.
        h_local: Local CUSUM decision threshold.
        mu0_global: Expected honest global cosine-deviation baseline.
            Ignored if `warmup_rounds > 0`.
        k_global: Global CUSUM slack / allowance parameter.
        h_global: Global CUSUM decision threshold.
        slope: Sigmoid scaling factor for continuous anomaly scoring.
            CORRECTED default (was 2.0): a slope of 2.0 leaves only ~0.03
            of margin between the mathematical single-round score floor
            (0.2689) and the standard 0.3 detection threshold. 6.0 gives a
            single-round floor of ~0.047.
        warmup_rounds: If > 0, mu0_local/mu0_global are estimated from the
            median observed deviation over this many initial rounds instead
            of using the fixed constants above. CUSUM accumulation for a
            client begins only after its own warmup completes.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        maturity_rounds: int = 20,
        mu0_local: float = 0.33,
        k_local: float = 0.03,
        h_local: float = 1.2,
        mu0_global: float = 0.28,
        k_global: float = 0.02,
        h_global: float = 1.2,
        slope: float = 6.0,
        warmup_rounds: int = 0,
    ):
        self.alpha = alpha
        self.maturity_rounds = maturity_rounds
        self.mu0_local = mu0_local
        self.k_local = k_local
        self.h_local = h_local
        self.mu0_global = mu0_global
        self.k_global = k_global
        self.h_global = h_global
        self.slope = slope
        self.warmup_rounds = warmup_rounds

        # Per-client state: {client_id: {...}}
        self.client_state: Dict[Any, Dict[str, Any]] = {}

    def score(
        self,
        gradients: Union[List[torch.Tensor], torch.Tensor],
        client_ids: Union[List, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score current round gradients for temporal and global consistency.

        Args:
            gradients: List of 1D tensors or 2D tensor (N, d)
            client_ids: List/array of client identifiers, length N

        Returns:
            a3: Anomaly scores [0, 1], shape (N,)
            c3: Confidence scores [0, 1], shape (N,)
        """
        if isinstance(gradients, list):
            gradients = torch.stack([g.flatten() for g in gradients])
        elif gradients.dim() == 1:
            gradients = gradients.unsqueeze(0)

        if isinstance(client_ids, torch.Tensor):
            client_ids = client_ids.tolist()

        # CORRECTED: validate shapes match before indexing, rather than
        # silently producing NaN-filled output on mismatch.
        if gradients.shape[0] != len(client_ids):
            raise ValueError(
                f"Layer3Temporal.score: gradients has {gradients.shape[0]} "
                f"rows but {len(client_ids)} client_ids were provided."
            )

        N = len(client_ids)
        device = gradients.device

        a3 = torch.full((N,), float('nan'), device=device)
        c3 = torch.zeros(N, device=device)

        if N > 1:
            ref_global = torch.median(gradients, dim=0).values
        else:
            ref_global = None

        for idx, client_id in enumerate(client_ids):
            g_current = gradients[idx]

            if client_id not in self.client_state:
                self.client_state[client_id] = {
                    "trajectory": g_current.clone(),
                    "cusum_local": 0.0,
                    "cusum_global": 0.0,
                    "rounds_seen": 1,
                    "warmup_d_local": [],
                    "warmup_d_global": [],
                    "mu0_local": self.mu0_local,
                    "mu0_global": self.mu0_global,
                }
                a3[idx] = 1.0
                c3[idx] = 1.0 / self.maturity_rounds
                continue

            state = self.client_state[client_id]
            rounds_seen = state["rounds_seen"] + 1
            state["rounds_seen"] = rounds_seen
            c3[idx] = min(rounds_seen / self.maturity_rounds, 1.0)

            # --- 1. Local Trajectory Channel ---
            trajectory_prev = state["trajectory"]
            cos_local = self._cosine_similarity(g_current, trajectory_prev)
            d_local = 1.0 - cos_local

            in_warmup = self.warmup_rounds > 0 and rounds_seen <= self.warmup_rounds
            if in_warmup:
                state["warmup_d_local"].append(d_local)
                if rounds_seen == self.warmup_rounds:
                    # Warmup complete: calibrate mu0_local from observed
                    # median honest deviation instead of the fixed constant.
                    state["mu0_local"] = float(
                        torch.median(torch.tensor(state["warmup_d_local"])).item()
                    )
                a3_local = 1.0  # no anomaly scoring during warmup
            else:
                s_local = max(
                    0.0,
                    state["cusum_local"] + d_local - (state["mu0_local"] + self.k_local),
                )
                state["cusum_local"] = s_local
                a3_local = 1.0 - 1.0 / (1.0 + math.exp(-self.slope * (s_local - self.h_local)))

            alpha_eff = self.alpha * a3_local
            state["trajectory"] = (1.0 - alpha_eff) * trajectory_prev + alpha_eff * g_current

            # --- 2. Global Consensus Channel ---
            if N > 1 and ref_global is not None:
                cos_global = self._cosine_similarity(g_current, ref_global)
                d_global = 1.0 - cos_global

                if in_warmup:
                    state["warmup_d_global"].append(d_global)
                    if rounds_seen == self.warmup_rounds:
                        state["mu0_global"] = float(
                            torch.median(torch.tensor(state["warmup_d_global"])).item()
                        )
                    a3_global = 1.0
                else:
                    s_global = max(
                        0.0,
                        state["cusum_global"] + d_global - (state["mu0_global"] + self.k_global),
                    )
                    state["cusum_global"] = s_global
                    a3_global = 1.0 - 1.0 / (1.0 + math.exp(-self.slope * (s_global - self.h_global)))
            else:
                # N == 1: no peers, global channel is a deliberate no-op.
                a3_global = 1.0

            a3[idx] = min(a3_local, a3_global)

        return a3, c3

    def _cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        norm_a = torch.norm(a, p=2)
        norm_b = torch.norm(b, p=2)

        if norm_a.item() < eps or norm_b.item() < eps:
            return 0.0

        dot_product = torch.dot(a, b)
        denominator = norm_a * norm_b + eps
        cosine = dot_product / denominator
        return max(-1.0, min(1.0, cosine.item()))

    def reset_client(self, client_id: Any) -> None:
        if client_id in self.client_state:
            del self.client_state[client_id]

    def get_client_state(self, client_id: Any) -> Union[Dict[str, Any], None]:
        return self.client_state.get(client_id, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_single_round_floor_has_real_margin():
    """
    Regression test for the slope defect: with the corrected slope, a
    single maximally-adversarial round (cos = -1, the worst case) must
    score well below the 0.3 detection threshold, not just barely below it.
    """
    print("=" * 70)
    print("Test: single-round adversarial floor has real margin")
    print("=" * 70)

    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    dim = 50
    mu = torch.randn(dim)
    client_ids = [f"client_{i}" for i in range(5)]

    # Round 1: establish trajectory/history.
    G1 = torch.stack([mu + 0.05 * torch.randn(dim) for _ in range(5)])
    layer3.score(G1, client_ids)

    # Round 2: client 0 flips to the maximally adversarial direction (cos ~= -1).
    G2 = torch.stack(
        [-mu + 0.01 * torch.randn(dim)] + [mu + 0.05 * torch.randn(dim) for _ in range(4)]
    )
    a3, c3 = layer3.score(G2, client_ids)

    print(f"  Attacker a3 after single extreme round: {a3[0].item():.4f}")
    print(f"  (mathematical floor under slope=6.0 is ~0.047; must have real margin below 0.3)")

    assert a3[0].item() < 0.15, (
        f"Attacker a3={a3[0].item():.4f} too close to the 0.3 threshold -- "
        f"insufficient margin under a single extreme round"
    )
    print("  PASS")
    print("=" * 70)


def _test_warmup_calibration():
    """Verify warmup-mode baseline calibration runs and produces a finite mu0."""
    print("\n" + "=" * 70)
    print("Test: warmup calibration estimates mu0 from data")
    print("=" * 70)

    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20, warmup_rounds=5)
    dim = 50
    mu = torch.randn(dim)
    client_ids = ["client_0"]

    for r in range(6):
        g = mu + 0.2 * torch.randn(dim)  # noisier than the module defaults
        a3, c3 = layer3.score([g], client_ids)

    state = layer3.get_client_state("client_0")
    print(f"  Calibrated mu0_local: {state['mu0_local']:.4f} (was fixed at 0.33)")
    assert 0.0 <= state["mu0_local"] < 1.0
    print("  PASS")
    print("=" * 70)


def test_consistency_honest_clients():
    print("\n" + "=" * 70)
    print("TEST: Honest Clients Consistency")
    print("=" * 70)

    layer3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
    n_clients, n_rounds, dim = 10, 30, 100
    mu = torch.randn(dim) * 10.0
    client_ids = [f"client_{i}" for i in range(n_clients)]

    all_a3_after_warmup = []
    for round_idx in range(1, n_rounds + 1):
        gradients = torch.stack([mu + torch.randn(dim) * 0.05 for _ in range(n_clients)])
        a3, c3 = layer3.score(gradients, client_ids)
        if round_idx >= 5:
            all_a3_after_warmup.extend(a3.tolist())

    mean_a3 = sum(all_a3_after_warmup) / len(all_a3_after_warmup)
    print(f"  mean(a3) rounds 5-30: {mean_a3:.4f} (expect > 0.9)")
    assert mean_a3 > 0.9
    print("  PASS")


def test_input_length_mismatch_raises():
    print("\n" + "=" * 70)
    print("TEST: mismatched gradients/client_ids raises instead of NaN")
    print("=" * 70)

    layer3 = Layer3Temporal()
    try:
        layer3.score(torch.randn(3, 10), ["a", "b"])
        raised = False
    except ValueError:
        raised = True

    assert raised, "Expected ValueError on length mismatch"
    print("  PASS")


if __name__ == "__main__":
    _test_single_round_floor_has_real_margin()
    _test_warmup_calibration()
    test_consistency_honest_clients()
    test_input_length_mismatch_raises()
    print("\nAll Layer 3 tests passed.")
