"""
Numerical Verification for Chunk 4: Threshold Controller & Reputation Tracker
"""

import torch
import numpy as np
from orchestration.threshold_controller import ThresholdController
from orchestration.reputation import ReputationTracker, reputation_weighted_aggregate


def test_threshold_escalation():
    print("\n--- Test 1: Threshold Escalation ---")
    controller = ThresholdController(
        beta=0.2,
        threshold_trigger=0.15,
        escalation_step=0.05,
        decay_step=0.01,
        base_threshold=0.5,
        max_threshold=0.8,
    )
    
    # 100 clients per round
    # Rounds 1-10: 5% rejection rate
    print("Rounds 1-10: Low rejection rate (5%)")
    t1 = 0.5
    for r in range(1, 11):
        scores = torch.tensor([0.1]*5 + [0.9]*95)
        thresh = controller.update({"layer1": scores})
        t1 = thresh["layer1"]
        if r in [1, 5, 10]:
            print(f"  Round {r:2d}: layer1 threshold = {t1:.4f}")
        assert t1 == 0.5, f"Threshold should remain at base 0.5 under low rejection, got {t1}"

    # Rounds 11-20: 25% rejection rate
    print("Rounds 11-20: High rejection rate (25%)")
    for r in range(11, 21):
        scores = torch.tensor([0.1]*25 + [0.9]*75)
        thresh = controller.update({"layer1": scores})
        t1 = thresh["layer1"]
        if r in [11, 15, 20]:
            print(f"  Round {r:2d}: layer1 threshold = {t1:.4f}")
    
    assert t1 > 0.5, f"Threshold should escalate above 0.5 under high rejection, got {t1}"
    print("✓ Threshold escalation test PASSED")


def test_reputation_tracking():
    print("\n--- Test 2: Reputation Tracking ---")
    tracker = ReputationTracker(alpha=0.1, gamma=0.05, R_SS=0.85)
    client_ids = [f"client_{i}" for i in range(5)]
    
    # High acceptance (0.95) for 10 rounds
    print("Feeding high acceptance (a=0.95) for 10 rounds...")
    reps = {}
    for r in range(1, 11):
        scores = torch.tensor([0.95] * 5)
        reps = tracker.update(client_ids, scores)
    
    rep_vals = list(reps.values())
    print(f"  After 10 rounds, reputations: {[round(v, 4) for v in rep_vals]}")
    # Should converge toward 0.85 (from initial 1.0)
    for v in rep_vals:
        assert abs(v - 0.85) < 0.1, f"Reputation should approach R_SS=0.85, got {v}"
    
    # Feed client_0 low scores (0.3) for 5 rounds
    print("Feeding client_0 low acceptance (a=0.3) for 5 rounds...")
    for r in range(11, 16):
        scores = torch.tensor([0.3, 0.95, 0.95, 0.95, 0.95])
        reps = tracker.update(client_ids, scores)
    
    print(f"  After drop, client_0 rep = {reps['client_0']:.4f}, others = {reps['client_1']:.4f}")
    assert reps['client_0'] < 0.7, f"client_0 reputation should drop significantly, got {reps['client_0']}"
    assert reps['client_0'] < reps['client_1'], "client_0 reputation should be lower than honest peers"
    print("✓ Reputation tracking test PASSED")


def test_reputation_weighted_aggregate():
    print("\n--- Test 3: Reputation Weighted Aggregate ---")
    g1 = torch.tensor([1.0, 2.0])
    g2 = torch.tensor([3.0, 4.0])
    reps = torch.tensor([1.0, 0.0])
    
    g_agg = reputation_weighted_aggregate([g1, g2], reps, ["c1", "c2"])
    assert torch.allclose(g_agg, g1), f"Expected g1 since c2 has 0 weight, got {g_agg}"
    print("✓ Reputation weighted aggregate test PASSED")


def test_threshold_empty_round():
    print("\n--- Test 4: Threshold Controller Empty Round ---")
    controller = ThresholdController()
    # Update with empty scores
    thresh = controller.update({"layer1": torch.tensor([])})
    assert thresh["layer1"] == 0.5, "Threshold should remain base on empty round"
    # Next normal round should work without NaN
    thresh = controller.update({"layer1": torch.ones(10)})
    assert not np.isnan(thresh["layer1"]), "Threshold should not be NaN after empty round"
    print("✓ Threshold empty round test PASSED")


def test_reputation_decay_and_streak_cutoff():
    print("\n--- Test 5: Reputation Decay and K-streak Cutoff ---")
    tracker = ReputationTracker(alpha=0.1, gamma=0.05, R_SS=0.85, min_reputation=0.1, K=5)
    client_ids = ["attacker"]
    
    # Feed 0 acceptance scores for many rounds
    reps = []
    for r in range(1, 50):
        res = tracker.update(client_ids, torch.tensor([0.0]))
        reps.append(res["attacker"])
    
    # Reputation should decay toward 0 (not stuck at 0.28)
    assert reps[-1] < 0.05, f"Reputation should decay near 0, got {reps[-1]}"
    # Effective weight should be 0.0 due to K=5 streak below min_reputation=0.1
    assert tracker.get_weight("attacker") == 0.0, "Attacker weight should be forced to 0 after K rounds"
    print(f"  Final reputation: {reps[-1]:.6f}, Effective weight: {tracker.get_weight('attacker')}")
    print("✓ Reputation decay and streak cutoff test PASSED")


if __name__ == "__main__":
    test_threshold_escalation()
    test_reputation_tracking()
    test_reputation_weighted_aggregate()
    test_threshold_empty_round()
    test_reputation_decay_and_streak_cutoff()
    print("\n==========================================")
    print("ALL CHUNK 4 ORCHESTRATION TESTS PASSED ✓")
    print("==========================================")
