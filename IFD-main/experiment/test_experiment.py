"""
Numerical Verification for Chunk 9: Experiment Simulation & Metrics
"""

import numpy as np
import torch

from experiment.client import FraudMLP
from experiment.metrics import compute_eval_metrics, MetricTracker
from experiment.simulation import run_simulation
from experiment.ablation import run_ablation_configs


def test_fraud_mlp():
    print("\n--- Test 1: FraudMLP Forward Pass ---")
    model = FraudMLP(input_dim=30)
    x = torch.randn(16, 30)
    out = model(x)
    assert out.shape == (16,), f"Output shape mismatch: {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Sigmoid outputs must be in [0, 1]"
    print("✓ FraudMLP test PASSED")


def test_metrics_computation():
    print("\n--- Test 2: Metrics Computation ---")
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = compute_eval_metrics(y_true, y_pred_prob)
    assert metrics["roc_auc"] > 0.8, f"ROC-AUC should be high, got {metrics['roc_auc']}"
    assert metrics["fpr"] == 0.0, f"FPR at 0.5 threshold should be 0, got {metrics['fpr']}"
    assert metrics["tpr"] == 1.0, f"TPR at 0.5 threshold should be 1, got {metrics['tpr']}"
    print(f"  Computed metrics: ROC-AUC={metrics['roc_auc']:.4f}, FPR={metrics['fpr']:.4f}, TPR={metrics['tpr']:.4f}")

    tracker = MetricTracker()
    tracker.log_round(1, metrics)
    summary = tracker.summary()
    assert "mean_roc_auc" in summary, "Tracker summary missing mean_roc_auc"
    print("✓ Metrics computation test PASSED")


def test_simulation_run():
    print("\n--- Test 3: Flower Simulation Run (2 rounds, 3 clients) ---")
    history = run_simulation(num_clients=3, num_rounds=2)
    assert history is not None, "Simulation history should not be None"
    print("✓ Simulation run test PASSED")


def test_single_class_metrics():
    print("\n--- Test 5: Single-Class Batch Metrics & Summary Robustness ---")
    y_true_all_zeros = np.zeros(20, dtype=int)
    y_pred_prob = np.random.uniform(0.1, 0.4, size=20)
    
    metrics_single = compute_eval_metrics(y_true_all_zeros, y_pred_prob)
    assert metrics_single["roc_auc"] == 0.5, f"Expected 0.5 for single-class roc_auc, got {metrics_single['roc_auc']}"
    assert not np.isnan(metrics_single["roc_auc"]), "roc_auc should not be NaN"
    
    tracker = MetricTracker()
    tracker.log_round(1, metrics_single)
    # Round 2 has valid labels
    y_true_valid = np.array([0]*10 + [1]*10)
    metrics_valid = compute_eval_metrics(y_true_valid, np.array([0.1]*10 + [0.9]*10))
    # Introduce extra metric key in round 2
    metrics_valid["extra_metric"] = 0.99
    tracker.log_round(2, metrics_valid)
    
    summary = tracker.summary()
    assert "mean_roc_auc" in summary and not np.isnan(summary["mean_roc_auc"]), "mean_roc_auc should be valid float"
    assert "mean_extra_metric" in summary, "Metric introduced in round 2 should be included in summary"
    print(f"  Summary with mixed rounds: {summary}")
    print("✓ Single-class metrics and summary robustness test PASSED")


if __name__ == "__main__":
    test_fraud_mlp()
    test_metrics_computation()
    test_single_class_metrics()
    test_simulation_run()
    test_ablation_study()
    print("\n==========================================")
    print("ALL CHUNK 9 EXPERIMENT TESTS PASSED ✓")
    print("==========================================")
